from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from data_transformation import remove_first_x_days
from experiment_config import ContaminationType, ExperimentConfig
from models.VAE import VAE, VAEModel
from models.CNN import CNNModel

class VAEEncoder(VAE):
    def encode(self, x):
        """
        Encodes input data into the latent space.

        Parameters:
        - x: input tensor (shape (batch_size, input_dim))

        Returns:
        - z: embedding in the latent space (shape (batch_size, latent_dim))
        """

        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)
        return z  
    
class VAEEncoderModel(VAEModel):
    def _get_name_model(self):
        return VAEEncoder

class VAECNN(nn.Module):
    """ The CNN takes the embedding space and predicts for each point of the initial time series if it's an anomaly or not. """
    def __init__(self, latent_dim, hidden_dim, window_size):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),      
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),     
            nn.Linear(hidden_dim, window_size)  
        )
        
    def forward(self, x):
        return self.cnn(x)

    
class VAECNNModel(CNNModel):
    """ 
    Class for VAE CNN model. It combines the VAE Encoder and the VAE CNN: 
    - the VAE Encoder built a embedding of a time series in the latent space with the encoder part.
    - the VAE CNN takes the embedding and predicts for each point of the time series if it's an anomaly or not.  
    """

    def _call_vae_model(self, node):
        """
        Calls the VAE model used to generate embeddings used for the CNN.
   
        Parameters:
        - node: the node id 
        
        Returns:
        - vae_model: an instantiated vae model 
        """
                
        config_vae = ExperimentConfig(
                config_name="VAE_ENCODER",
                contaminated_files=self.config.contaminated_files,
                example_files=self.config.example_files,
                nodes=[node],
                window_size=400,
                model_name="VAE_ENCODER",
                model_params={},
                contaminants=[ContaminationType.PATHOGEN]
            )
                
        vae_model = VAEEncoderModel(config_vae)
        return vae_model


    def run_model(self, train_dataloader, val_dataloader, test_dataloader, weights, epochs=10):
        """ 
        Trains the CNN model and evaluates it on the test set.
        The model predicts a label for each point in each window. 
        For each time step, labels are given by majority vote: it is an anomaly (-1) if more than 50% of the windows covering it predict an anomaly.

        Parameters:
        - train_dataloader: DataLoader for the training set.
        - val_dataloader: DataLoader for the validation set
        - test_dataloader: DataLoader for the test set
        - weights: tensor containing the weight for the positive class (anomalies)
        - epochs: number of epochs 
   
        Returns:
        - mean_results_per_time_step : a list containing the predicted labels for each time step in the test set, where -1 corresponds to an anomaly and 1 to a normal point
        """
        model = VAECNN(input_size=self._get_input_size())

        criterion = nn.BCEWithLogitsLoss(pos_weight=weights) # loss for binary classification
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        train_loss = []
        val_loss = []
        for epoch in range(epochs):
            n_corrects_train = 0
            n_corrects_val = 0
            n_total_train = 0
            n_total_val = 0
            losses = []
            model.train()
            for _, data in enumerate(train_dataloader):
                windows, labels = data # windows shape (batch, window_size, number of features), labels shape (batch, window_size)
 
                outputs = model(windows) # outputs shape (batch, 1, window_size)

                probs = torch.sigmoid(outputs) # Convert logits to probabilities

                preds = (probs > 0.5).float() # Threshold at 0.5 to get binary predictions 
                
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
                
                n_total_train += labels.numel()
                n_corrects_train += (preds == labels).sum().item()
            train_loss.append(np.mean(losses))
            losses = []
            
            model.eval()
            with torch.no_grad():
                for _, data in enumerate(val_dataloader):
                    windows, labels = data # windows shape (batch, window_size, number of features), labels shape (batch, window_size)
    
                    outputs = model(windows) # outputs shape (batch, 1, window_size)
                    
                    probs = torch.sigmoid(outputs) # Convert logits to probabilities

                    preds = (probs > 0.5).float() # Threshold at 0.5 to get binary predictions 
                    
                    loss = criterion(outputs, labels)
                    losses.append(loss.item())
                    n_total_val += labels.numel()
                    n_corrects_val += (preds == labels).sum().item()
            val_loss.append(np.mean(losses))
            losses = []
                
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Training Accuracy: {n_corrects_train/n_total_train:.4f}, Validation Accuracy: {n_corrects_val/n_total_val:.4f}")
            
            
        plt.figure()
        plt.plot(train_loss, label="train")
        plt.plot(val_loss, label="validation")
        plt.title("Loss evolution over epochs")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.show()
        
        model.eval()
        n_corrects = 0
        n_total = 0
        f1_scores = []
        recall_scores = []
        
        results_per_time_step = [[0, 0] for _ in range(len(test_dataloader.dataset) + self.config.window_size)]
        with torch.no_grad():
            i = 0
            for _, data in enumerate(test_dataloader):
                windows, labels = data # windows shape (batch, window_size, number of features), labels shape (batch, window_size)
 
                outputs = model(windows) # outputs shape (batch, 1, window_size                
                probs = torch.sigmoid(outputs) # Convert logits to probabilities

                preds = (probs > 0.5).float() # Threshold at 0.5 to get binary predictions 
                

                labels = labels.flatten()
                n_total += len(labels)
                preds = preds.flatten()
                
                j = 0
                for element in preds:
                    results_per_time_step[i+j][0] += int(element) # add the predicted label (0 or 1) to the first element of the list corresponding to the time step i+j
                    results_per_time_step[i+j][1] += 1
                    j += 1
                    
                i += 1
                
                n_corrects += (preds == labels).sum().item()
                f1 = f1_score(labels, preds, average="binary", zero_division=1)
                f1_scores.append(f1)
                recall = recall_score(labels, preds, average="binary", zero_division=1)
                recall_scores.append(recall)
            
            print(f"Final Accuracy: {n_corrects/n_total:.4f}")
            print(f"Final F1 Score: {np.mean(f1_scores):.4f}")
            print(f"Final Recall Score: {np.mean(recall_scores):.4f}")
            
            # get the mean predicted label for each time step across all windows, and label as anomaly (-1) if the mean is greater than 0.5 and normal (1) otherwise
            mean_results_per_time_step = []
            for element, count in results_per_time_step:
                new_element = element / count if count > 0 else 0
                if new_element > 0.5:
                    mean_results_per_time_step.append(-1)
                else:
                    mean_results_per_time_step.append(1)

            return mean_results_per_time_step
    

    def get_results(self):
        results = {}
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()
        
        for node, contaminated_dfs in all_contaminated_dfs.items():
            clean_dfs = all_clean_dfs[node]
            
            print(f"Calculating results for node {node}")

        return results
            

    def _prepare_data(self, contaminated_df, clean_dfs, node):
        """ 
        Prepares data for training and testing the CNN model.
        The encoder of the VAE produces 
        
        Parameters:
        - contaminated_df : a contaminated dataframe 
        - clean_dfs: a list of clean dataframes 
        - node: the node id used 
        
        Returns:
        - prepared_df: the contaminated dataframe after removing the first 3 days
        - z : embeddings computed by the encoder (shape (number of windows, latent_dim))
        - labels: the labels for training/testing the CNN model, where each label is a sliding window of the original labels (shape (number of windows, window_size))
        """
        # Train the VAE and save the weights of the model
        vae_encoder = self._call_vae_model(node)
        X_train, X_test, _ = vae_encoder._prepare_data(clean_dfs, [contaminated_df])
        train_batches = DataLoader(X_train, batch_size=32, shuffle=True)
        test_batches = DataLoader(X_test, batch_size=32, shuffle=False)
        vae_encoder.run_model(train_batches, test_batches, epochs=300, hidden_dim=64, latent_dim=4, node = node)
            
        # Generate z with the VAE encoder 
        sample_batch = next(iter(train_batches))
        input_dim = sample_batch.shape[1]
        encoder = VAEEncoder(input_dim, hidden_dim=64, latent_dim=4)
        encoder.load_state_dict(torch.load(f"VAE_model_{node}.pth"))
        encoder.eval()  
        with torch.no_grad(): 
            z = encoder.encode(X_test)
        
        # Generate labels 
        prepared_df = remove_first_x_days(contaminated_df, 3) 
        _ , labels = self.create_labeled_features(prepared_df, self.config.disinfectant.value, self.config.contaminants[0].value, window_size=self.config.window_size)

        return prepared_df, z, labels
