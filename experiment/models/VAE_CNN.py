from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from data_transformation import calculate_labels_alarm, remove_first_x_days
from utils import add_noisy_dfs, detect_change_point
from experiment_config import ContaminationType, ExperimentConfig
from models.VAE import VAE, VAEModel
from models.CNN import CNNModel

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
            nn.Dropout(0.1),    
            nn.Linear(hidden_dim, window_size)  
        )
        
    def forward(self, x):
        return self.cnn(x)

    
class VAECNNModel(CNNModel):
    """ 
    Class for VAE CNN model. It combines the VAE Encoder and the VAE CNN: 
    - the VAE Encoder built a embedding of a time series in the latent space with the encoder part.
    - the VAE CNN takes the embedding and predicts for each point of the time series if it's an anomaly or not.  

    CNN training uses all contaminated files, excepting the last one, which is used for testing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU not available, using CPU")

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
                window_size= self.config.window_size,
                model_name="VAE",
                model_params={},
                contaminants=[ContaminationType.PATHOGEN]
            )
                
        vae_model = VAEModel(config_vae)
        return vae_model


    def run_model(self, train_dataloader, val_dataloader, test_dataloader, weights, epochs, patience):
        """ 
        Trains the CNN model and evaluates it on the test set.
        The model predicts a label for each point in each window. 
        For each time step, labels are given by majority vote: it is an anomaly (-1) if more than 50% of the windows covering it predict an anomaly.

        Parameters:
        - train_dataloader: DataLoader for the training set.
        - val_dataloader: DataLoader for the validation set
        - test_dataloader: DataLoader for the test set
        - weights: tensor containing the weight for the positive class (anomalies)
        - epochs: number of epochs (maximum, may stop earlier if early stopping is triggered)
        - patience: number of epochs to wait for improvement in validation F1 score before stopping training (early stopping)

        Returns:
        - mean_results_per_time_step : a list containing the predicted labels for each time step in the test set, where -1 corresponds to an anomaly and 1 to a normal point
        """
        model = VAECNN(latent_dim=32, hidden_dim=128, window_size=self.config.window_size).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weights) # loss for binary classification
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        train_loss = []
        val_loss = []
        best_val_f1 = 0
        nb_epochs_without_improvment = 0
        node = self.config.nodes[0]

        for epoch in range(epochs):
            n_corrects_train = 0
            n_corrects_val = 0
            n_total_train = 0
            n_total_val = 0
            losses = []
            train_preds_all = []
            train_labels_all = []
            val_preds_all = []
            val_labels_all = []

            model.train()
            for _, data in enumerate(train_dataloader):
                windows, labels = data # windows shape (batch, latent_dim), labels shape (batch, window_size)
                windows = windows.to(self.device)
                labels = labels.to(self.device)

                outputs = model(windows) # outputs shape (batch, window_size)
                probs = torch.sigmoid(outputs) # Convert logits to probabilities
                preds = (probs > 0.5).float() # Threshold at 0.5 to get binary predictions 
                
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
                
                n_total_train += labels.numel()
                n_corrects_train += (preds == labels).sum().item()
                train_preds_all.append(preds.flatten().cpu().numpy())
                train_labels_all.append(labels.flatten().cpu().numpy())

            train_loss.append(np.mean(losses))
            train_preds_all = np.concatenate(train_preds_all)
            train_labels_all = np.concatenate(train_labels_all)
            train_f1 = f1_score(train_labels_all, train_preds_all, average="binary", zero_division=1)
            losses = []
            
            model.eval()
            with torch.no_grad():
                for _, data in enumerate(val_dataloader):
                    windows, labels = data # windows shape (batch, latent_dim), labels shape (batch, window_size)
                    windows = windows.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(windows) # outputs shape (batch, window_size)
                    probs = torch.sigmoid(outputs) # Convert logits to probabilities
                    preds = (probs > 0.5).float() # Threshold at 0.5 to get binary predictions 
                    
                    loss = criterion(outputs, labels)
                    losses.append(loss.item())
                    n_total_val += labels.numel()
                    n_corrects_val += (preds == labels).sum().item()
                    val_preds_all.append(preds.flatten().cpu().numpy())
                    val_labels_all.append(labels.flatten().cpu().numpy())

            val_loss.append(np.mean(losses))
            val_preds_all = np.concatenate(val_preds_all)
            val_labels_all = np.concatenate(val_labels_all)
            val_f1 = f1_score(val_labels_all, val_preds_all, average="binary", zero_division=1)
            losses = []
                
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Training F1: {train_f1:.4f}, Validation F1: {val_f1:.4f}")
            
            # Save model with best validation F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                nb_epochs_without_improvment = 0
                torch.save(model.state_dict(), f"vaecnn_{node}.pth")
                print(f" Best model saved with validation F1: {best_val_f1:.4f}")

            else: 
                nb_epochs_without_improvment += 1
                if nb_epochs_without_improvment >= patience:
                    print(f"Early stopping at epoch {epoch+1}") 
                    break


        # Load best model before evaluation
        model.load_state_dict(torch.load(f"vaecnn_{node}.pth", map_location=self.device, weights_only=True))
            
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
                windows, labels = data # windows shape (batch, latent_dim), labels shape (batch, window_size) 
                windows = windows.to(self.device)
                labels = labels.to(self.device)
                outputs = model(windows) # outputs shape (batch, window_size)              
                probs = torch.sigmoid(outputs) # Convert logits to probabilities
                preds = (probs > 0.5).float() # Threshold at 0.5 to get binary predictions 

                labels = labels.flatten().detach().cpu().numpy()
                preds = preds.flatten().detach().cpu().numpy()
                n_total += len(labels)

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
            
            # Add noise to the train data
            # clean_dfs = add_noisy_dfs(clean_dfs)
            # test_contaminated_df = contaminated_dfs[-1]
            # contaminated_dfs = add_noisy_dfs(contaminated_dfs[:-1]) + [test_contaminated_df]
            
            print(f"Calculating results for node {node}")
                        
            data_train = []
            y_train = []

            # Train data 
            encoder = None
            for df in contaminated_dfs[:-1]:
                _, z, labels, encoder = self._prepare_data(df, clean_dfs, node, encoder=encoder)
                data_train.extend(z.cpu().numpy())
                y_train.extend(labels)
            
            # Test data (on the last contaminated df)
            prepared_df_test, z, labels_test, _ = self._prepare_data(contaminated_dfs[-1], clean_dfs, node, encoder=encoder)
            data_test = z.cpu().numpy() 
            y_true = calculate_labels_alarm(prepared_df_test, self.config.contaminants[0].value, 0)

            # Turn data and y into tensors
            data_train = np.array(data_train) # shape of (number of total train elements, latent_dim)
            data_train = torch.tensor(data_train, dtype=torch.float32) # shape of (number of total train elements, latent_dim)
            data_test = torch.tensor(data_test, dtype=torch.float32) # shape of (number of total test elements, latent_dim)
            
            y_train = torch.tensor(np.array(y_train), dtype=torch.float32)
            y_test = torch.tensor(np.array(labels_test), dtype=torch.float32)

            # split into train, val and test sets
            X_train, X_val, y_train, y_val = train_test_split(data_train, y_train, test_size=0.15, random_state=42)
            
            # create DataLoaders
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            test_dataset = TensorDataset(data_test, y_test)
            train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True) # one batch = (batch_size, latent_dim)
            val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
                
            weights = self._compute_weight(y_train)
            y_pred = self.run_model(train_dataloader, val_dataloader, test_dataloader, weights, epochs=200, patience=20)
<<<<<<< HEAD
            y_pred = detect_change_point(y_pred, count_required=15)
=======
<<<<<<< HEAD
            y_pred = detect_change_point(y_pred, count_required=20)
=======
            y_pred = detect_change_point(y_pred, count_required=10)
>>>>>>> 73ae07771acab47a2ff711d0e98e2e412e110e3c
>>>>>>> 471a4cc94243346fc647d2b072e9217898d9fca8
            results[node] = {"y_pred": y_pred, "y_true": y_true}
        
        return results
            

    def _prepare_data(self, contaminated_df, clean_dfs, node, encoder = None):
        """ 
        Prepares data for training and testing the CNN model.
        It trains a VAE on clean data and uses it to produce embeddings z for the contaminated data. 

        Parameters:
        - contaminated_df : a contaminated dataframe 
        - clean_dfs: a list of clean dataframes 
        - node: the node id used 
        - encoder: the VAE encoder to use for generating embeddings. By default, a new VAE encoder will be trained on the clean data.
        
        Returns:
        - prepared_df: the contaminated dataframe after removing the first 3 days
        - z : embeddings computed by the encoder (shape (number of windows, latent_dim))
        - labels: the labels for training/testing the CNN model, where each label is a sliding window of the original labels (shape (number of windows, window_size))
        - encoder: the VAE encoder used to generate embeddings (trained or reused)
        """
        hidden_dim = 128
        latent_dim = 32

        vae_encoder = self._call_vae_model(node)
        X_train, X_test, _ = vae_encoder._prepare_data(clean_dfs, [contaminated_df])

        # Train the VAE and save the weights of the model
        if encoder is None:
            train_batches = DataLoader(X_train, batch_size=32, shuffle=True)
            test_batches = DataLoader(X_test, batch_size=32, shuffle=False)
            vae_encoder.run_model(train_batches, test_batches, epochs=1000, hidden_dim=hidden_dim, latent_dim=latent_dim, node=node)

            sample_batch = next(iter(train_batches))
            input_dim = sample_batch.shape[1]
            encoder = VAE(input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(self.device)
            encoder.load_state_dict(torch.load(f"VAE_model_{node}.pth", map_location=self.device, weights_only=True))
            encoder.eval()

        # Generate z with the VAE encoder
        with torch.no_grad():
            X_test = X_test.to(self.device)
            z = encoder.encode(X_test)
            z = z.cpu()

        # Generate labels 
        prepared_df = remove_first_x_days(contaminated_df, 3) 
        _ , labels = self.create_labeled_features(prepared_df, self.config.disinfectant.value, self.config.contaminants[0].value, window_size=self.config.window_size)

        return prepared_df, z, labels, encoder  
