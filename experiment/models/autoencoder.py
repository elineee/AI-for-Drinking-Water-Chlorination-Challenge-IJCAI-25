from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from utils import plot_prediction, build_timestamps
from models.model import AnomalyModel

# Source: https://www.geeksforgeeks.org/deep-learning/implementing-an-autoencoder-in-pytorch/
# Source: https://www.datacamp.com/tutorial/introduction-to-autoencoders
# Source: https://keras.io/examples/timeseries/timeseries_anomaly_detection/
class Autoencoder(nn.Module):
    """ Class for the Autoencoder module"""
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    
class AutoencoderModel(AnomalyModel):
    """ Class for Autoencoder model"""
    
    def _get_threshold_multiplier(self):
        return 1.5


    def _prepare_data(self, clean_dfs: list[pd.DataFrame], contaminated_dfs: list[pd.DataFrame]):
        """
        Prepares train/test tensors from dataframes for the Autoencoder.
        
        Parameters: 
        - clean_dfs: dataframes with training data (clean data)
        - contaminated_dfs: dataframes with testing data (contamined data)

        Returns: 
        - tensor with the normalized training data (clean data)
        - tensor with the normalized testing data (contaminated data)
        - prepared_contaminated_dfs: contaminated dataframes after preprocessing
        """

        _, X_train = self._prepare_dataset(clean_dfs, feature_type="extended")
        prepared_contaminated_dfs , X_test = self._prepare_dataset(contaminated_dfs, feature_type="extended")

        # Normalize data with train mean and train std 
        train_mean = X_train.mean(axis=0)
        train_std = X_train.std(axis=0)
        train_std[train_std == 0] = 1

        X_train = (X_train - train_mean) / train_std
        X_test = (X_test - train_mean) / train_std

        return (
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(X_test, dtype=torch.float32),
            prepared_contaminated_dfs
        )

    def run_model(self, train_batches : torch.Tensor, test_batches : torch.Tensor, epochs : int , latent_dim : int):
        """ 
        Trains the Autoencoder on the training data and detects anomalies on the test data.
        
        Parameters:
        - train_batches: the training data in batches (clean data)
        - test_batches: the test data in batches (contaminated data)
        - epochs: the number of epochs to train the model 
        - latent_dim: the dimension of the latent space of the Autoencoder 

        Returns:
        - anomalies: a numpy array of boolean values indicating whether each test sample is an anomaly (True) or not (False)
        - test_reconstruction: the reconstructed test data from the Autoencoder
        - test_error: the reconstruction error for each test sample
        """
        torch.manual_seed(42)   
        sample_batch = next(iter(train_batches))
        input_dim = sample_batch.shape[1]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = Autoencoder(input_dim, latent_dim)
        model = model.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)

        threshold_std = 1/ np.sqrt(latent_dim) # heuristic
        latent_stds = []

        # Training 
        model.train()
        
        for epoch in range(epochs):

            epoch_losses = []

            for batch in train_batches:
                batch = batch.to(device)
                optimizer.zero_grad()
                train_reconstruction = model(batch)
                train_loss = criterion(train_reconstruction, batch)
                train_loss.backward()
                optimizer.step()
                epoch_losses.append(train_loss.item())

            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}, Loss: {np.mean(epoch_losses):.4f}")
                
            # Compute latent stds
            with torch.no_grad(): 
                latent_values = []

                for batch in train_batches:
                    batch = batch.to(device)
                    embeddings = model.encoder(batch)
                    latent_values.append(embeddings)

                embeddings = torch.cat(latent_values)
                mean_std = embeddings.std(dim=0).mean().item()
                latent_stds.append(mean_std)

        # Plot latent stds 
        plt.figure(figsize=(10, 4))
        plt.plot(latent_stds, label="Mean std of embeddings")
        plt.axhline(y=threshold_std, color='r', linestyle='--', label=f"Threshold std = {threshold_std:.3f}")
        plt.xlabel("Epoch")
        plt.ylabel("Mean std")
        plt.title("Latent stds during training")
        plt.legend()
        plt.show()

        model.eval()

        with torch.no_grad():

            # Computes the threshold 
            
            train_errors = []

            for batch in train_batches:

                batch = batch.to(device)
                train_reconstruction = model(batch)
                error = torch.mean((train_reconstruction - batch) ** 2, dim=1)
                train_errors.append(error)

            train_error = torch.cat(train_errors)
            threshold = train_error.mean() + self._get_threshold_multiplier() * train_error.std()

            # Anomaly detection with the threshold  
            test_errors = []
            test_reconstructions = []

            for batch in test_batches:

                batch = batch.to(device)
                test_reconstruction = model(batch)
                error = torch.mean((test_reconstruction - batch) ** 2, dim=1)
                test_errors.append(error)
                test_reconstructions.append(test_reconstruction)

            test_error = torch.cat(test_errors)
            test_reconstruction = torch.cat(test_reconstructions)
            anomalies = test_error > threshold
            
            return (
                anomalies.cpu().numpy(),
                test_reconstruction.cpu().numpy(),
                test_error.cpu().numpy()
            )        
        

    def get_results(self):
        results = {}
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()

        for node, clean_dfs in all_clean_dfs.items():

            contaminated_dfs = all_contaminated_dfs[node]
            X_train, X_test, prepared_contaminated_dfs = self._prepare_data(clean_dfs, contaminated_dfs)
            prepared_contaminated_df = pd.concat(prepared_contaminated_dfs)

            # Anomaly detection
            # TODO : handle multiple contaminants, for now only one contaminant is handled
            y_true = self._calculate_labels(prepared_contaminated_df, self.config.contaminants[0].value, self.config.window_size)            

            train_batches = DataLoader(X_train, batch_size=128, shuffle=True)
            test_batches = DataLoader(X_test, batch_size=128, shuffle=False)

            anomalies, reconstructions, test_error = self.run_model( train_batches, test_batches, epochs=300, latent_dim=4)

            y_pred = np.where(anomalies, -1, 1)  
            y_pred = self._post_predictions(y_pred)
            results[node] = {"y_true": y_true, "y_pred": y_pred}
            
            # Plot the reconstruction of the disinfectant value
            test_timestamps = build_timestamps(prepared_contaminated_dfs, self.config.window_size)            
            signal = X_test[:, -1].cpu().numpy()
            disinfectant_reconstruction = reconstructions[:, -1]
            plot_prediction(test_timestamps, signal, disinfectant_reconstruction, f"Test reconstruction node {node}")

        return results
    