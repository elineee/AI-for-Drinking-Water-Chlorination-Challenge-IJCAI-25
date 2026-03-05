from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from utils import plot_prediction, build_timestamps
from models.model import AnomalyModel

# Source: https://www.geeksforgeeks.org/deep-learning/implementing-an-autoencoder-in-pytorch/
# Source: https://www.datacamp.com/tutorial/introduction-to-autoencoders
# Source: https://keras.io/examples/timeseries/timeseries_anomaly_detection/
class AutoEncoder(nn.Module):
    """ Class for the autoencoder module"""
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    
class AutoEncoderModel(AnomalyModel):
    """ Class for AutoEncoder model"""
    
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
        X_train_mean = X_train.mean(axis=0)
        X_train_std = X_train.std(axis=0)
        X_train_std[X_train_std == 0] = 1

        X_train = (X_train - X_train_mean) / X_train_std
        X_test = (X_test - X_train_mean) / X_train_std

        return (
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(X_test, dtype=torch.float32),
            prepared_contaminated_dfs
        )


    def run_model(self, X_train : torch.Tensor, X_test : torch.Tensor, epochs: int, latent_dim : int) :
        """ 
        Trains the AutoEncoder on the training data and detects anomalies on the test data.
        
        Parameters:
        - X_train: the training data (clean data)
        - X_test: the test data (contaminated data)
        - epochs: the number of epochs to train the model 
        - latent_dim: the dimension of the latent space of the AutoEncoder 

        Returns:
        - anomalies: a numpy array of boolean values indicating whether each test sample is an anomaly (True) or not (False)
        - test_reconstruction: the reconstructed test data from the AutoEncoder
        - test_error: the reconstruction error for each test sample
        """
        torch.manual_seed(42)

        model = AutoEncoder(X_train.shape[1], latent_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)

        threshold_std = 1/ np.sqrt(latent_dim) # heuristic
        latent_stds = []


        # Training 
        for epoch in range(epochs):

            optimizer.zero_grad()
            train_reconstruction = model(X_train)
            train_loss = criterion(train_reconstruction, X_train)
            train_loss.backward()
            optimizer.step()

            print(f'Training: Epoch {epoch+1}, Loss: {train_loss}')

            # Compute latent stds
            with torch.no_grad(): 
                embeddings = model.encoder(X_train)
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
            train_reconstruction = model(X_train)
            train_error = torch.mean((train_reconstruction - X_train) ** 2, dim=1)            
            threshold = train_error.mean() + self._get_threshold_multiplier() * train_error.std()

            # Anomaly detection with the threshold  
            test_reconstruction = model(X_test)
            test_error = torch.mean((test_reconstruction - X_test) ** 2, dim=1)
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
            anomalies, reconstructions, test_error = self.run_model(X_train, X_test, 500, 8)
            
            y_pred = np.where(anomalies, -1, 1)  
            y_pred = self._post_predictions(y_pred)
            results[node] = {"y_true": y_true, "y_pred": y_pred}
            
            # Plot the reconstruction of the disinfectant value
            test_timestamps = build_timestamps(prepared_contaminated_dfs, self.config.window_size)            
            signal = X_test[:, -1].cpu().numpy()
            disinfectant_reconstruction = reconstructions[:, -1]
            plot_prediction(test_timestamps, signal, disinfectant_reconstruction, f"Test reconstruction node {node}")

        return results
    