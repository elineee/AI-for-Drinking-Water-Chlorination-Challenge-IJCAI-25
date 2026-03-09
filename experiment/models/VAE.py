import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from models.autoencoder import AutoencoderModel
from utils import plot_prediction, build_timestamps

# https://www.datacamp.com/tutorial/variational-autoencoders
# https://medium.com/@sofeikov/implementing-variational-autoencoders-from-scratch-533782d8eb95
# https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb

class VAE(nn.Module):
    """ Class for the Variational AutoEncoder module  """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def reparameterize ( self, mu, log_var ): 
        """
        Does reparameterization to have a differentiable process. 

        Parameters:
        - mu: mean of the latent distribution
        - log_var: log variance of the latent distribution

        Returns: 
        - mu + eps * std 
        """

        std = torch.exp( 0.5 * log_var) 
        eps = torch.randn_like(std) # Generate random noise with the same shape than std 
        return mu + eps * std 
    
    def forward(self, x):
        """
        The encoder learns two vectors (mu and log_var) defining a gaussian distribution.
        The latent variable z is sampled via reparameterization and passed to the decoder. 
        """
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        return decoded, mu, log_var
    
    
class VAEModel(AutoencoderModel):
    """ Class for Variational Autoencoder model"""

    def _get_KLD_multiplier(self): 
        return 0.01

    def run_model(self, train_batches : torch.Tensor, test_batches: torch.Tensor, epochs: int, hidden_dim : int, latent_dim : int) :
        """ 
        Trains the VAE on the training data and detects anomalies on the test data.
        The criterion is the sum of MSELoss and Kullback-Leibler divergence.
        
        Parameters:
        - train_batches: the training data in batches (clean data)
        - test_batches: the test data in batches (contaminated data)
        - epochs: the number of epochs to train the model 
        - hidden_dim: the dimension of the hidden layer
        - latent_dim: the dimension of the latent space of the VAE

        Returns:
        - anomalies: a numpy array of boolean values indicating whether each test sample is an anomaly (True) or not (False)
        - test_reconstruction: the reconstructed test data from the VAE
        - test_error: the reconstruction error for each test sample
        """
        torch.manual_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # shape: (batch_size, num_features)
        sample_batch = next(iter(train_batches))
        input_dim = sample_batch.shape[-1]
        model = VAE(input_dim, hidden_dim, latent_dim).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)

        threshold_std = 1/ np.sqrt(latent_dim) # heuristic

        # Training 
        model.train()
        for epoch in range(epochs):
 
            KLD_multiplier = min(self._get_KLD_multiplier(), (epoch / 200) * self._get_KLD_multiplier())
            epoch_losses = []
            epoch_mse = []
            epoch_kld = []

            for batch in train_batches:
                batch = batch.to(device) 
                optimizer.zero_grad()
                train_reconstruction, mu, log_var = model(batch)
                KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                reconstruction_loss = criterion(train_reconstruction, batch)
                train_loss = reconstruction_loss + KLD_multiplier * KLD
                train_loss.backward()
                optimizer.step()

                epoch_losses.append(train_loss.item())
                epoch_mse.append(reconstruction_loss.item())
                epoch_kld.append(KLD.item())

            if (epoch + 1) % 50 == 0:
                print(f" Epoch {epoch+1}, Loss: {np.mean(epoch_losses):.4f}, MSE: {np.mean(epoch_mse):.4f}, KLD: {np.mean(epoch_kld):.4f}")

        # Evaluation 
        model.eval()
        with torch.no_grad():

            # Computes the threshold 
            train_errors = []

            for batch in train_batches: 
                batch = batch.to(device) 
                train_reconstruction, _, _ = model(batch)
                train_error = torch.mean((train_reconstruction - batch) ** 2, dim=1)            
                train_errors.append(train_error)

            train_error = torch.cat(train_errors)
            threshold = (train_error.mean() + self._get_threshold_multiplier() * train_error.std()).item()

            # Anomaly detection with the threshold  

            test_errors =[]
            test_reconstructions = []

            for batch in test_batches:
                batch = batch.to(device) 
                test_reconstruction, _, _  = model(batch)
                test_error = torch.mean((test_reconstruction - batch) ** 2, dim=1)
                test_errors.append(test_error) 
                test_reconstructions.append(test_reconstruction)

            test_error = torch.cat(test_errors)
            test_reconstruction  = torch.cat(test_reconstructions)
            anomalies = test_error > threshold
                
            print(f"threshold: {threshold:.4f}")
            print(f"train_error mean: {train_error.mean():.4f}, std: {train_error.std():.4f}")
            print(f"test_error mean: {test_error.mean():.4f}, std: {test_error.std():.4f}")
        
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

            train_batches = DataLoader(X_train, batch_size=32, shuffle=True)
            test_batches = DataLoader(X_test, batch_size=32, shuffle=False)

            # Anomaly detection
            # TODO : handle multiple contaminants, for now only one contaminant is handled
            anomalies, reconstructions, test_error = self.run_model(train_batches, test_batches, epochs=300, hidden_dim=64, latent_dim=4 )
            y_true = self._calculate_labels(prepared_contaminated_df, self.config.contaminants[0].value, self.config.window_size)
            
            y_pred = np.where(anomalies, -1, 1)  
            y_pred = self._post_predictions(y_pred)
            results[node] = {"y_true": y_true, "y_pred": y_pred}
            
            # Plot the reconstruction of the disinfectant value
            test_timestamps = build_timestamps(prepared_contaminated_dfs, self.config.window_size)            
            signal = X_test[:, -1].cpu().numpy()
            disinfectant_reconstruction = reconstructions[:, -1]
            # plot_prediction(test_timestamps, signal, disinfectant_reconstruction, f"Test reconstruction node {node}")

        return results