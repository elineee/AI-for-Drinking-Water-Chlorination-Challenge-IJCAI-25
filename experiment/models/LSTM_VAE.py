import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from data_transformation import calculate_labels
from models.LSTM_AE import LSTMAutoencoderModel
        
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, latent_dim):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.mu = nn.Linear(hidden_size, latent_dim)
        self.log_var = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x) 
        hidden_state = hidden[-1] # take hidden state of the last layer to get the latent representation
        return self.mu(hidden_state), self.log_var(hidden_state) 

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, seq_len, latent_dim):
        super().__init__()

        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_len = seq_len  
        
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        
        self.linear = nn.Linear(hidden_size, input_size) 
    
    def forward(self, z):
        # z: (batch, hidden_size)
        z = z.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, hidden_size) to repeat the latent representation for each timestep in the sequence
        output, _ = self.lstm(z)  # (batch, seq_len, hidden_size)
        output = self.linear(output)  # (batch, seq_len, input_size)
        return output
    
class LSTMVAE(nn.Module):
    """ Class for the LSTM VAE module"""
    def __init__(self, input_size, hidden_size, num_layers, dropout, seq_len, latent_dim):

        super().__init__()

        self.input_size = input_size # number of features per timestep (if only chlorine, then 1)
        self.hidden_size = hidden_size # dimension of the hidden state (latent space dimension)
        self.num_layers = num_layers # number of LSTM layers in the encoder and decoder
        self.dropout = dropout
        self.seq_len = seq_len # sequence length (window_size)

        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout, latent_dim)
        self.decoder = Decoder(input_size, hidden_size, num_layers, dropout, seq_len, latent_dim)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        return decoded, mu, log_var
    
class LSTMVAEModel(LSTMAutoencoderModel):
    """ Class for LSTM VAE model"""

    def _get_KLD_multiplier(self): 
        return 0.01
    
    def run_model(self, train_batches: torch.Tensor, test_batches: torch.Tensor, epochs:int, latent_dim:int):
        """
        Trains the LSTM Autoencoder on the training data and returns the anomaly scores for the test data.
        The anomaly threshold is computed from the training reconstruction errors as: mean(training_error) + 3 * std(training_error).

        Parameters: 
        - train_batches : DataLoader containing the training data
        - test_batches : DataLoader containing the test data 
        - epochs : number of epochs used to train the Autoencoder
        - latent_dim : the dimension of the latent space of the LSTM VAE

        Returns:
        - mean_true_seq_per_timestep : list of mean true value per timestep. 
        - mean_decoded_seq_per_timestep : list of mean reconstructed value per timestep.
        - anomalies : list of predicted values per timestep based on the reconstruction error threshold. Values are -1 for anomalies and 1 for normal data.
        """

        # Get tensor shape: (batch_size, seq_len, num_features)
        sample_batch = next(iter(train_batches))
        seq_len = sample_batch.shape[1]
        num_features = sample_batch.shape[2]
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = LSTMVAE(num_features,64, 2, 0.2, seq_len, latent_dim)
        model = model.to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay= 1e-8)
        
        # Train the model
        model.train()
        for epoch in range(epochs):
            train_loss = 0
            KLD_multiplier = min(self._get_KLD_multiplier(), (epoch / 30) * self._get_KLD_multiplier())


            for batch in train_batches:
                batch = batch.to(device)
                optimizer.zero_grad()
                decoded, mu, log_var = model(batch)
                KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                reconstruction_loss = criterion(decoded, batch)
                loss = reconstruction_loss + KLD_multiplier * KLD
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_batches):.6f}")
                
        # Evaluation
        model.eval()
        with torch.no_grad():
            # get training reconstruction errors to calculate the threshold for anomaly detection
            training_errors_per_window = []
            for batch in train_batches:
                batch = batch.to(device)
                decoded, _, _ = model(batch)
                error_per_window = torch.mean((decoded - batch) ** 2, dim=(1, 2))  # shape: (batch_size,)
                training_errors_per_window.extend(error_per_window.cpu().numpy())

            training_errors_per_window_np = np.array(training_errors_per_window)
            threshold = training_errors_per_window_np.mean() + 1.5 * training_errors_per_window_np.std()
        
            print(f"Threshold for anomaly detection: {threshold}")
            
        seq_decoded = [[0, 0] for _ in range(len(test_batches.dataset) + self.config.window_size)] 
        true_seq = [[0, 0] for _ in range(len(test_batches.dataset) + self.config.window_size)]
        anomalies = []
        scores_per_timestep = [[0, 0] for _ in range(len(test_batches.dataset) + self.config.window_size)]
        with torch.no_grad(): 
            i = 0
            for batch in test_batches:
                batch = batch.to(device) # shape : (batch_size, seq_len, num_features), here batch_size is 1 since we set batch_size=1 for the test data loader to get predictions for each sample in the test set

                decoded, _, _ = model(batch)  # shape : (batch_size, seq_len, num_features)
                # accumulate error per timestep 
                j = 0
                for element in batch.cpu().numpy()[0]:
                    error = float(np.mean((decoded.cpu().numpy()[0][j] - element) ** 2))
                    
                    scores_per_timestep[i+j][0] += error
                    scores_per_timestep[i+j][1] += 1
                    
                    true_seq[i+j][0] += element
                    true_seq[i+j][1] += 1
                    
                    seq_decoded[i+j][0] += decoded.cpu().numpy()[0][j]
                    seq_decoded[i+j][1] += 1
                    
                    j += 1
                i += 1

        # calculate mean error per timestep
        mean_scores_per_timestep = []
        for total_error, count in scores_per_timestep:
            mean_scores_per_timestep.append(total_error / count if count > 0 else 0)
        
        # calculate mean true value per timestep for plotting
        mean_true_seq_per_timestep = []
        for total_true, count_true in true_seq:
            mean_true_seq_per_timestep.append(total_true / count_true if count_true > 0 else 0)
        
        # calculate mean decoded value per timestep for plotting
        mean_decoded_seq_per_timestep = [] 
        for total_decoded, count_decoded in seq_decoded:
            mean_decoded_seq_per_timestep.append(total_decoded / count_decoded if count_decoded > 0 else 0)
        
        # get anomalies based on the threshold
        for element in mean_scores_per_timestep:
            if element > threshold:
                anomalies.append(-1) # anomaly
            else:
                anomalies.append(1) # normal

        return mean_true_seq_per_timestep, mean_decoded_seq_per_timestep, anomalies

    def get_results(self):
        results = {}
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()

        for node, clean_dfs in all_clean_dfs.items():
            print(f"Calculating results for node {node}")
            
            contaminated_dfs = all_contaminated_dfs[node]
            X_train, X_test, prepared_contaminated_dfs = self._prepare_data(clean_dfs, contaminated_dfs)
            prepared_contaminated_df = pd.concat(prepared_contaminated_dfs)

            train_batches = DataLoader(X_train, batch_size=32, shuffle=True)
            test_batches = DataLoader(X_test, batch_size=1, shuffle=False) 
            
            mean_true_seq_per_timestep, mean_decoded_seq_per_timestep, anomalies = self.run_model(train_batches, test_batches, epochs=50, latent_dim = 4)
            
            y_true = calculate_labels(prepared_contaminated_df, self.config.contaminants[0].value, 0)
            
            true_seq = self._convert_sequence_to_float(mean_true_seq_per_timestep)
            decoded_seq = self._convert_sequence_to_float(mean_decoded_seq_per_timestep)

            self._plot_reconstruction(true_seq, decoded_seq)
                        
            y_true = np.array(y_true)
            print(f"ok: {(y_true == 1).sum()}, ano: {(y_true == -1).sum()}")

            y_pred = self._post_predictions(anomalies)
            
            results[node] = { "y_true": y_true, "y_pred": y_pred}

        return results