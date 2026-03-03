import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from data_transformation import calculate_labels
from models.model import AnomalyModel

# from https://github.com/vincrichard/LSTM-AutoEncoder-Unsupervised-Anomaly-Detection/blob/master/src/model/LSTM_auto_encoder.py
# and from https://github.com/matanle51/LSTM_AutoEncoder/blob/master/models/LSTMAE.py

class LSTMAutoEncoder(nn.Module):
    """ Class for the LSTM autoencoder module"""
    def __init__(self, input_size, hidden_size, num_layers, dropout, seq_len):

        super().__init__()

        self.input_size = input_size # number of features per timestep (if only chlorine, then 1)
        self.hidden_size = hidden_size # dimension of the hidden state (latent space dimension)
        self.num_layers = num_layers # number of LSTM layers in the encoder and decoder
        self.dropout = dropout
        self.seq_len = seq_len # sequence length (window_size)

        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(input_size, hidden_size, num_layers, dropout, seq_len)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded
        

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
    
    def forward(self, x):
        _, (hidden, _) = self.lstm(x) 
        return hidden[-1] # take hidden state of the last layer to get the latent representation

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, seq_len):
        super().__init__()

        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_len = seq_len  
        
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        
        self.linear = nn.Linear(hidden_size, input_size) 
    
    def forward(self, x):
        # x: (batch, hidden_size)
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, hidden_size) to repeat the latent representation for each timestep in the sequence
        output, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        output = self.linear(output)  # (batch, seq_len, input_size)
        return output
    
    
class LSTMAutoEncoderModel(AnomalyModel):
    """ Class for LSTM Autoencoder model"""
    
    def run_model(self, train_batches, test_batches, epochs):
        """ 
        Trains the autoencoder on the training data and returns the anomaly scores for the test data.
        
        Parameters:
        - train_batches: data loader for the training data
        - test_batches: data loader for the validation/test data
        - epochs: the number of epochs to train the model 

        Returns:
        - anomalies: a numpy array of boolean values indicating whether each test sample is an anomaly (True) or not (False)
        - test_reconstruction: the reconstructed test data from the autoencoder
        - test_error: the reconstruction error for each test sample
        - threshold: the threshold used to classify anomalies
        """

        # Get tensor shape: (batch_size, seq_len, num_features)
        sample_batch = next(iter(train_batches))
        seq_len = sample_batch.shape[1]
        num_features = sample_batch.shape[2]
        
        model = LSTMAutoEncoder(num_features, 16, 2, 0.2, seq_len)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = model.to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # if os.path.exists("lstm_autoencoder.pth"):
        if False:
            model.load_state_dict(torch.load("lstm_autoencoder.pth", weights_only=True))
            
        else: 
            # train the model
            model.train()
            for epoch in range(epochs):
                train_loss = 0
                for batch in train_batches:
                    batch = batch.to(device)
                    
                    optimizer.zero_grad()
                    
                    decoded = model(batch)
                    loss = criterion(batch, decoded)
                    
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    
                print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_batches):.6f}")
            
            torch.save(model.state_dict(), "lstm_autoencoder.pth")
        
        # evaluate the model
        model.eval()
        with torch.no_grad():
            # get training reconstruction errors to calculate the threshold for anomaly detection
            training_errors_per_window = []
            for batch in train_batches:
                batch = batch.to(device)
                decoded = model(batch)
                error_per_window = torch.mean((decoded - batch) ** 2, dim=(1, 2))  # shape: (batch_size,)
                training_errors_per_window.extend(error_per_window.cpu().numpy())

            training_errors_per_window = np.array(training_errors_per_window)
            threshold = training_errors_per_window.mean() + 3 * training_errors_per_window.std()
        
            print(f"Threshold for anomaly detection: {threshold}")
            
        seq_decoded = [[0, 0] for _ in range(len(test_batches.dataset) + self.config.window_size)] 
        true_seq = [[0, 0] for _ in range(len(test_batches.dataset) + self.config.window_size)]
        anomalies = []
        scores_per_timestep = [[0, 0] for _ in range(len(test_batches.dataset) + self.config.window_size)]
        with torch.no_grad(): 
            i = 0
            for batch in test_batches:
                batch = batch.to(device) # shape : (batch_size, seq_len, num_features), here batch_size is 1 since we set batch_size=1 for the test data loader to get predictions for each sample in the test set

                decoded = model(batch) # shape : (batch_size, seq_len, num_features)
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

            
            _ , train = self._prepare_dataset(clean_dfs, feature_type="extended", stats = "False")
            new_contaminated_dfs, test = self._prepare_dataset(contaminated_dfs, feature_type="extended", stats = "False")
            new_contaminated_df = pd.concat(new_contaminated_dfs)

            X_train, X_test = self._prepare_tensors(train, test)

            train_batches = torch.utils.data.DataLoader(X_train, batch_size=32, shuffle=True)
            test_batches = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=False) 
            
            mean_true_seq_per_timestep, mean_decoded_seq_per_timestep, anomalies = self.run_model(train_batches, test_batches, epochs=20)
            
            y_true = calculate_labels(new_contaminated_df, self.config.contaminants[0].value, 0)
            
            true_seq = self._to_float_sequence(mean_true_seq_per_timestep)
            decoded_seq = self._to_float_sequence(mean_decoded_seq_per_timestep)

            self._plot_reconstruction(true_seq, decoded_seq)
                        
            y_true = np.array(y_true)
            print(f"ok: {(y_true == 1).sum()}, ano: {(y_true == -1).sum()}")
            
            results[node] = { "y_true": y_true, "y_pred": anomalies,}

        return results
    

    def _prepare_tensors(self, train, test):
        """
        Normalizes sliding window datasets and converts them into PyTorch tensors.

        Parameters: 
        - train : training sliding windows 
        - test : test sliding windows

        Returns:
        - train and test tensors. 
        """

        # Normalize the data 
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train)
        X_test = scaler.transform(test)

        # convert numpy arrays of type float64 to type float32 for PyTorch
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        # Reshape to 3D: (num_samples, seq_len, num_features)
        X_train = X_train[:, :, np.newaxis] # (num_samples, window_size, 1) since only one feature (chlorine) is used for the model
        X_test = X_test[:, :, np.newaxis] # (num_samples, window_size, 1) since only one feature (chlorine) is used for the model

        return torch.from_numpy(X_train), torch.from_numpy(X_test)
    

    def _to_float_sequence(self, seq):
        """
        Converts a sequence containing floats or numpy arrays into a list of floats.
        
        Parameters: 
        - seq: the sequence to convert 

        Returns: 
        - a list of floats with the sequence values 
        """
        return [
            float(val.mean()) if isinstance(val, np.ndarray) else float(val)
            for val in seq
        ]


    def _plot_reconstruction(self, true_seq, pred_seq):
            """
            Plots the reconstructed sequence vs the true sequence.
            
            Parameters: 
            - true_seq: the true sequence of data (list of floats)
            - pred_seq: the reconstructed sequence (list of floats)
            """
            plt.figure(figsize=(18,6))
            plt.plot(true_seq, color="red", linewidth=2.0, alpha=0.6)
            plt.plot(pred_seq, color="blue", linewidth=0.8)
            plt.legend(["Actual", "Predicted"])
            plt.xlabel("Timestamp")
            plt.title("Test data reconstruction")
            plt.show()