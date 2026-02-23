from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from data_transformation import calculate_labels, create_extended_features, remove_first_x_days
from models.model import AnomalyModel

# from https://github.com/vincrichard/LSTM-AutoEncoder-Unsupervised-Anomaly-Detection/blob/master/src/model/LSTM_auto_encoder.py
# and from https://github.com/matanle51/LSTM_AutoEncoder/blob/master/models/LSTMAE.py
class LSTMAutoEncoder(nn.Module):
    """ Class for the LSTM autoencoder module"""
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        # input size : length of the input sequence (window size * number of features)
        super().__init__()

        self.input_size = input_size # size of the input sequence 
        self.hidden_size = hidden_size # dimension of the hidden state (latent space dimension later)
        self.num_layers = num_layers # number of LSTM layers in the encoder and decoder
        self.dropout = dropout

        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(input_size, hidden_size, num_layers, dropout)
    
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
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        
        self.linear = nn.Linear(hidden_size, input_size) 
    
    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.input_size, 1) # unsqueeze to add a sequence dimension and repeat to match the input size of the LSTM (so we have a representation for each timestep in the input sequence)
        output, _ = self.lstm(x) 
        output = self.linear(output)
        return output

    
    
    
class LSTMAutoEncoderModel(AnomalyModel):
    """ Class for Autoencoder model"""
    
    def run_model(self, train_batches, test_batches, epochs):
        """ 
        Trains the autoencoder on the training data and returns the anomaly scores for the test data.
        
        Parameters:
        - train_batches: data loader for the training data
        - val_batches: data loader for the validation/test data
        - epochs: the number of epochs to train the model 

        Returns:
        - anomalies: a numpy array of boolean values indicating whether each test sample is an anomaly (True) or not (False)
        - test_reconstruction: the reconstructed test data from the autoencoder
        - test_error: the reconstruction error for each test sample
        - threshold: the threshold used to classify anomalies
        """
        torch.manual_seed(42)

        print(train_batches.dataset[0].shape)
        model = LSTMAutoEncoder(train_batches.dataset[0].shape[1], 64, 2, 0.2)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = model.to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(epochs):
            train_loss = 0
            for batch in train_batches:
                
                batch = batch.to(device)
                
                optimizer.zero_grad()
                
                decoded = model(batch)
                loss = criterion(decoded, batch)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_batches)}")
        
        model.eval()
        
        
          

    def get_results(self):
        results = {}
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()

        for key, value in all_clean_dfs.items():
            print(f"Calculating results for node {key}")
            node = key
            
            # get contaminated dataset for the same key node
            contaminated_dfs = all_contaminated_dfs[key]
            clean_dfs = value

            train = []
            test = []
            
            new_clean_dfs = []
            # iterate over datasets for each node
            for i in range(len(clean_dfs)): 
                train_data = remove_first_x_days(clean_dfs[i], 3)
                new_clean_dfs.append(train_data)
                train_data = create_extended_features(train_data, self.config.disinfectant.value, self.config.window_size, stats=False)
                train.extend(train_data)
            train = np.array(train)

            new_contaminated_dfs = []
            for i in range(len(contaminated_dfs)): 
                test_data = remove_first_x_days(contaminated_dfs[i], 3)
                new_contaminated_dfs.append(test_data)
                test_data = create_extended_features(test_data, self.config.disinfectant.value, self.config.window_size, stats=False)
                test.extend(test_data)
            test = np.array(test)

            # Normalize the data 
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train)
            X_test = scaler.transform(test)
    
            X_train = torch.from_numpy(X_train)
            X_test = torch.from_numpy(X_test)
            
            train_batches = torch.utils.data.DataLoader(X_train, batch_size=32, shuffle=True)
            test_batches = torch.utils.data.DataLoader(X_test, batch_size=32, shuffle=False)
            
            self.run_model(train_batches, test_batches, epochs=20)
            
            

            # TODO : handle multiple contaminants, for now only one contaminant is handled
            # y_true = calculate_labels(contaminated_df, self.config.contaminants[0].value, self.config.window_size-1)
            # y_true = y_true[:len(X_test)]   
            # anomalies, reconstructions, test_error, threshold = self.run_model(X_train, X_test, 100)
            # y_pred = np.where(anomalies, -1, 1)  
            # results[node] = {"y_true": y_true, "y_pred": y_pred}

            # test_timestamps = contaminated_df.iloc[self.config.window_size:]["timestep"].values

            # self.get_plots(node, test_timestamps, X_test, reconstructions, anomalies, y_true, threshold, test_error)

        return results
    

    def get_plots(self, node, timestamps, X_test, reconstructions, anomalies, y_true, threshold, test_error):
        """
        Plots:
        1. Real vs reconstructed signal
        2. Signal with detected anomalies and true anomalies
        3. Reconstruction error with threshold
        """

        original = X_test.cpu().numpy()

        # Real vs Reconstructed
        plt.figure(figsize=(16,5))
        plt.plot(timestamps, original[:, 0], label="Real signal")
        plt.plot(timestamps, reconstructions[:, 0], label="Reconstructed signal")
        plt.title(f"Node {node} - Real vs Reconstructed")
        plt.legend()
        plt.show()


        # Detected vs True anomalies
        plt.figure(figsize=(16,5))
        plt.plot(timestamps, original[:, 0], label="Signal")


        plt.scatter(
            timestamps[anomalies],
            original[anomalies, 0],
            color="red",
            label="Detected anomalies"
        )

        true_anomalies = y_true == -1
        plt.scatter(
            timestamps[true_anomalies],
            original[true_anomalies, 0],
            color="green",
            marker="x",
            label="True anomalies"
        )

        plt.title(f"Node {node} - Detected vs True anomalies")
        plt.legend()
        plt.show()

        # Reconstruction error
        plt.figure(figsize=(16,4))
        plt.plot(timestamps, test_error, label="Reconstruction error")
        plt.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
        plt.title(f"Node {node} - Reconstruction error")
        plt.legend()
        plt.show()