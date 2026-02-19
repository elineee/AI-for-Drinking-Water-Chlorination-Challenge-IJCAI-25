from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from data_transformation import calculate_labels, create_extended_features
from models.model import AnomalyModel

# Source: https://www.geeksforgeeks.org/deep-learning/implementing-an-autoencoder-in-pytorch/
# Source: https://www.datacamp.com/tutorial/introduction-to-autoencoders
# Source: https://keras.io/examples/timeseries/timeseries_anomaly_detection/
class Autoencoder(nn.Module):
    """ Class for the autoencoder module"""
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    
class AutoencoderModel(AnomalyModel):
    """ Class for Autoencoder model"""
    
    def run_model(self, X_train : torch.Tensor, X_test : torch.Tensor, epochs: int) :
        """ 
        Trains the autoencoder on the training data and returns the anomaly scores for the test data.
        
        Parameters:
        - X_train: the training data (clean data)
        - X_test: the test data (contaminated data)
        - epochs: the number of epochs to train the model 

        Returns:
        - anomalies: a numpy array of boolean values indicating whether each test sample is an anomaly (True) or not (False)
        """
        torch.manual_seed(42)

        model = Autoencoder(X_train.shape[1])
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-8)

        # Training
        for epoch in range(epochs):
            train_reconstruction = model(X_train)
            loss = criterion(train_reconstruction, X_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Training: Epoch {epoch+1}, Loss: {loss}')
        
        model.eval()

        with torch.no_grad():
            train_reconstruction = model(X_train)
            train_error = torch.mean((train_reconstruction - X_train) ** 2, dim=1)
            
            threshold = train_error.max()

            # Testing
            test_reconstruction = model(X_test)
            test_error = torch.mean((test_reconstruction - X_test) ** 2, dim=1)
            anomalies = test_error > threshold
            
            return anomalies.cpu().numpy()
        

    def get_results(self):
        results = {}
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()

        for node, clean_dfs in all_clean_dfs.items():
            clean_df = pd.concat(clean_dfs)
            contaminated_dfs = all_contaminated_dfs[node]
            contaminated_df = pd.concat(contaminated_dfs)

            X_train = create_extended_features(clean_df, self.config.disinfectant.value, self.config.window_size)
            X_test = create_extended_features(contaminated_df, self.config.disinfectant.value, self.config.window_size)

            # Normalize the data 
            mean = X_train.mean(axis=0)
            std = X_train.std(axis=0)
            std[std == 0] = 1  

            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std
    
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)

            # TODO : handle multiple contaminants, for now only one contaminant is handled
            y_true = calculate_labels(contaminated_df, self.config.contaminants[0].value, self.config.window_size-1)
            y_true = y_true[:len(X_test)]   
            anomalies = self.run_model(X_train, X_test, 50)
            y_pred = np.where(anomalies, -1, 1)  
            results[node] = {"y_true": y_true, "y_pred": y_pred}

            # test_timestamps = contaminated_df.iloc[self.config.window_size:]["timestep"]

            # plt.figure(figsize=(17,5))
            # plt.plot(test_timestamps, X_test[:,0].cpu().numpy(), label="Actual")
            # plt.scatter(test_timestamps[anomalies], X_test[anomalies,0].cpu().numpy(),
            #             color="red", label="Anomaly")
            # plt.legend()
            # plt.title(f"Node {node} anomalies")
            # plt.show()

        return results