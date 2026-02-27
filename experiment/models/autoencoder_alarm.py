from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from data_transformation import calculate_labels_alarm, create_extended_features
from utils import cusum_detection
from models.model import AnomalyModel
from models.autoencoder import Autoencoder

# https://klaviyo.tech/developing-our-first-anomaly-detection-algorithm-7c84cab7ca46
# https://blog.stackademic.com/the-cusum-algorithm-all-the-essential-information-you-need-with-python-examples-f6a5651bf2e5

class AutoencoderAlarmModel(AnomalyModel):
    """ Class for Autoencoder with alarm model"""
    
    def run_model(self, X_train : torch.Tensor, X_test : torch.Tensor, epochs: int) :
        """ 
        Trains the autoencoder on the training data and returns the anomaly scores for the test data.
        
        Parameters:
        - X_train: the training data (clean data)
        - X_test: the test data (contaminated data)
        - epochs: the number of epochs to train the model 

        Returns:
        - anomalies: a numpy array of boolean values indicating whether each test sample is an anomaly (True) or not (False)
        - test_reconstruction: the reconstructed test data from the autoencoder
        - test_error: the reconstruction error for each test sample
        - threshold: the threshold used to classify anomalies
        """
        torch.manual_seed(42)

        model = Autoencoder(X_train.shape[1])
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-8)

        # Training
        for epoch in range(epochs):
            optimizer.zero_grad()

            train_reconstruction = model(X_train)
            loss = criterion(train_reconstruction, X_train)

            loss.backward()
            optimizer.step()

            print(f'Training: Epoch {epoch+1}, Loss: {loss}')
        
        model.eval()

        with torch.no_grad():
            train_reconstruction = model(X_train)
            train_error = torch.mean((train_reconstruction - X_train) ** 2, dim=1)
            train_error_np = train_error.cpu().numpy()
            train_mean = train_error.mean().item()
            train_std = train_error.std().item()

            # Testing
            test_reconstruction = model(X_test)
            test_reconstruction_np = test_reconstruction.cpu().numpy()
            test_error = torch.mean((test_reconstruction - X_test) ** 2, dim=1)
            test_error_np = test_error.cpu().numpy()

            # CUSUM 
            _, cusum_train, _ = cusum_detection(train_error_np, train_mean, train_std, k=0.6, threshold=99999) # Or k=0.5?
            threshold = cusum_train.max() * 1.2
            print(f'Threshold {threshold}')
            anomalies, cusum_scores = cusum_detection(test_error_np, train_mean, train_std, k=0.9, threshold=threshold)

            plt.figure(figsize=(18, 4))
            plt.plot(cusum_scores, label='CUSUM score')
            plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
            plt.legend()
            plt.title("CUSUM score")
            plt.show()
        
            return (anomalies, test_reconstruction_np, test_error_np)        


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
            y_true = calculate_labels_alarm(contaminated_df, self.config.contaminants[0].value, self.config.window_size)
            anomalies, _, _ = self.run_model(X_train, X_test, 100)
            results[node] = {"y_true": y_true, "y_pred": anomalies}

        return results

