from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from data_transformation import calculate_labels_alarm
from utils import cusum_detection
from models.autoencoder import AutoencoderModel, Autoencoder

# https://klaviyo.tech/developing-our-first-anomaly-detection-algorithm-7c84cab7ca46
# https://blog.stackademic.com/the-cusum-algorithm-all-the-essential-information-you-need-with-python-examples-f6a5651bf2e5
class AutoencoderAlarmModel(AutoencoderModel):
    """ Class for Autoencoder with alarm model"""
    
    def _calculate_labels(self, df, contaminant, window_size):
        return calculate_labels_alarm(df, contaminant, window_size)

    def run_model(self, X_train : torch.Tensor, X_test : torch.Tensor, epochs: int) :
        """ 
        Trains the autoencoder on the training data and returns the anomaly scores for the test data.
        
        Parameters:
        - X_train: the training data (clean data)
        - X_test: the test data (contaminated data)
        - epochs: the number of epochs to train the model 

        Returns:
        - anomalies: a numpy array of boolean values indicating whether each test sample is an anomaly (True) or not (False)
        - test_reconstruction_np: a numpy array of the reconstructed test data from the autoencoder
        - test_error_np : a numpy array of the reconstruction error for each test sample
        """
        torch.manual_seed(42)

        model = Autoencoder(X_train.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)

        # Training 
        for epoch in range(epochs):

            optimizer.zero_grad()
            train_reconstruction = model(X_train)
            train_loss = criterion(train_reconstruction, X_train)
            train_loss.backward()
            optimizer.step()

            print(f'Training: Epoch {epoch+1}, Loss: {train_loss}')
        
        model.eval()

        with torch.no_grad():
            # To compute the threshold 
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

            # CUSUM on the reconstruction error 
            _, cusum_train = cusum_detection(train_error_np, train_mean, train_std, k=0.6, threshold=float('inf')) # Or k=0.5?
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
