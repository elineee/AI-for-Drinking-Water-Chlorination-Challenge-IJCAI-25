from matplotlib import pyplot as plt
import numpy as np
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

    def run_model(self, train_batches: torch.Tensor, test_batches: torch.Tensor, epochs: int, latent_dim: int) :
        """ 
        Trains the Autoencoder on the training data and detects anomalies on the test data.
        The model reconstructs each window and computes a reconstruction error (MSELoss) per window.
        A window is an anomaly if its CUSUM score exceeds a threshold computed from the training data.
        
        Parameters:
        - train_batches: the training data in batches (clean data)
        - test_batches: the test data in batches (contaminated data)
        - epochs: the number of epochs to train the model 
        - latent_dim: the dimension of the latent space of the Autoencoder 

        Returns:
        - anomalies: a numpy array of boolean values indicating whether each test window is an anomaly (True) or not (False)
        - a numpy array of the reconstructed test data from the Autoencoder
        - a numpy array of the reconstruction error for each test window
        """
        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        sample_batch = next(iter(train_batches))
        input_dim = sample_batch.shape[1]
        model = Autoencoder(input_dim, latent_dim).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)

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

        model.eval()

        with torch.no_grad():
            # Compute the threshold 
            train_errors = []

            for batch in train_batches:
                batch = batch.to(device)
                train_reconstruction = model(batch)
                error = torch.mean((train_reconstruction - batch) ** 2, dim=1)
                train_errors.append(error)

            train_error = torch.cat(train_errors)
            train_mean = train_error.mean().item()
            train_std = train_error.std().item()

            # Testing
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

            # CUSUM on the reconstruction error 
            _, cusum_train = cusum_detection(train_error.cpu().numpy(), train_mean, train_std, k=0.6, threshold=float('inf'))
            threshold = cusum_train.max() * 1.2
            print(f"Threshold: {threshold:.4f}")

            anomalies, cusum_scores = cusum_detection(test_error.cpu().numpy(), train_mean, train_std, k=0.9, threshold=threshold)

            plt.figure(figsize=(18, 4))
            plt.plot(cusum_scores, label='CUSUM score')
            plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
            plt.legend()
            plt.title("CUSUM score")
            plt.show()

            return (anomalies, test_reconstruction.cpu().numpy(), test_error.cpu().numpy())
    