import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.LSTM_AE import LSTMAutoEncoder, LSTMAutoEncoderModel
from utils import detect_change_point, cusum_detection
from data_transformation import calculate_labels_alarm


class LSTMAutoEncoderAlarmModel(LSTMAutoEncoderModel):
    """ Class for LSTM Autoencoder with alarm model"""

    def _calculate_labels(self, df, contaminant, window_size):
        return calculate_labels_alarm(df, contaminant, window_size)

    def _post_predictions(self, y_pred):
        use_cusum = self.config.model_params.get("use_cusum", False)
        if use_cusum:
            return y_pred 
        return detect_change_point(y_pred)
    
    def run_model(self, train_batches, test_batches, epochs):
        """
        Trains the LSTM autoencoder on the training data and returns the anomaly scores for the test data.
        The anomaly threshold is computed from the training reconstruction errors as: mean(training_error) + 3 * std(training_error).

        Parameters: 
        - train_batches : DataLoader containing the training data
        - test_batches : DataLoader containing the test data 
        - epochs : number of epochs used to train the autoencoder.

        Returns:
        - mean_true_seq_per_timestep : list of mean true value per timestep. 
        - mean_decoded_seq_per_timestep : list of mean reconstructed value per timestep.
        - anomalies : list of predicted values per timestep based on the reconstruction error threshold. Values are -1 for anomalies and 1 for normal data.
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


        # evaluate the model
        model.eval()
        with torch.no_grad():

            training_errors_per_window = []
            for batch in train_batches:
                batch = batch.to(device)
                decoded = model(batch)
                error_per_window = torch.mean((decoded - batch) ** 2, dim=(1, 2))  # shape: (batch_size,)
                training_errors_per_window.extend(error_per_window.cpu().numpy())

            training_errors_per_window_np = np.array(training_errors_per_window)
            train_mean = training_errors_per_window_np.mean()
            train_std = training_errors_per_window_np.std()

            
        seq_decoded = [[0, 0] for _ in range(len(test_batches.dataset) + self.config.window_size)] 
        true_seq = [[0, 0] for _ in range(len(test_batches.dataset) + self.config.window_size)]
        anomalies = []
        scores_per_timestep = [[0, 0] for _ in range(len(test_batches.dataset) + self.config.window_size)]

        with torch.no_grad(): 
            i = 0
            for batch in test_batches:
                batch = batch.to(device) 

                decoded = model(batch) 

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
        
        use_cusum = self.config.model_params.get("use_cusum", False)

        if use_cusum:
            mean_scores = np.array(mean_scores_per_timestep)
            _, cusum_train = cusum_detection(training_errors_per_window, train_mean, train_std, k=0.6, threshold=float('inf'))
            threshold = cusum_train.max() * 1.2
            anomalies, _ = cusum_detection(mean_scores, train_mean, train_std, k=0.9, threshold=threshold)
        else:
            threshold = train_mean + 3 * train_std
            anomalies = np.array([-1 if element > threshold else 1 for element in mean_scores_per_timestep])

        return mean_true_seq_per_timestep, mean_decoded_seq_per_timestep, anomalies