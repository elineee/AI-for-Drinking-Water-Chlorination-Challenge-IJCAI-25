from models.LSTM_AE import LSTMAutoEncoderModel
from utils import detect_change_point
from data_transformation import calculate_labels
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class LSTMAutoEncoderAlarmModel(LSTMAutoEncoderModel):
    """ Class for LSTM Autoencoder with alarm model"""
    def get_results(self):
        results = {}
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()

        for node, clean_dfs in all_clean_dfs.items():
            print(f"Calculating results for node {node}")
            
            contaminated_dfs = all_contaminated_dfs[node]

            _ , train = self._prepare_dataset(clean_dfs, feature_type="extended")
            new_contaminated_dfs, test = self._prepare_dataset(contaminated_dfs, feature_type="extended")
            new_contaminated_df = pd.concat(new_contaminated_dfs)

            X_train, X_test = self._prepare_tensors(train, test)

            train_batches = torch.utils.data.DataLoader(X_train, batch_size=32, shuffle=True)
            test_batches = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=False) 
            
            mean_true_seq_per_timestep, mean_decoded_seq_per_timestep, anomalies = self.run_model(train_batches, test_batches, epochs=20)
            
            y_true = calculate_labels(new_contaminated_df, self.config.contaminants[0].value, 0)
            
            # convert mean_true_seq_per_timestep and mean_decoded_seq_per_timestep to float for plotting
            float_mean_true_seq_per_timestep = [
                float(val.mean()) if isinstance(val, np.ndarray) else float(val)
                for val in mean_true_seq_per_timestep
            ]

            float_mean_decoded_seq_per_timestep = [
                float(val.mean()) if isinstance(val, np.ndarray) else float(val)
                for val in mean_decoded_seq_per_timestep
            ]

            # Plot 
            plt.figure(figsize=(18,6))
            plt.plot(float_mean_true_seq_per_timestep, color = "red", linewidth=2.0, alpha = 0.6)
            plt.plot(float_mean_decoded_seq_per_timestep, color = "blue", linewidth=0.8)
            plt.legend(["Actual","Predicted"])
            plt.xlabel("Timestamp")
            plt.title("Test data reconstruction")
            plt.show()
            
            y_pred = detect_change_point(anomalies)
            
            results[node] = {"y_true": y_true, "y_pred": y_pred,}

        return results