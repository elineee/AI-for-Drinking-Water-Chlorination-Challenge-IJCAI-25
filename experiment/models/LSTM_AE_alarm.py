from models.LSTM_AE import LSTMAutoEncoderModel
from utils import detect_change_point
from data_transformation import calculate_labels_alarm
import torch
import pandas as pd


class LSTMAutoEncoderAlarmModel(LSTMAutoEncoderModel):
    """ Class for LSTM Autoencoder with alarm model"""
    def get_results(self):
        results = {}
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()

        for node, clean_dfs in all_clean_dfs.items():
            print(f"Calculating results for node {node}")
            
            contaminated_dfs = all_contaminated_dfs[node]

            _ , train = self._prepare_dataset(clean_dfs, feature_type="extended", stats = False)
            new_contaminated_dfs, test = self._prepare_dataset(contaminated_dfs, feature_type="extended", stats = False)
            new_contaminated_df = pd.concat(new_contaminated_dfs)

            X_train, X_test = self._prepare_tensors(train, test)

            train_batches = torch.utils.data.DataLoader(X_train, batch_size=32, shuffle=True)
            test_batches = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=False) 
            
            mean_true_seq_per_timestep, mean_decoded_seq_per_timestep, anomalies = self.run_model(train_batches, test_batches, epochs=20)
            
            y_true = calculate_labels_alarm(new_contaminated_df, self.config.contaminants[0].value, 0)
            
            true_seq = self._to_float_sequence(mean_true_seq_per_timestep)
            decoded_seq = self._to_float_sequence(mean_decoded_seq_per_timestep)

            self._plot_reconstruction(true_seq, decoded_seq)

            y_pred = detect_change_point(anomalies)
            
            results[node] = {"y_true": y_true, "y_pred": y_pred,}

        return results