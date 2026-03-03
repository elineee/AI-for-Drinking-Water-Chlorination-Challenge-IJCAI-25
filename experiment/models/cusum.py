from unittest import result

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from data_transformation import calculate_labels, create_extended_features, create_features, remove_first_x_days
from models.model import AnomalyModel

class cusumModel(AnomalyModel):
    """ Class for CUSUM model"""
    
    def get_results(self):
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()
        
        results = {}
        
        for node, clean_dfs in all_clean_dfs.items():
            
            contaminated_dfs = all_contaminated_dfs[node] 
    
            X_test = []
            
            _, X_train = self._prepare_dataset(clean_dfs, feature_type="extended", stats=False)
            X_train = np.array(X_train)
            X_train = X_train.flatten()
            
            new_contaminated_dfs, X_test = self._prepare_dataset(contaminated_dfs, feature_type="extended", stats=False)
            X_test = np.array(X_test)
            X_test = X_test.flatten()
            
            new_contaminated_df = pd.concat(new_contaminated_dfs)
            
            cusum_train = self.cusum(X_train, X_train.mean(), X_train.std())
            
            cusum = self.cusum(X_test, X_test.mean(), X_test.std())
            
            threshold = cusum_train.mean() + 3*cusum_train.std()
            print(f"Threshold: {threshold}")
            
            y_pred = []
            for c in cusum:
                if c > threshold:
                    y_pred.append(-1)
                else:
                    y_pred.append(1)
            
            y_true = calculate_labels(new_contaminated_df, self.config.contaminants[0].value, self.config.window_size)

            # plot
            plt.figure(figsize=(10, 6))
            plt.plot(X_test, label='Data')
            plt.plot(cusum, label='CUSUM', color='red')
            plt.axhline(y=0, color='gray', linestyle='--')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.title('CUSUM Chart')
            plt.grid(lw=2,ls=':')
            plt.show()
            
            results[node] = {
                "y_pred": y_pred,
                "y_true": y_true
            }
            
        return results
    
    def cusum(self, data, reference_mean, reference_std=None, k=0.5):
        """
        CUSUM algorithm 
        
        Parameters:
        data: input data (1D array)
        target: mean of the data (float)
        k: reference value for detecting change points (float)
        h: threshold for detecting change points (float)
        """
        
        n = len(data)
        C_pos = np.zeros(n)  # CUSUM for increases
        C_neg = np.zeros(n)  # CUSUM for decreases
        
        for i in range(1, n):
            C_pos[i] = max(0, C_pos[i-1] + data[i] - reference_mean - k*reference_std)
            C_neg[i] = min(0, C_neg[i-1] + data[i] - reference_mean + k*reference_std)
        
        # take the maximum of the absolute values of C_pos and C_neg
        return np.maximum(np.abs(C_pos), np.abs(C_neg))