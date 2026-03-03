
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from data_transformation import calculate_labels
from models.model import AnomalyModel

class CusumModel(AnomalyModel):
    """ Class for CUSUM model"""
    
    def _get_threshold_multiplier(self):
        return 3
    
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
            
            threshold = cusum_train.mean() + self._get_threshold_multiplier() *cusum_train.std()
            print(f"Threshold: {threshold}")
            
            y_pred = []
            for c in cusum:
                if c > threshold:
                    y_pred.append(-1)
                else:
                    y_pred.append(1)
            
            y_pred = self._post_predictions(y_pred)
            y_true = calculate_labels(new_contaminated_df, self.config.contaminants[0].value, self.config.window_size)

            # plot
            plt.figure(figsize=(10, 6))
            plt.plot(X_test, label="Data")
            plt.plot(cusum, label="CUSUM", color="red")
            plt.axhline(y=0, color="gray", linestyle="--")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.title("CUSUM Chart")
            plt.grid(lw=2,ls=":")
            plt.show()
            
            results[node] = { "y_pred": y_pred, "y_true": y_true}
            
        return results
    
    def cusum(self, data, reference_mean, reference_std=None, k=0.5):
        """
        Computes the CUSUM for change point detection.
        
        Parameters:
        - data: input time series 
        - reference_mean: mean of the data 
        - reference_std: std of the data 
        - k: parameter controlling sensitivity (default=0.5)

        Returns:
        - array of CUSUM scores, where each value is the maximum of the absolute values of the positive and negative cumulative sums.
        """   
        n = len(data)
        c_pos = np.zeros(n)  # CUSUM for increases
        c_neg = np.zeros(n)  # CUSUM for decreases
        
        for i in range(1, n):
            c_pos[i] = max(0, c_pos[i-1] + data[i] - reference_mean - k*reference_std)
            c_neg[i] = min(0, c_neg[i-1] + data[i] - reference_mean + k*reference_std)
        
        return np.maximum(np.abs(c_pos), np.abs(c_neg))