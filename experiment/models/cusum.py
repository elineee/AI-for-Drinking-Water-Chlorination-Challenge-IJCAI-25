
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from models.model import AnomalyModel

class CusumModel(AnomalyModel):
    """ Class for CUSUM model"""
    
    def _get_threshold_multiplier(self):
        return 3
    

    def _prepare_data(self, clean_dfs: list[pd.DataFrame], contaminated_dfs: list[pd.DataFrame]):
        """
        Prepares train/test data for the CUSUM model.

        Parameters:
        - clean_dfs: dataframes with training data (clean data)
        - contaminated_dfs: dataframes with testing data (contaminated data)

        Returns:
        - X_train: flattened training features
        - X_test: flattened test features
        - prepared_contaminated_dfs: contaminated dataframes after preprocessing
        """

        _, X_train = self._prepare_dataset(clean_dfs, feature_type="extended", stats=False)
        prepared_contaminated_dfs, X_test = self._prepare_dataset(contaminated_dfs, feature_type="extended", stats=False)

        # Flatten the windows for CUSUM
        X_train = np.array(X_train).flatten()
        X_test = np.array(X_test).flatten()

        return X_train, X_test, prepared_contaminated_dfs


    def get_results(self):
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()
        
        results = {}
        
        for node, clean_dfs in all_clean_dfs.items():
            
            contaminated_dfs = all_contaminated_dfs[node] 
            X_train, X_test, prepared_contaminated_dfs = self._prepare_data(clean_dfs, contaminated_dfs)
            prepared_contaminated_df = pd.concat(prepared_contaminated_dfs)

            cusum_train = self.cusum(X_train, X_train.mean(), X_train.std())
            cusum_scores = self.cusum(X_test, X_train.mean(), X_train.std())
            
            threshold = cusum_train.mean() + self._get_threshold_multiplier() *cusum_train.std()
            print(f"Threshold: {threshold}")
            
            y_pred = np.where(cusum_scores > threshold, -1, 1)
            y_pred = self._post_predictions(y_pred)
            y_true = self._calculate_labels(prepared_contaminated_df, self.config.contaminants[0].value, self.config.window_size)

            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(X_test, label="Data")
            plt.plot(cusum_scores, label="CUSUM", color="red")
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