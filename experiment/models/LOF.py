import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from data_transformation import calculate_labels
from models.model import AnomalyModel


class LOFModel(AnomalyModel):
    """ Class for Local Outlier Factor (LOF) model"""
  
    def get_results(self):
        
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()
        
        results = {}
        
        for node, value in all_clean_dfs.items():

            contaminated_dfs = all_contaminated_dfs[node]
            clean_dfs = value
            
            X_train = []
            X_test = []
            
            # create features and concatenate the datasets for each node 
            _ , X_train = self._prepare_dataset(clean_dfs)
            new_contaminated_dfs, X_test = self._prepare_dataset(contaminated_dfs)
            new_contaminated_df = pd.concat(new_contaminated_dfs)

            y_true = calculate_labels(new_contaminated_df, self.config.contaminants[0].value, self.config.window_size)
            
            n_neighbors = self.config.model_params.get("n_neighbors", 20)
            contamination = self.config.model_params.get("contamination", 0.1)
            
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=contamination)
            lof.fit(X_train)

            y_pred = lof.predict(X_test)
            
            results[node] = {"y_true": y_true,"y_pred": y_pred }
            
        return results