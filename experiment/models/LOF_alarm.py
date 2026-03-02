import pandas as pd
from models.LOF import LOFModel
from sklearn.neighbors import LocalOutlierFactor
from utils import detect_change_point
from data_transformation import calculate_labels_alarm

class LOFAlarmModel(LOFModel):
    """ Class for Local Outlier Factor (LOF) model"""
    
    def get_results(self):
        
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()
        
        results = {}
        
        for node, clean_dfs in all_clean_dfs.items():
            
            contaminated_dfs = all_contaminated_dfs[node]
            
            # create features and concatenate the datasets for each node 
            _ , X_train = self._prepare_dataset(clean_dfs)
            new_contaminated_dfs, X_test = self._prepare_dataset(contaminated_dfs)
            new_contaminated_df = pd.concat(new_contaminated_dfs)

            n_neighbors = self.config.model_params.get("n_neighbors", 20)
            contamination = self.config.model_params.get("contamination", 0.1)
            
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=contamination)
            lof.fit(X_train)
            
            y_true = calculate_labels_alarm(new_contaminated_df, self.config.contaminants[0].value, self.config.window_size)
            y_pred_temp = lof.predict(X_test)
            y_pred = detect_change_point(y_pred_temp)
            
            results[node] = {"y_true": y_true,"y_pred": y_pred }
            