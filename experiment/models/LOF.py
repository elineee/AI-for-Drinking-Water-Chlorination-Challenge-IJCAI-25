import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from data_transformation import calculate_labels, create_features, remove_first_x_days
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
            
            # create features and concatenate the example datasets for each node 
            new_clean_dfs = []
            for i in range(len(clean_dfs)):
                train_data = remove_first_x_days(clean_dfs[i], 3)
                new_clean_dfs.append(train_data)
                train_data = create_features(train_data, self.config.disinfectant.value, self.config.window_size)
                X_train.extend(train_data)
            X_train = np.array(X_train)
            
            # create features and concatenate the contaminated datasets for each node
            new_contaminated_dfs = []
            for i in range(len(contaminated_dfs)):
                test_data = remove_first_x_days(contaminated_dfs[i], 3)
                new_contaminated_dfs.append(test_data)
                test_data = create_features(test_data, self.config.disinfectant.value, self.config.window_size)
                X_test.extend(test_data)

            X_test = np.array(X_test)
            new_contaminated_df = pd.concat(new_contaminated_dfs)
            y_true = calculate_labels(new_contaminated_df, self.config.contaminants[0].value, self.config.window_size)
            
            n_neighbors = self.config.model_params.get("n_neighbors", 20)
            contamination = self.config.model_params.get("contamination", 0.1)
            
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=contamination)
            lof.fit(X_train)

            y_pred = lof.predict(X_test)
            
            results[node] = {"y_true": y_true,"y_pred": y_pred }
            
        return results