import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from data_transformation import calculate_labels, create_features, remove_first_x_days
from models.model import AnomalyModel

class LOFModel(AnomalyModel):
    """ Class for Local Outlier Factor (LOF) model"""
  
    def get_results(self):
        
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()
        
        results = {}
        
        for key, value in all_clean_dfs.items():
            print(key)
            node = key
            
            contaminated_dfs = all_contaminated_dfs[key]
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
            
            y_true = calculate_labels(new_contaminated_dfs[i], self.config.contaminants[0].value, self.config.window_size)
            
            n_neighbors = self.config.model_params.get("n_neighbors", 20)
            contamination = self.config.model_params.get("contamination", 0.1)
            
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=contamination)
            lof.fit(X_train)

            y_pred = lof.predict(X_test)
            
            results[node] = {"y_true": y_true,"y_pred": y_pred }
            
        return results
        
        
        clean_dfs, contaminated_dfs = self.load_datasets()
        
        results = {}
        
        for i in range(len(contaminated_dfs)):
            node = contaminated_dfs[i]['node'].iloc[0] # get node number (should be the same for all rows inside one dataframe)
            node = str(node)

            X_train = create_features(clean_dfs[i], self.config.disinfectant.value, self.config.window_size)
            X_test = create_features(contaminated_dfs[i], self.config.disinfectant.value, self.config.window_size)

            # TODO : handle multiple contaminants?
            y_true = calculate_labels(contaminated_dfs[i], self.config.contaminants[0].value, self.config.window_size)
            
            n_neighbors = self.config.model_params.get("n_neighbors", 20)
            contamination = self.config.model_params.get("contamination", 0.1)
            
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=contamination)
            lof.fit(X_train)

            y_pred = lof.predict(X_test)
            
            results[node] = {"y_true": y_true,"y_pred": y_pred }
        
        return results