from sklearn.neighbors import LocalOutlierFactor
from data_transformation import calculate_labels, create_features
from models.model import AnomalyModel

class LOFModel(AnomalyModel):
    """ Class for Local Outlier Factor (LOF) model"""
  
    def get_results(self):
        clean_dfs, contaminated_dfs = self.load_datasets()
        
        results = {}
        
        for i in range(len(contaminated_dfs)):
            node = contaminated_dfs[i]['node'].iloc[0] # get node number (should be the same for all rows inside one dataframe)
            node = str(node)

            X_train = create_features(clean_dfs[i], self.config.disinfectant.value, self.config.window_size)
            X_test = create_features(contaminated_dfs[i], self.config.disinfectant.value, self.config.window_size)

            y_true = calculate_labels(contaminated_dfs[i], self.config.contaminant.value, self.config.window_size)
            
            n_neighbors = self.config.model_params.get("n_neighbors", 20)
            contamination = self.config.model_params.get("contamination", 0.1)
            
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=contamination)
            lof.fit(X_train)

            y_pred = lof.predict(X_test)
            
            results[node] = {"y_true": y_true,"y_pred": y_pred }
        
        return results