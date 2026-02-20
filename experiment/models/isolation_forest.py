from sklearn.ensemble import IsolationForest 
from data_transformation import calculate_labels
from models.model import AnomalyModel

class IsolationForestModel(AnomalyModel):
    """ Class for Isolation Forest model"""

    def get_results(self):
        
        _, contaminated_dfs = self.load_datasets()

        results = {}
        
        for i in range(len(contaminated_dfs)):
            node = contaminated_dfs[i]['node'].iloc[0] # get node number (should be the same for all rows inside one dataframe)
            node = str(node)
        
            X = contaminated_dfs[i][['chlorine_concentration']]
            
            contamination = self.config.model_params.get("contamination", "auto")
            
            model = IsolationForest(contamination=contamination, random_state=42)
            y_pred = model.fit_predict(X)
            y_true = calculate_labels(contaminated_dfs[i], self.config.contaminant.value, self.config.window_size)
            
            results[node] = {"y_true": y_true, "y_pred": y_pred}
        
        return results