from sklearn.ensemble import IsolationForest
from typing import List  
from data_transformation import aggregate_data_for_several_nodes, change_data_format, get_data_for_one_node, calculate_labels
from experiment.models.model import AnomalyModel

class IsolationForestModel(AnomalyModel):
    """ Class for Isolation Forest model"""
    def __init__(self, config):
        super().__init__(config)
    
    
    def get_results(self):
        
        contaminated_dfs = self.load_datasets()
        
        results = {}
        
        for i in range(len(contaminated_dfs)):
            node = contaminated_dfs[i]['node'].iloc[0] # get node number (should be the same for all rows inside one dataframe)
            node = str(node)
        
            X = contaminated_dfs[i][['chlorine_concentration']]
            
            # TODO : voir pour mettre + de paramètres 
            contamination = self.config.model_params.get("contamination", "auto")
            
            
            model = IsolationForest(contamination=contamination, random_state=42)
            y_pred = model.fit_predict(X)
            
            y_true = calculate_labels(contaminated_dfs[i], self.config.contaminants[0].value, self.config.window_size)
            
            results[node] = {
                "y_true": y_true,
                "y_pred": y_pred
            }
        
        return results
    
    def load_and_filter(self, file_path: str, nodes: List[int]):
        """
        Loads the dataset from the given file path and filter it based on the specified nodes. Return a list of dataframes corresponding to each node.
        
        Parameters:
        - file_path: the path to the data file (csv) to load
        - nodes: a list of node numbers to filter the data by
        
        Returns:
        - dfs: a list of pandas DataFrames, each containing the data for one of the specified nodes
        """

        df_all = change_data_format(file_path, self.config.contaminants, to_csv=False)  
        
        dfs = []
        
        if self.config.aggregate_method is None:
            dfs = []
            for node in nodes:
                df_node = get_data_for_one_node(df_all, node, to_csv=False)
                dfs.append(df_node)
        
        else:
            df = aggregate_data_for_several_nodes(df_all, nodes, method=self.config.aggregate_method, to_csv=False)
            dfs.append(df)
        
        return dfs

    def load_datasets(self):
        """
        Loads the datasets for the experiment based on the contaminated files specified in the configuration. 
        For each contaminated file, it loads the data and filters it based on the specified nodes.
        
        Returns: 
        contamined_dfs : list of dataframes for each contaminated file.
        """     
        contaminated_dfs = []
        for fp in self.config.contaminated_files:
            contaminated_dfs.extend(self.load_and_filter(fp, self.config.nodes))
        
        return contaminated_dfs