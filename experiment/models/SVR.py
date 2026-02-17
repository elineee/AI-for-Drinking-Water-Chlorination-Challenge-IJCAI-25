from sklearn.svm import SVR

from typing import List

from sklearn.preprocessing import MinMaxScaler
from data_transformation import aggregate_data_for_several_nodes, change_data_format, create_features_2, get_data_for_one_node, calculate_labels, create_features
from model import AnomalyModel

# based on https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/README.md

""" Class for SVR model"""
class SVRModel(AnomalyModel):
    def __init__(self, config):
        super().__init__(config)
    
    def get_results(self):
        clean_dfs, _ = self.load_datasets()
        
        results = {}
        
        for i in range(len(clean_dfs)):
            node = clean_dfs[i]['node'].iloc[0] # get node number (should be the same for all rows inside one dataframe)
            node = str(node)
            train = clean_dfs[i].iloc[:800]
            test = clean_dfs[i].iloc[800:]
            
            
            train = create_features_2(train, self.config.disinfectant.value, self.config.window_size)
            test = create_features_2(test, self.config.disinfectant.value, self.config.window_size)
            
            scaler = MinMaxScaler()
            train = scaler.fit_transform(train)
            test = scaler.transform(test)
            print(train)
            print(test)
            
            # results[node] = {
            #     "y_true": y_true,
            #     "y_pred": y_pred
            # }
        
            
        return results
    
    def load_and_filter(self, file_path: str, nodes: List[int]):
        """Load the dataset from the given file path and filter it based on the specified nodes. Return a list of dataframes corresponding to each node.
        
        Parameters:
        - file_path: the path to the data file (csv) to load
        - nodes: a list of node numbers to filter the data by
        
        Returns:
        - a list of pandas DataFrames, each containing the data for one of the specified nodes
        """
        # TODO : add parameters contaminants when changed in function 
        df_all = change_data_format(file_path, self.config.contaminants, to_csv=False)  # returns rows with columns: timestep, node, chlorine_concentration, arsenic_concentration
        
        dfs = []
        
        if self.config.aggregate_method is None:
            for node in nodes:
                df_node = get_data_for_one_node(df_all, node, to_csv=False)
                dfs.append(df_node)
        
        else:
            df = aggregate_data_for_several_nodes(df_all, nodes, method=self.config.aggregate_method, to_csv=False)
            dfs.append(df)
        
        return dfs

    def load_datasets(self):
        """Return lists of dataframes for each contaminated and example file.""" 
        
        example_dfs = []
        if self.config.example_files is not None:
            for fp in self.config.example_files:
                example_dfs.extend(self.load_and_filter(fp, self.config.nodes))
                
        contaminated_dfs = []
        for fp in self.config.contaminated_files:
            contaminated_dfs.extend(self.load_and_filter(fp, self.config.nodes))
        
        return example_dfs, contaminated_dfs
    
