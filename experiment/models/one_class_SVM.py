from sklearn import svm

from typing import List

from sklearn.discriminant_analysis import StandardScaler
from data_transformation import aggregate_data_for_several_nodes, change_data_format, get_data_for_one_node, calculate_labels, create_features, create_features_2
from models.model import AnomalyModel


""" Class for One Class SVM model"""
class OneClassSVMModel(AnomalyModel):
    def __init__(self, config):
        super().__init__(config)
    
    def get_results(self):
        clean_dfs, contaminated_dfs = self.load_datasets()
        
        results = {}
        
        for i in range(len(contaminated_dfs)):
            node = contaminated_dfs[i]['node'].iloc[0] # get node number (should be the same for all rows inside one dataframe)
            node = str(node)
            print(contaminated_dfs[i]['arsenic_concentration'][40:60])

            X_train = create_features(clean_dfs[i], self.config.disinfectant.value, self.config.window_size)

            X_test = create_features(contaminated_dfs[i], self.config.disinfectant.value, self.config.window_size)

            # TODO : handle multiple contaminants, for now only one contaminant is handled
            y_true = calculate_labels(contaminated_dfs[i], self.config.contaminants[0].value, self.config.window_size)
            
            # standardize the features before applying the model 
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train) 
            X_test = scaler.transform(X_test)
            

            # larger gamma if more complex patterns of anomalies
            gamma = self.config.model_params.get("gamma", "scale")
            # smaller nu if scenarios with a lot of anomalies
            nu = self.config.model_params.get("nu", 0.1)
    
            kernel = self.config.model_params.get("kernel", "rbf")
            if kernel == "poly":
                degree = self.config.model_params.get("degree", 4)
                ocsvm = svm.OneClassSVM(kernel=kernel, gamma=gamma, nu=nu, degree=degree)
            else: 
                ocsvm = svm.OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)

            ocsvm.fit(X_test)

            y_pred = ocsvm.predict(X_test)
            
            results[node] = {
                "y_true": y_true,
                "y_pred": y_pred
            }
        
            
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