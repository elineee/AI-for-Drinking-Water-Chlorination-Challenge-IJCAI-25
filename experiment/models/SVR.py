from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

from typing import List

from sklearn.preprocessing import MinMaxScaler
from data_transformation import aggregate_data_for_several_nodes, change_data_format, create_features_2, get_data_for_one_node, calculate_labels, create_features
from models.model import AnomalyModel

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
            
            
            # get x and y to train on 
            x_train = np.array([row[:-1] for row in train])
            y_train = np.array([row[-1] for row in train]).reshape(-1,1)

            # scale the data, two separate scalers are needed to be able to inverse transform the predictions later 'cause different shapes
            scaler_X = MinMaxScaler()
            x_train = scaler_X.fit_transform(x_train)
            scaler_y = MinMaxScaler()
            y_train = scaler_y.fit_transform(y_train)
            
            # params = self.get_best_params(x_train, y_train)
            # print(params)
            
            gamma = self.config.model_params.get("gamma", "scale")
            kernel = self.config.model_params.get("kernel", "rbf")
            C = self.config.model_params.get("C", 10)
            epsilon = self.config.model_params.get("epsilon", 0.01)
            
            model = SVR(kernel=kernel,gamma=gamma, C=C, epsilon = epsilon)
            
            model.fit(x_train, y_train.ravel())
            
            # get x and y to test on
            x_test = np.array([row[:-1] for row in test])
            y_test = np.array([row[-1] for row in test]).reshape(-1,1)
            
            # scale the test data using the same scalers as for the training data
            x_test = scaler_X.transform(x_test)
            y_test = scaler_y.transform(y_test)
            
            # reshape y to be in the right format for evaluation
            y_train_pred = model.predict(x_train).reshape(-1,1)
            y_test_pred = model.predict(x_test).reshape(-1,1)
            
            # inverse transform the predictions to get them back in the original scale
            y_train_pred = scaler_y.inverse_transform(y_train_pred)
            y_test_pred = scaler_y.inverse_transform(y_test_pred)
            
            y_train = scaler_y.inverse_transform(y_train)
            y_test = scaler_y.inverse_transform(y_test)
            
            # get timestamps for plotting
            train_timestamps = clean_dfs[i].iloc[:800-self.config.window_size+1]['timestep']
            test_timestamps = clean_dfs[i].iloc[800+self.config.window_size-1:]['timestep']
            
            plt.figure(figsize=(25,6))
            plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
            plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
            plt.legend(['Actual','Predicted'])
            plt.xlabel('Timestamp')
            plt.title("Training data prediction")
            plt.show()
            
            plt.figure(figsize=(10,3))
            plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
            plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
            plt.legend(['Actual','Predicted'])
            plt.xlabel('Timestamp')
            plt.show()
            
            
            results[node] = {
                "y_true": y_test,
                "y_pred": y_test_pred
            }
        
            
        return results
    
    def get_best_params(self, x_train, y_train):
        """Use grid search to find the best hyperparameters for the SVR model."""
        
        param_grid = {
            'C': [0.1, 1, 10, 50, 100, 500, 1000],
            'epsilon': [0.001, 0.01, 0.1, 0.5, 1],
            'gamma': ['scale', 0.0001, 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf']
        }

        grid = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error')

        grid.fit(x_train, y_train.ravel())
        
        return grid.best_params_
    
    
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
    
