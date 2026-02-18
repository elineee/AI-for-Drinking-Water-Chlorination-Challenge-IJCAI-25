from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

from typing import List

from sklearn.preprocessing import MinMaxScaler
from data_transformation import aggregate_data_for_several_nodes, change_data_format, create_features_2, get_data_for_one_node, calculate_labels, create_features, remove_first_x_days
from models.model import AnomalyModel

# based on https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/README.md

""" Class for SVR model"""
class SVRModel(AnomalyModel):
    def __init__(self, config):
        super().__init__(config)
    
    def get_results(self):
        all_clean_dfs, all_contaminated_dfs = self.load_datasets()
        
        results = {}
        
        # iterate over the nodes
        for key, value in all_clean_dfs.items():
            print(key)
            node = key
            
            # get contaminated dataset for the same key node

            contaminated_dfs = all_contaminated_dfs[key]
            clean_dfs = value

            train = []
            test = []
            
            new_clean_dfs = []
            # iterate over datasets for each node
            for i in range(len(clean_dfs)): 
                train_data = remove_first_x_days(clean_dfs[i], 3)
                new_clean_dfs.append(train_data)
                train_data = create_features_2(train_data, self.config.disinfectant.value, self.config.window_size)
                train.extend(train_data)
            train = np.array(train)
              
            new_contaminated_dfs = []
            for i in range(len(contaminated_dfs)): 
                test_data = remove_first_x_days(contaminated_dfs[i], 3)
                new_contaminated_dfs.append(test_data)
                test_data = create_features_2(test_data, self.config.disinfectant.value, self.config.window_size)
                test.extend(test_data)
            test = np.array(test)

        
        # # TODO : ici, une boucle = gère pour un noeud
        # for i in range(len(clean_dfs)):
        #     node = clean_dfs[i]['node'].iloc[0] # get node number (should be the same for all rows inside one dataframe)
        #     node = str(node)

        #     train = clean_dfs[i]
        #     test = contaminated_dfs[i]
            
            
        #     train = create_features_2(train, self.config.disinfectant.value, self.config.window_size)
        #     test = create_features_2(test, self.config.disinfectant.value, self.config.window_size)
            
            
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
            

            # train_timestamps = dataset.iloc[:-self.config.window_size+1]['timestep']
            
            train_timestamps = []
            for dataset in new_clean_dfs:
                train_timestamps += list(range(len(dataset)-self.config.window_size+1))
            
            plt.figure(figsize=(25,6))
            plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
            plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
            plt.legend(['Actual','Predicted'])
            plt.xlabel('Timestamp')
            plt.title("Training data prediction")
            plt.show()
                
            test_timestamps = []
            for dataset in new_contaminated_dfs:
                test_timestamps += list(range(len(dataset)-self.config.window_size+1))
                # test_timestamps += dataset.iloc[:-self.config.window_size+1:]['timestep']
              
            plt.figure(figsize=(10,3))
            plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
            plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
            plt.legend(['Actual','Predicted'])
            plt.xlabel('Timestamp')
            plt.show()
            
            y_true = calculate_labels(new_contaminated_dfs[i], self.config.contaminants[0].value, self.config.window_size-1)
            
            ok = []
            ano = []
            for element in y_true:
                if element == 1:
                    ok.append(element)
                else:
                    ano.append(element)
            print(f"ok: {len(ok)}, ano: {len(ano)}")
            
            y_pred = self.get_anomalies(y_test_pred, y_test, 0.25) 
            
            
            results[node] = {
                "y_true": y_true,
                "y_pred": y_pred
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
    
    def get_anomalies(self, y_pred, y_true, threshold):
        """Get the anomalies based on the predictions and the true values. An anomaly is detected if the absolute difference between the predicted value and the true value is greater than the threshold."""
        anomalies = []
        for i in range(len(y_true)):
            if abs(y_pred[i] - y_true[i]) > threshold:
                anomalies.append(1)
            else:
                anomalies.append(-1)
        return np.array(anomalies)
    
    
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
        
        dfs = {}
        
        if self.config.aggregate_method is None:
            for node in nodes:
                df_node = get_data_for_one_node(df_all, node, to_csv=False)
                dfs[str(node)] = df_node
        
        else:
            df = aggregate_data_for_several_nodes(df_all, nodes, method=self.config.aggregate_method, to_csv=False)
            dfs[str(nodes)] = df 
        
        return dfs

    def load_datasets(self):
        """Return dico of dataframes for each contaminated and example file where keys are nodes concern by dataframe.""" 
        
        example_dfs = {}
        if self.config.example_files is not None:
            for fp in self.config.example_files:
                dfs = self.load_and_filter(fp, self.config.nodes)
                for key, value in dfs.items():
                    if example_dfs.get(key) is None:
                        example_dfs[key] = [value]
                    else:
                        example_dfs[key].append(value)
                # example_dfs.extend(self.load_and_filter(fp, self.config.nodes))
                
        contaminated_dfs = {}
        for fp in self.config.contaminated_files:
            dfs = self.load_and_filter(fp, self.config.nodes)
            
            # add df to corresponding node in the dico, each node can have several dataframes
            for key, value in dfs.items():
                if contaminated_dfs.get(key) is None:
                    contaminated_dfs[key] = [value]
                else:
                    contaminated_dfs[key].append(value)

            # contaminated_dfs.extend(self.load_and_filter(fp, self.config.nodes))
        
        return example_dfs, contaminated_dfs
    
