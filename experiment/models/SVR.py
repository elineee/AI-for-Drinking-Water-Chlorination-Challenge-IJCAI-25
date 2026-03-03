from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from data_transformation import calculate_labels, create_extended_features, remove_first_x_days
from utils import plot_prediction, build_timestamps
from models.model import AnomalyModel

# based on https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/README.md

class SVRModel(AnomalyModel):
    """ Class for SVR model"""
    
    def predict(self, node, clean_dfs, contaminated_dfs): 
        train = []
        test = []
        
        new_clean_dfs = []
        # iterate over datasets for each node
        for i in range(len(clean_dfs)): 
            train_data = remove_first_x_days(clean_dfs[i], 3)
            new_clean_dfs.append(train_data)
            train_data = create_extended_features(train_data, self.config.disinfectant.value, self.config.window_size)
            train.extend(train_data)
        train = np.array(train)
            
        new_contaminated_dfs = []
        for i in range(len(contaminated_dfs)): 
            test_data = remove_first_x_days(contaminated_dfs[i], 3)
            new_contaminated_dfs.append(test_data)
            test_data = create_extended_features(test_data, self.config.disinfectant.value, self.config.window_size)
            test.extend(test_data)
        test = np.array(test)
        
        # get x and y to train on 
        x_train = np.array([row[:-1] for row in train])
        y_train = np.array([row[-1] for row in train]).reshape(-1,1)

        # scale the data, two separate scalers are needed to be able to inverse transform the predictions later "cause different shapes
        scaler_x = StandardScaler()
        x_train = scaler_x.fit_transform(x_train)
        scaler_y = StandardScaler()
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
        x_test = scaler_x.transform(x_test)
        y_test = scaler_y.transform(y_test)
        
        # reshape y to be in the right format for evaluation
        y_train_pred = model.predict(x_train).reshape(-1,1)
        y_test_pred = model.predict(x_test).reshape(-1,1)
        
        # inverse transform the predictions to get them back in the original scale
        y_train_pred = scaler_y.inverse_transform(y_train_pred)
        y_test_pred = scaler_y.inverse_transform(y_test_pred)
        
        y_train = scaler_y.inverse_transform(y_train)
        y_test = scaler_y.inverse_transform(y_test)
        
        # Plots
        train_timestamps = build_timestamps(new_clean_dfs, self.config.window_size)
        plot_prediction( train_timestamps, y_train, y_train_pred, title=f"Training prediction node {node} ")

        test_timestamps = build_timestamps(new_contaminated_dfs, self.config.window_size)
        plot_prediction( test_timestamps, y_test, y_test_pred,title=f"Test prediction node {node}")
                    
        y_true = calculate_labels(new_contaminated_dfs[i], self.config.contaminants[0].value, self.config.window_size)
        
        y_true = np.array(y_true)
        print(f"ok: {(y_true == 1).sum()}, ano: {(y_true == -1).sum()}")
        
        
        #calculate the threshold for anomaly detection based on the training data residuals (difference between predicted and true values)
        residual_train = np.abs(y_train - y_train_pred)
        threshold = residual_train.mean() + 60 * residual_train.std()
        
        print(f"Threshold: {threshold:.4f}")
        print(len(y_test), len(y_test_pred))

        residual_test = np.abs(y_test - y_test_pred)
        y_pred = np.where(residual_test > threshold, -1, 1)
        
        return y_true, y_pred, y_test, y_test_pred      
                
    def get_results(self):
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()
        
        results = {}
        
        # iterate over the nodes
        for node, clean_dfs in all_clean_dfs.items():
            
            # get contaminated dataset for the same node
            contaminated_dfs = all_contaminated_dfs[node]
            
            y_true, y_pred, _, _ = self.predict(node, clean_dfs, contaminated_dfs)
  
            results[node] = {"y_true": y_true, "y_pred": y_pred}
                
        return results

    
    def get_best_params(self, x_train, y_train):
        """
        Uses grid search to find the best hyperparameters for the SVR model.
        """
        
        param_grid = {
            "C": [0.1, 1, 10, 50, 100, 500, 1000],
            "epsilon": [0.001, 0.01, 0.1, 0.5, 1],
            "gamma": ["scale", 0.0001, 0.001, 0.01, 0.1, 1],
            "kernel": ["rbf"]
        }

        grid = GridSearchCV(SVR(), param_grid, cv=5, scoring="neg_mean_squared_error")
        grid.fit(x_train, y_train.ravel())
        
        return grid.best_params
    