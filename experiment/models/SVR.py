from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from data_transformation import calculate_labels, remove_first_x_days
from models.model import AnomalyModel

# based on https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/README.md

""" Class for SVR model"""
class SVRModel(AnomalyModel):
    def __init__(self, config):
        super().__init__(config)
    
    def get_results(self):
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()
        
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
                train_data = self.create_features_svr(train_data, self.config.disinfectant.value, self.config.window_size)
                train.extend(train_data)
            train = np.array(train)
              
            new_contaminated_dfs = []
            for i in range(len(contaminated_dfs)): 
                test_data = remove_first_x_days(contaminated_dfs[i], 3)
                new_contaminated_dfs.append(test_data)
                test_data = self.create_features_svr(test_data, self.config.disinfectant.value, self.config.window_size)
                test.extend(test_data)
            test = np.array(test)
            
            # get x and y to train on 
            x_train = np.array([row[:-1] for row in train])
            y_train = np.array([row[-1] for row in train]).reshape(-1,1)

            # scale the data, two separate scalers are needed to be able to inverse transform the predictions later 'cause different shapes
            scaler_X = StandardScaler()
            x_train = scaler_X.fit_transform(x_train)
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
                train_timestamps += list(range(len(dataset)-self.config.window_size))
            
            plt.figure(figsize=(25,6))
            plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
            plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
            plt.legend(['Actual','Predicted'])
            plt.xlabel('Timestamp')
            plt.title("Training data prediction")
            plt.show()
                
            test_timestamps = []
            for dataset in new_contaminated_dfs:
                test_timestamps += list(range(len(dataset)-self.config.window_size))
                # test_timestamps += dataset.iloc[:-self.config.window_size+1:]['timestep']
              
            plt.figure(figsize=(10,3))
            plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
            plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
            plt.legend(['Actual','Predicted'])
            plt.xlabel('Timestamp')
            plt.show()
            
            y_true = calculate_labels(new_contaminated_dfs[i], self.config.contaminants[0].value, self.config.window_size)
            
            ok = []
            ano = []
            for element in y_true:
                if element == 1:
                    ok.append(element)
                else:
                    ano.append(element)
            print(f"ok: {len(ok)}, ano: {len(ano)}")
            
            #calculate the threshold for anomaly detection based on the training data residuals (difference between predicted and true values)
            residual_train = np.abs(y_train - y_train_pred)
            threshold = residual_train.mean() + 60 * residual_train.std()
            
            print(f"Threshold: {threshold:.4f}")
            
            print(len(y_test), len(y_test_pred))
            residual_test = np.abs(y_test - y_test_pred)
            y_pred = np.where(residual_test > threshold, -1, 1)
  
            results[node] = {"y_true": y_true, "y_pred": y_pred}
                
        return results

    def create_features_svr(self, df: pd.DataFrame, feature_column: str, window_size: int = 10):
        for column in df.columns:
            if feature_column in column:
                feature_column = column
                break
        
        feature = df[feature_column].values
        features = []
        
        for i in range(window_size, len(feature)):
            window = feature[i-window_size:i]
            
            current_value = feature[i]
            mean = window.mean()
            std = window.std()
            slope = window[-1] - window[0]
            delta = feature[i] - feature[i-1]
            
            row = np.concatenate([
                window,              
                [mean, std, slope, delta, current_value]
            ])
            
            features.append(row)
        
        return np.array(features)

    
    def get_best_params(self, x_train, y_train):
        """
        Uses grid search to find the best hyperparameters for the SVR model.
        """
        
        param_grid = {
            'C': [0.1, 1, 10, 50, 100, 500, 1000],
            'epsilon': [0.001, 0.01, 0.1, 0.5, 1],
            'gamma': ['scale', 0.0001, 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf']
        }

        grid = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error')
        grid.fit(x_train, y_train.ravel())
        
        return grid.best_params
    