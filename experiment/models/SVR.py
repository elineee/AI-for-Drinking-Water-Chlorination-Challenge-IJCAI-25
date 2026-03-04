import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from data_transformation import calculate_labels
from utils import plot_prediction, build_timestamps
from models.model import AnomalyModel

# based on https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/README.md

class SVRModel(AnomalyModel):
    """ Class for SVR model"""
    
    def _get_threshold_multiplier(self):
        return 60
    

    def _prepare_data(self, clean_dfs: list[pd.DataFrame], contaminated_dfs: list[pd.DataFrame]):
        """
        Prepares and scales train/test data for the SVR model.

        Parameters:
        - clean_dfs: dataframes with training data (clean data)
        - contaminated_dfs: dataframes with testing data (contaminated data)

        Returns:
        - x_train: scaled training features
        - y_train: scaled training targets
        - x_test: scaled test features
        - y_test: scaled test targets
        - scaler_y: fitted scaler for y
        - prepared_clean_dfs: clean dataframes after preprocessing
        - prepared_contaminated_dfs: contaminated dataframes after preprocessing
        """
        prepared_clean_dfs, X_train = self._prepare_dataset(clean_dfs, feature_type="extended")
        prepared_contaminated_dfs, X_test = self._prepare_dataset(contaminated_dfs, feature_type="extended")

        # Get x and y to train on and x and y to test on
        x_train = np.array([row[:-1] for row in X_train])
        y_train = np.array([row[-1] for row in X_train]).reshape(-1, 1)

        x_test = np.array([row[:-1] for row in X_test])
        y_test = np.array([row[-1] for row in X_test]).reshape(-1, 1)

        # Scale the data 
        # Two separate scalers are needed to inverse transform the predictions later because different shapes
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        x_train = scaler_x.fit_transform(x_train)
        y_train = scaler_y.fit_transform(y_train)

        x_test = scaler_x.transform(x_test)
        y_test = scaler_y.transform(y_test)

        return x_train, y_train, x_test, y_test, scaler_y, prepared_clean_dfs, prepared_contaminated_dfs


    def predict(self, node: str, clean_dfs: list[pd.DataFrame], contaminated_dfs: list[pd.DataFrame]):
        """
        Trains the SVR model on clean data and detects anomalies on contaminated data.

        Parameters:
        - node: the node id
        - clean_dfs: dataframes with training data (clean data)
        - contaminated_dfs: dataframes with testing data (contaminated data)

        Returns:
        - y_true: true labels
        - y_pred: predicted labels (-1 for anomaly, 1 for normal)
        - y_test: actual test values
        - y_test_pred: predicted test values
        """
        x_train, y_train, x_test, y_test, scaler_y, prepared_clean_dfs, prepared_contaminated_dfs = self._prepare_data(clean_dfs, contaminated_dfs)
        prepared_contaminated_df = pd.concat(prepared_contaminated_dfs)  

        # params = self.get_best_params(x_train, y_train)
        # print(params)
        
        gamma = self.config.model_params.get("gamma", "scale")
        kernel = self.config.model_params.get("kernel", "rbf")
        C = self.config.model_params.get("C", 10)
        epsilon = self.config.model_params.get("epsilon", 0.01)
        
        model = SVR(kernel=kernel,gamma=gamma, C=C, epsilon = epsilon)
        
        model.fit(x_train, y_train.ravel())
        
        # Reshape y to be in the right format for evaluation
        y_train_pred = model.predict(x_train).reshape(-1,1)
        y_test_pred = model.predict(x_test).reshape(-1,1)
        
        # Inverse transform the predictions to get them back in the original scale
        y_train_pred = scaler_y.inverse_transform(y_train_pred)
        y_test_pred = scaler_y.inverse_transform(y_test_pred)
        
        y_train = scaler_y.inverse_transform(y_train)
        y_test = scaler_y.inverse_transform(y_test)
        
        # Plots
        train_timestamps = build_timestamps(prepared_clean_dfs, self.config.window_size)
        plot_prediction( train_timestamps, y_train, y_train_pred, title=f"Training prediction node {node} ")

        test_timestamps = build_timestamps(prepared_contaminated_dfs, self.config.window_size)
        plot_prediction( test_timestamps, y_test, y_test_pred,title=f"Test prediction node {node}")
                    
        y_true = self._calculate_labels(prepared_contaminated_df, self.config.contaminants[0].value, self.config.window_size)
        
        y_true = np.array(y_true)
        print(f"ok: {(y_true == 1).sum()}, ano: {(y_true == -1).sum()}")
        
        #calculate the threshold for anomaly detection based on the training data residuals (difference between predicted and true values)
        residual_train = np.abs(y_train - y_train_pred)
        threshold = residual_train.mean() + self._get_threshold_multiplier() * residual_train.std()

        print(f"Threshold: {threshold:.4f}")

        residual_test = np.abs(y_test - y_test_pred)
        y_pred = np.where(residual_test > threshold, -1, 1)
        
        return y_true, y_pred, y_test, y_test_pred      


    def get_best_params(self, x_train, y_train):
        """
        Uses grid search to find the best hyperparameters for the SVR model.
        
        Parameters:
        - x_train: scaled training features
        - y_train: scaled training targets

        Returns:
        - dictionary of best hyperparameters found
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
    

    def get_results(self):
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()
        
        results = {}
        
        for node, clean_dfs in all_clean_dfs.items():
            
            contaminated_dfs = all_contaminated_dfs[node]
            
            y_true, y_pred, _, _ = self.predict(node, clean_dfs, contaminated_dfs)
            y_pred = self._post_predictions(y_pred)

            results[node] = {"y_true": y_true, "y_pred": y_pred}
                
        return results

    
