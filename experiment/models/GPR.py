import pandas as pd
import numpy as np 
from scipy.stats import norm 
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from models.model import AnomalyModel

# https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#gaussianprocessregressor


class GPRModel(AnomalyModel):
    """ Class for Gaussian Process Regressor model"""

    def _prepare_data(self, clean_dfs: list[pd.DataFrame], contaminated_dfs: list[pd.DataFrame]):
        """
        Prepares and scales train/test data for the Gaussian Process Regressor model.

        Parameters:
        - clean_dfs: dataframes with training data (clean data)
        - contaminated_dfs: dataframes with testing data (contaminated data)

        Returns:
        - x_train: scaled training features (shape (number of train instances, number of features))
        - y_train: scaled training targets (shape (number of train instances, 1))
        - x_test: scaled test features (shape (number of train instances, number of features))
        - y_test: scaled test targets (shape (number of train instances, 1))
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

        return x_train, y_train, x_test, y_test, prepared_clean_dfs, prepared_contaminated_dfs


    def run_model(self, node: str, clean_dfs: list[pd.DataFrame], contaminated_dfs: list[pd.DataFrame]):
        """
        Trains the Gaussian Process Regressor model on clean data and detects anomalies on contaminated data.

        Parameters:
        - node: the node id
        - clean_dfs: dataframes with training data (clean data)
        - contaminated_dfs: dataframes with testing data (contaminated data)

        Returns:
        - y_true: true labels (shape(number of test instances,))
        - y_pred: predicted labels (-1 for anomaly, 1 for normal) (shape(number of test instances,))
        - y_test: actual test values (shape(number of test instances,1))
        - y_test_pred: predicted test values (shape(number of test instances,))
        """
        x_train, y_train, x_test, y_test, prepared_clean_dfs, prepared_contaminated_dfs = self._prepare_data(clean_dfs, contaminated_dfs)
        prepared_contaminated_df = pd.concat(prepared_contaminated_dfs)  
        y_true = self._calculate_labels(prepared_contaminated_df, self.config.contaminants[0].value, self.config.window_size)
        y_true = np.array(y_true)

        kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e4))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        gpr.fit(x_train, y_train)

        train_mean, train_std = gpr.predict(x_train, return_std = True)
        test_mean, test_std = gpr.predict(x_test, return_std=True)

        # Compute likelihood 
        train_likelihood = norm.logpdf(y_train.flatten(), loc=train_mean.flatten(), scale=train_std)
        test_likelihood = norm.logpdf(y_test.flatten(), loc=test_mean.flatten(), scale=test_std)

        # Detect anomalies 
        threshold = np.percentile(train_likelihood, 0.1)
        y_pred = np.where(test_likelihood < threshold, -1, 1)
    
        return y_true, y_pred, y_test


    def get_results(self):
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()
        
        results = {}
        
        for node, clean_dfs in all_clean_dfs.items():
            contaminated_dfs = all_contaminated_dfs[node]
            
            y_true, y_pred, _ = self.run_model(node, clean_dfs, contaminated_dfs)
            y_pred = self._post_predictions(y_pred)

            results[node] = {"y_true": y_true, "y_pred": y_pred}
                
        return results
