import pandas as pd 
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from models.model import AnomalyModel

class OneClassSVMModel(AnomalyModel):
    """ Class for One Class SVM model"""

    def _prepare_data(self, clean_dfs: list[pd.DataFrame], contaminated_dfs: list[pd.DataFrame]):
        """
        Prepares and scales train/test data for the One Class SVM model.

        Parameters:
        - clean_dfs: dataframes with training data (clean data)
        - contaminated_dfs: dataframes with testing data (contaminated data)

        Returns:
        - X_train: scaled training features
        - X_test: scaled test features
        - prepared_contaminated_dfs: contaminated dataframes after preprocessing
        """
        _, X_train = self._prepare_dataset(clean_dfs)
        prepared_contaminated_dfs, X_test = self._prepare_dataset(contaminated_dfs)

        # Standardize data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, prepared_contaminated_dfs


    def get_results(self):
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()
        
        results = {}
        
        for node, clean_dfs in all_clean_dfs.items():
            
            contaminated_dfs = all_contaminated_dfs[node]
            X_train, X_test, prepared_contaminated_dfs = self._prepare_data(clean_dfs, contaminated_dfs)
            prepared_contaminated_df = pd.concat(prepared_contaminated_dfs)

            y_true = self._calculate_labels(prepared_contaminated_df, self.config.contaminants[0].value, self.config.window_size)

            # Larger gamma if more complex patterns of anomalies
            gamma = self.config.model_params.get("gamma", "scale")
            # Smaller nu if scenarios with a lot of anomalies
            nu = self.config.model_params.get("nu", 0.1)
    
            kernel = self.config.model_params.get("kernel", "rbf")
            
            # Add degree parameter if the kernel is polynomial
            if kernel == "poly":
                degree = self.config.model_params.get("degree", 4)
                ocsvm = svm.OneClassSVM(kernel=kernel, gamma=gamma, nu=nu, degree=degree)
            else: 
                ocsvm = svm.OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)

            ocsvm.fit(X_train)

            y_pred = ocsvm.predict(X_test)
            y_pred = self._post_predictions(y_pred)
            
            results[node] = {"y_true": y_true, "y_pred": y_pred}
        
        return results