from sklearn import svm
from sklearn.discriminant_analysis import StandardScaler
from data_transformation import calculate_labels, create_features
from models.model import AnomalyModel

class OneClassSVMModel(AnomalyModel):
    """ Class for One Class SVM model"""

    def get_results(self):
        clean_dfs, contaminated_dfs = self.load_datasets()
        
        results = {}
        
        for i in range(len(contaminated_dfs)):
            node = contaminated_dfs[i]['node'].iloc[0] # get node number (should be the same for all rows inside one dataframe)
            node = str(node)

            X_train = create_features(clean_dfs[i], self.config.disinfectant.value, self.config.window_size)

            X_test = create_features(contaminated_dfs[i], self.config.disinfectant.value, self.config.window_size)

            y_true = calculate_labels(contaminated_dfs[i], self.config.contaminant.value, self.config.window_size)
            
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

            ocsvm.fit(X_train)

            y_pred = ocsvm.predict(X_test)
            
            results[node] = {"y_true": y_true, "y_pred": y_pred}
            
        return results