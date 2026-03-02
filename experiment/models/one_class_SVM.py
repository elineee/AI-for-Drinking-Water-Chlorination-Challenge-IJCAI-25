import pandas as pd 
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from data_transformation import calculate_labels
from models.model import AnomalyModel

class OneClassSVMModel(AnomalyModel):
    """ Class for One Class SVM model"""

    def get_results(self):
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()
        
        results = {}
        
        for node, clean_dfs in all_clean_dfs.items():
            
            contaminated_dfs = all_contaminated_dfs[node]
    
            
            # create features and concatenate the datasets for each node 
            _ , X_train = self._prepare_dataset(clean_dfs)
            new_contaminated_dfs, X_test = self._prepare_dataset(contaminated_dfs)
            new_contaminated_df = pd.concat(new_contaminated_dfs)

            y_true = calculate_labels(new_contaminated_df, self.config.contaminants[0].value, self.config.window_size)
            
            # standardize the data before applying the model
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train) 
            X_test = scaler.transform(X_test)
            
            # larger gamma if more complex patterns of anomalies
            gamma = self.config.model_params.get("gamma", "scale")
            # smaller nu if scenarios with a lot of anomalies
            nu = self.config.model_params.get("nu", 0.1)
    
            kernel = self.config.model_params.get("kernel", "rbf")
            
            # add degree parameter if the kernel is polynomial
            if kernel == "poly":
                degree = self.config.model_params.get("degree", 4)
                ocsvm = svm.OneClassSVM(kernel=kernel, gamma=gamma, nu=nu, degree=degree)
            else: 
                ocsvm = svm.OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)

            ocsvm.fit(X_train)

            y_pred = ocsvm.predict(X_test)
            
            results[node] = {"y_true": y_true, "y_pred": y_pred}
        
        return results