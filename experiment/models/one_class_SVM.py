import numpy as np
from sklearn import svm
from sklearn.discriminant_analysis import StandardScaler
from data_transformation import calculate_labels, create_features, remove_first_x_days
from models.model import AnomalyModel

class OneClassSVMModel(AnomalyModel):
    """ Class for One Class SVM model"""

    def get_results(self):
        clean_dfs, contaminated_dfs = self.load_datasets()
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()
        
        results = {}
        
        for key, value in all_clean_dfs.items():
            print(key)
            node = key
            
            contaminated_dfs = all_contaminated_dfs[key]
            clean_dfs = value
            
            X_train = []
            X_test = []
            
            # create features and concatenate the example datasets for each node 
            new_clean_dfs = []
            for i in range(len(clean_dfs)):
                train_data = remove_first_x_days(clean_dfs[i], 3)
                new_clean_dfs.append(train_data)
                train_data = create_features(train_data, self.config.disinfectant.value, self.config.window_size)
                X_train.extend(train_data)
            X_train = np.array(X_train)
            
            # create features and concatenate the contaminated datasets for each node
            new_contaminated_dfs = []
            for i in range(len(contaminated_dfs)):
                test_data = remove_first_x_days(contaminated_dfs[i], 3)
                new_contaminated_dfs.append(test_data)
                test_data = create_features(test_data, self.config.disinfectant.value, self.config.window_size)
                X_test.extend(test_data)
            X_test = np.array(X_test)
            
            y_true = calculate_labels(new_contaminated_dfs[i], self.config.contaminants[0].value, self.config.window_size)
            
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