import numpy as np
from sklearn import svm
from sklearn.discriminant_analysis import StandardScaler
from data_transformation import calculate_labels, calculate_labels_alarm, create_features, remove_first_x_days
from models.model import AnomalyModel

class OneClassSVMAlarmModel(AnomalyModel):
    """ Class for One Class SVM model"""

    def get_results(self):
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
            
            y_true = calculate_labels_alarm(new_contaminated_dfs[i], self.config.contaminants[0].value, self.config.window_size)

            y_pred_temp = ocsvm.predict(X_test)
            
            y_pred = self.detect_change_points(y_pred_temp)
            
            results[node] = {"y_true": y_true, "y_pred": y_pred}
        
        return results
    
    def detect_change_points(self, predictions: np.array, count_required=20):
        """Detects the change point and returns an array of 1 until the change point and -1 after the change point """
        y_pred = []
        counter = 0
        for i in range(len(predictions)):
            element = predictions[i]
            if element == -1:
                y_pred.append(-1)
                counter += 1
                if counter >= count_required:
                    y_pred.extend([-1] * (len(predictions) - i - 1))
                    return np.array(y_pred)
            else:
                counter = 0
                y_pred.append(1)
        return np.array(y_pred)