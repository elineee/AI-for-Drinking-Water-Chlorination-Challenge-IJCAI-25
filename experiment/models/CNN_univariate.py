import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from data_transformation import remove_first_x_days, calculate_labels_alarm
from utils import detect_change_point
from models.CNN import CNNModel

class CNNUnivariateModel(CNNModel):
    """ Class for CNN model. It takes into account the raw signal (univariate)"""

    def _get_input_size(self):
        return 1
    

    def get_results(self):
        results = {}
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()
        
        for node, contaminated_dfs in all_contaminated_dfs.items():
            clean_dfs = all_clean_dfs[node]
            
            print(f"Calculating results for node {node}")

            data_train = []
            y_train = []

            for df in contaminated_dfs[:-1]:
                df_clean, features, labels = self._prepare_data(df)
                data_train.extend(features)
                y_train.extend(labels)

            # test data (last dataset)
            df_clean_test, features_test, labels_test = self._prepare_data(contaminated_dfs[-1])
            y_true = calculate_labels_alarm(df_clean_test, self.config.contaminants[0].value, 0)

            # turn data and y into tensors
            data_train = np.array(data_train) # shape of (4706, 48)
            data_train = torch.tensor(data_train, dtype=torch.float32) 
            data_train = data_train.unsqueeze(2) # shape of (4706, 48, 1)
            
            data_test = np.array(features_test)  # shape of (2401, 48)
            data_test = torch.tensor(data_test, dtype=torch.float32) 
            data_test = data_test.unsqueeze(2) # shape of (4706, 48, 1)

            y_train = np.array(y_train) # shape of (4706, 48)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            y_test = torch.tensor(labels_test, dtype=torch.float32)

            # split into train, val and test sets
            X_train, X_val, y_train, y_val = train_test_split(data_train, y_train, test_size=0.15, random_state=42)

            # create DataLoaders
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            test_dataset = TensorDataset(data_test, y_test)
            train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True) # one batch = (32, 48)
            val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            

            weights = self._compute_weight(y_train)
            y_pred = self.run_model(train_dataloader, val_dataloader, test_dataloader, weights, epochs=20)
            y_pred = detect_change_point(y_pred, count_required=5)
            results[node] = {"y_pred": y_pred, "y_true": y_true}
        
        return results

    
    def _prepare_data(self, df):
        """ Prepares the data for training and testing the CNN model.
        
        Parameters:
        - df: the contaminated dataframe to use for training and testing

        Returns:
        - df_clean: the cleaned dataframe after removing the first 3 days
        - features: the features for training/testing the CNN model, where each feature is a sliding window of the time series data
        - labels: the labels for training/testing the CNN model, where each label is a sliding window of the original labels
        
        """
        df_clean = remove_first_x_days(df, 3) 
        features, labels = self.create_labeled_features(df_clean, self.config.disinfectant.value, self.config.contaminants[0].value, window_size=self.config.window_size)
        
        return df_clean, features, labels
        