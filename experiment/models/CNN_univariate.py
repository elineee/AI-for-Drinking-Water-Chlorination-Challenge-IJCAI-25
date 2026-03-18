import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from data_transformation import remove_first_x_days, calculate_labels_alarm
from utils import detect_change_point
from models.CNN import CNNModel

class TimeSeriesAugmentation:
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data

class AugmentedTensorDataset(Dataset):
    def __init__(self, data, labels, augment=None):
        self.data = data
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.augment is not None:
            x = self.augment(x)
            
        return x, y

def gaussian_noise(data, mean=0.0, prob=0.5):
    if np.random.rand() > prob:
        return data
    std_values = [0.01, 0.03, 0.05] 
    std = np.random.choice(std_values)  
    noise = np.random.normal(mean, std, data.shape)
    data = data + noise 
    data = data.float()
    return data 

def blank_value(data, percentage=0.025, prob=0.5):
    if np.random.rand() > prob:
        return data
    num_blank = int(percentage * data.numel())
    indices = np.random.choice(data.numel(), num_blank, replace=False)
    data[indices] = 0.0
    data = data.float()
    return data

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
            train_dataset = AugmentedTensorDataset(X_train, y_train, augment=TimeSeriesAugmentation([gaussian_noise, blank_value]))
            # train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            test_dataset = TensorDataset(data_test, y_test)
            train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True) # one batch = (32, 48)
            val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            

            weights = self._compute_weight(y_train)
            y_pred = self.run_model(train_dataloader, val_dataloader, test_dataloader, weights, epochs=15)
            y_pred = detect_change_point(y_pred, count_required=15)
            results[node] = {"y_pred": y_pred, "y_true": y_true}
        
        return results

    
    def _prepare_data(self, df):
        """ Prepares the data for training and testing the CNN univariate model.
        
        Parameters:
        - df: the contaminated dataframe to use for training and testing

        Returns:
        - df_clean: the cleaned dataframe after removing the first 3 days
        - features: the features for training/testing the CNN model, where each feature is a sliding window of the time series data (shape (number of windows, window_size))
        - labels: the labels for training/testing the CNN model, where each label is a sliding window of the original labels (shape (number of windows, window_size))
        
        """
        df_clean = remove_first_x_days(df, 3) 
        features, labels = self.create_labeled_features(df_clean, self.config.disinfectant.value, self.config.contaminants[0].value, window_size=self.config.window_size)
        
        return df_clean, features, labels
        