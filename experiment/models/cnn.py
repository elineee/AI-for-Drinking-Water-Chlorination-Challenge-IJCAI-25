
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader

from data_transformation import remove_first_x_days
from experiment_config import ContaminationType, ExperimentConfig
from models.SVR import SVRModel
from models.model import AnomalyModel


class CNN(nn.Module):
    def __init__(self, input_size, num_classes, sequence_length=48):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU()
        
        self.conv_out = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1)
    def forward(self, x):

        x = x.transpose(1, 2)  # -> (batch, 2, 48)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)  

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv_out(x)

        
        return x


class CNNModel(AnomalyModel):
    
    def train_model(self, dataloader, epochs=10):
        model = CNN(input_size=2, num_classes=2, sequence_length=48)
        criterion = nn.BCEWithLogitsLoss() # loss for binary classification
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        n_corrects = 0
        n_total = 0
        losses = []
        model.train()
        for epoch in range(epochs):
            for _, data in enumerate(dataloader):
                windows, labels = data # windows shape (batch, 48, 2), labels shape (batch, 48)
 
                outputs = model(windows) # outputs shape (batch, 1, 48)
                outputs = outputs.squeeze(1)  # Remove the channel dimension -> (batch, 48)
                
                probs = torch.sigmoid(outputs) # Convert logits to probabilities

                preds = (probs > 0.5).float() # Threshold at 0.5 to get binary predictions 
                
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
                
                n_total += labels.numel()
                n_corrects += (preds == labels).sum().item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {n_corrects/n_total:.4f}")
                
    
    def eval_model(self):
        pass
    
    def get_results(self):
        results = {}
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()
        
        for node, contaminated_dfs in all_contaminated_dfs.items():
            clean_dfs = all_clean_dfs[node]
            
            print(f"Calculating results for node {node}")
            
            config_svr = ExperimentConfig(
                    config_name="SVR",
                    contaminated_files=self.config.contaminated_files,
                    example_files=self.config.example_files,
                    nodes=[node],
                    window_size=48, # 48 correspond à 48*30 min donc 1 jour
                    model_name="SVR",
                    model_params={"gamma": "scale", "epsilon": 0.01, "kernel": "rbf", "C": 10},
                )
            
            svr_model = SVRModel(config_svr)
            
            new_contaminated_dfs = []
            data = []
            data_svr = []
            y = []

            for df in contaminated_dfs:
                print("df shape", df.shape) 
                _, _, _, y_svr = svr_model.predict(node, clean_dfs, [df])
                y_svr = y_svr.squeeze()  # Convert (N, 1) to (N,)
                # y_svr = np.random.rand(len(df)) # TODO : replace with real predictions of SVR model (temp random values for now)
                print("y svr shape", y_svr.shape)
                
                df_clean = remove_first_x_days(df, 3)
                print("df clean shape", df_clean.shape) # shape of (2401,) x2 = 4802
                new_contaminated_dfs.append(df_clean)
                
                # add padding because different shape
                if len(y_svr) < len(df_clean):
                    pad_size = len(df_clean) - len(y_svr)
                    y_svr = np.concatenate([np.zeros(pad_size), y_svr])
                
                features, labels = self.create_labeled_features(df_clean, self.config.disinfectant.value, self.config.contaminants[0].value, window_size=self.config.window_size)
                y_svr = self.create_direct_features(y_svr, window_size=self.config.window_size)
                
                print(f"features shape: {np.array(features).shape}, y_svr shape: {np.array(y_svr).shape}")
                
                data.extend(features)
                data_svr.extend(y_svr)
                y.extend(labels)
            
            # TODO : get multivariate time series. If two features, then shape of data should be (4706, 48, 2)
            # turn data and y into tensors
            data = torch.tensor(data, dtype=torch.float32) # shape of (4706, 48)
            data_svr = torch.tensor(data_svr, dtype=torch.float32) # shape of (4706, 48)
            # turn into multivarite 
            data = torch.stack((data, data_svr), dim=2) # shape of (4706, 48, 2)
            print("data shape", data.shape)
            y = torch.tensor(y, dtype=torch.float32) # shape of (4706, 48)
            
            # TODO : split into train and test set 
            # TODO : standardize data 
            
            # create DataLoader 
            dataset = TensorDataset(data, y)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True) # one batch = (32, 48)
            
            self.train_model(dataloader, epochs=10)
            

        
    
    def create_labeled_features(self, df: pd.DataFrame, feature_column: str, label_column: str, window_size: int = 10):
        """
        Creates labeled features for anomaly detection using a sliding window approach.
        
        Parameters:
        - df: a pandas DataFrame containing the data
        - feature_column: the name of the column to use as feature
        - label_column: the name of the column to use as label
        - window_size: the size of the sliding window
        Returns:
        - a numpy array containing the extended features for each time step
        - a numpy array containing the labels for each time step
        """
        for column in df.columns:
            if feature_column in column:
                feature_column = column
                break
        
        for column in df.columns:
            if label_column in column:
                label_column = column
                break
        
        feature = df[feature_column].values
        label = df[label_column].values
        label = self.get_labels(label)
        
        features = []
        labels = []
        for i in range(window_size, len(feature)):
            row = feature[i-window_size:i]
            label_value = label[i-window_size:i]
            
            features.append(row)
            labels.append(label_value)
        
        return np.array(features), np.array(labels)
    
    def create_direct_features(self, time_series, window_size: int = 10):
        """ Creates features for anomaly detection using a sliding window approach.
        
        Parameters:
        - time_series: a numpy array containing the time series data
        - window_size: the size of the sliding window
        
        Returns:
        - a numpy array containing the features for each time step, where each feature is the values of the time series in the sliding window
        """
        
        feature = time_series
        
        features = []
        for i in range(window_size, len(feature)):
            row = feature[i-window_size:i]
            
            features.append(row)
        
        return np.array(features)
    
    def get_labels(self, label, window=3):
        # change point = 1 
        # normal point = 0
        
        y = np.ones(len(label), dtype=int) 
        
        for i in range(len(label)):
            
            if i == 0 and label[i] > 0:
                start = 0
                end = min(len(label), i + window + 1)  
                y[start:end] = 1

            if i > 0 and label[i-1] == 0 and label[i] > 0:

                start = max(0, i - window)  
                end = min(len(label), i + window + 1)  
                y[start:end] = 1
        
        return y.tolist()