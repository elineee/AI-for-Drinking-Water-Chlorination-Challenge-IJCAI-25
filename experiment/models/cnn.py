
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
    
    def compute_weight(self, labels):
        # Compute the number of normal and anomalous samples
        n_normal = 0 
        n_anomalous = 0
        for window in labels:
            for label in window:
                if int(label) == 0:
                    n_normal += 1
                else:
                    n_anomalous += 1
        
        print(f"Number of normal samples: {n_normal}, Number of anomalous samples: {n_anomalous}")
        
        weight = n_normal / n_anomalous
        
        weights = torch.tensor([weight], dtype=torch.float32)
        print(f"Weight for anomalous samples: {weight}")

        return weights
    

    def run_model(self, train_dataloader, val_dataloader, test_dataloader, weights, epochs=10):
        model = CNN(input_size=2, num_classes=2, sequence_length=48)

        criterion = nn.BCEWithLogitsLoss(pos_weight=weights) # loss for binary classification
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        n_corrects_train = 0
        n_corrects_val = 0
        n_total_train = 0
        n_total_val = 0
        losses = []
        train_loss = []
        val_loss = []
        model.train()
        for epoch in range(epochs):
            for _, data in enumerate(train_dataloader):
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
                
                n_total_train += labels.numel()
                n_corrects_train += (preds == labels).sum().item()
            train_loss.append(np.mean(losses))
            losses = []
            
            for _, data in enumerate(val_dataloader):
                windows, labels = data # windows shape (batch, 48, 2), labels shape (batch, 48)
 
                outputs = model(windows) # outputs shape (batch, 1, 48)
                outputs = outputs.squeeze(1)  # Remove the channel dimension -> (batch, 48)
                
                probs = torch.sigmoid(outputs) # Convert logits to probabilities

                preds = (probs > 0.5).float() # Threshold at 0.5 to get binary predictions 
                
                loss = criterion(outputs, labels)
                losses.append(loss.item())
                n_total_val += labels.numel()
                n_corrects_val += (preds == labels).sum().item()
            val_loss.append(np.mean(losses))
            losses = []
                
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Training Accuracy: {n_corrects_train/n_total_train:.4f}, Validation Accuracy: {n_corrects_val/n_total_val:.4f}")
            
        plt.figure()
        plt.plot(train_loss, label='train')
        plt.plot(val_loss, label='validation')
        plt.title("Évolution de la loss en fonction des epochs")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.show()
        
        model.eval()
        n_corrects = 0
        n_total = 0
        with torch.no_grad():
            for _, data in enumerate(test_dataloader):
                windows, labels = data # windows shape (batch, 48, 2), labels shape (batch, 48)
 
                outputs = model(windows) # outputs shape (batch, 1, 48)
                outputs = outputs.squeeze(1)  # Remove the channel dimension -> (batch, 48)
                
                probs = torch.sigmoid(outputs) # Convert logits to probabilities

                preds = (probs > 0.5).float() # Threshold at 0.5 to get binary predictions 
                
                n_total += labels.numel()
                n_corrects += (preds == labels).sum().item()
                print("preds:", preds)
                print("labels:", labels)
            print(f"Final Accuracy: {n_corrects/n_total:.4f}")
    
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
            

  
            y = torch.tensor(y, dtype=torch.float32) # shape of (4706, 48)
            
            X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.15, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1125, random_state=42) # 0.15 * 0.75 = 0.1125

            weights = self.compute_weight(y_train)
            weights_2 = self.compute_weight(y_test)
            
            # TODO : split into train and test set 
            # TODO : standardize data 
            
            # create DataLoader 
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            test_dataset = TensorDataset(X_test, y_test)
            train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True) # one batch = (32, 48)
            val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            self.run_model(train_dataloader, val_dataloader, test_dataloader, weights, epochs=15)
            

        
    
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
        
        y = np.zeros(len(label), dtype=int) 
        
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