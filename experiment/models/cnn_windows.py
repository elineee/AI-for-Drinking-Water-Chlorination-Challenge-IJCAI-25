
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from data_transformation import calculate_labels_alarm, remove_first_x_days
from utils import detect_change_point
from experiment_config import ExperimentConfig
from models.SVR import SVRModel
from models.model import AnomalyModel


class CNN(nn.Module):
    def __init__(self, input_size, sequence_length):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # Reduce sequence length by half
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # Reduce sequence length by half
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)  # Reduce sequence length by half
        
        self.fc1 = nn.Linear(128*6, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1) # binary classification output
        
    
    def forward(self, x):
        # x shape: (batch, length, channels) = (batch, 48, 2)
        # Conv1d expects (batch, channels, length)
        x = x.transpose(1, 2)  # -> (batch, 2, 48)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)  # -> (batch, 64, 26)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)  # -> (batch, 128, 13)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)  # -> (batch, 128, 6)


        x = x.flatten(start_dim=1) # Flatten the output for the fully connected layers -> (batch, 128*6)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu3(x)
        x = self.fc3(x)
        
        return x


class CNNWindowsModel(AnomalyModel):
    
    def compute_weight(self, labels):
        """ Computes the weight for the positive class (anomalies) based on the imbalance of the dataset. 
        
        Parameters:
        - labels: a tensor containing the labels for the training set, where 1 corresponds to an anomaly and 0 to a normal point
        
        Returns :
        - a tensor containing the weight for the positive class, which can be used in the loss function to give more importance to the anomalies during training
        
        """
        n_normal = 0 
        n_anomalous = 0
        for label in labels:
            if label == 0:
                n_normal += 1
            else:
                n_anomalous += 1
        
        print(f"Number of normal samples: {n_normal}, Number of anomalous samples: {n_anomalous}")
        
        weight = n_normal / n_anomalous
        
        weights = torch.tensor([weight], dtype=torch.float32)

        return weights
    
    
    def run_model(self, train_dataloader, val_dataloader, test_dataloader, weights, epochs=10):
        """ Trains the CNN model and evaluates it on the test set.
        
        Parameters:
        - train_dataloader: DataLoader for the training set
        - val_dataloader: DataLoader for the validation set
        - test_dataloader: DataLoader for the test set
        
        Returns:
        - a list containing the predicted labels for each time step in the test set, where -1 corresponds to an anomaly and 1 to a normal point
        """
        model = CNN(input_size=2, sequence_length=self.config.window_size + 3)

        criterion = nn.BCEWithLogitsLoss(pos_weight=weights) # loss for binary classification
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        

        train_loss = []
        val_loss = []
        for epoch in range(epochs):
            n_corrects_train = 0
            n_corrects_val = 0
            n_total_train = 0
            n_total_val = 0
            losses = []
            model.train()
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
            
            model.eval()
            with torch.no_grad():
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
        f1_scores = []
        recall_scores = []
        
        final_preds = []
        with torch.no_grad():
            for _, data in enumerate(test_dataloader):
                windows, labels = data # windows shape (batch, 48, 2), labels shape (batch, 48)
 
                outputs = model(windows) # outputs shape (batch, 1, 48)
                outputs = outputs.squeeze(1)  # Remove the channel dimension -> (batch, 48)
                
                probs = torch.sigmoid(outputs) # Convert logits to probabilities

                preds = (probs > 0.5).float() # Threshold at 0.5 to get binary predictions 
                

                labels = labels.flatten()
                n_total += len(labels)
                preds = preds.flatten()
                final_preds.append(preds)
                
                n_corrects += (preds == labels).sum().item()
                f1 = f1_score(labels, preds, average='binary', zero_division=1)
                f1_scores.append(f1)
                recall = recall_score(labels, preds, average='binary', zero_division=1)
                recall_scores.append(recall)
            
            print(f"Final Accuracy: {n_corrects/n_total:.4f}")
            print(f"Final F1 Score: {np.mean(f1_scores):.4f}")
            print(f"Final Recall Score: {np.mean(recall_scores):.4f}")
            
            y_pred = []
            for element in final_preds: 
                if element == 1: 
                    y_pred.append(-1)
                else:
                    y_pred.append(1)

            return y_pred
    
    
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
            data_train = []
            data_svr_train = []
            y_train = []

            # last dataset for testing 
            # train data 
            for df in contaminated_dfs[:-1]:
                df_clean, features, labels, y_svr = self._prepare_data(svr_model, df, clean_dfs, node)
                new_contaminated_dfs.append(df_clean)
                data_train.extend(features)
                data_svr_train.extend(y_svr)
                y_train.extend(labels)
            
            # test data (last dataset)
            df_clean_test, features_test, labels_test, y_svr_test = self._prepare_data(svr_model, contaminated_dfs[-1], clean_dfs, node)
            
            y_true = calculate_labels_alarm(df_clean_test, self.config.contaminants[0].value, self.config.window_size+3)

            # turn data and y into tensors
            data_train = np.array(data_train) # shape of (4706, 48)
            data_train = torch.tensor(data_train, dtype=torch.float32) # shape of (4706, 48)
            data_test = np.array(features_test) # shape of (2401, 48)
            data_test = torch.tensor(data_test, dtype=torch.float32) # shape of (2401, 48)
            
            data_svr_train = np.array(data_svr_train) # shape of (4706, 48)
            data_svr_train = torch.tensor(data_svr_train, dtype=torch.float32) # shape of (4706, 48)
            data_svr_test = np.array(y_svr_test) # shape of (2401, 48)
            data_svr_test = torch.tensor(data_svr_test, dtype=torch.float32) # shape of (2401, 48)
            
            # turn into multivarite 
            data_train = torch.stack((data_train, data_svr_train), dim=2) # shape of (4706, 48, 2)
            y_train = np.array(y_train) # shape of (4706, 1)
            y_train = torch.tensor(y_train, dtype=torch.float32) # shape of (4706, 1)
            data_test = torch.stack((data_test, data_svr_test), dim=2) # shape of (2401, 48, 2)
            y_test = torch.tensor(labels_test, dtype=torch.float32) # shape of (2401, 1)
            
            # split into train, val and test sets
            X_train, X_val, y_train, y_val = train_test_split(data_train, y_train, test_size=0.15, random_state=42)


            weights = self.compute_weight(y_train)
            
            # create DataLoaders
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            test_dataset = TensorDataset(data_test, y_test)
            train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True) # one batch = (32, 48)
            val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            
            y_pred = self.run_model(train_dataloader, val_dataloader, test_dataloader, weights, epochs=20)
            
            y_pred = detect_change_point(y_pred, count_required=10)
            
            print(len(y_true))
            print(len(y_pred))
            
            results[node] = {"y_pred": y_pred, "y_true": y_true}
        
        return results


    def create_labeled_features(self, df: pd.DataFrame, feature_column: str, label_column: str, window_left, window_right):
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
        for i in range(window_left, len(feature)-window_right):
            row = feature[i-window_left:i+window_right]
            label_value = label[i] # label of the current time step
            
            features.append(row)
            labels.append(label_value)
        
        return np.array(features), np.array(labels)
    

    def create_direct_features(self, time_series, window_left, window_right):
        """ Creates features for anomaly detection using a sliding window approach.
        
        Parameters:
        - time_series: a numpy array containing the time series data
        - window_left: the size of the window to the left of each time step
        - window_right: the size of the window to the right of each time step
        
        Returns:
        - a numpy array containing the features for each time step, where each feature is the values of the time series in the sliding window
        """
        
        feature = time_series
        
        features = []
        for i in range(window_left, len(feature)-window_right):
            row = feature[i-window_left:i+window_right]
            
            features.append(row)
        
        return np.array(features)
    

    def get_labels(self, label, window=3, anomaly=True):
        """ Converts a label array into a new label array where each change point or anomaly is labeled as 1 and normal points are labeled as 0.
        If change point, a window is created around each change point to account for potential delays in detection (the window size is two times longer after than before the change point to account for potential delays in detection).
        The change points are defined as the points where the original label changes from 0 to >0 
        Parameters:
        - label: a numpy array containing the original labels 
        - window: the size of the window around each change point (default is 3, which means that 3 points before and 6 points after the change point will be labeled as 1)
        - anomaly : wheter the labels are all the anomalies or change points 
        
        Returns:
        - a numpy array containing the new labels, where each change point is labeled as 1 and normal points are labeled as 0
        """
        
        
        # change point/anomaly = 1 
        # normal point = 0
        
        y = np.zeros(len(label), dtype=int) 
        
        for i in range(len(label)):
            if anomaly : 
                if label[i] > 0.01 :
                    y[i] = 1
                else:
                    y[i] = 0
            else :
                if i == 0 and label[i] > 0:
                    start = 0
                    end = min(len(label), i + 2* window + 1)  
                    y[start:end] = 1

                if i > 0 and label[i-1] == 0 and label[i] > 0:

                    start = max(0, i - window)  
                    end = min(len(label), i + 2 * window + 1)  
                    y[start:end] = 1
        
        return y.tolist()
    

    def _prepare_data(self, svr_model, df, clean_dfs, node):
        """ Prepares the data for training and testing the CNN model.
        
        Parameters:
        - svr_model: the SVR model to use for generating features
        - df: the contaminated dataframe to use for training and testing
        - clean_dfs: a list of clean dataframes to use for training the SVR model
        - node: the node id to use for generating features with the SVR model
        
        Returns:
        - df_clean: the cleaned dataframe after removing the first 3 days
        - features: the features for training/testing the CNN model, where each feature is a sliding window of the time series data
        - labels: the labels for training/testing the CNN model, where each label is a sliding window of the original labels
        - y_svr: the features generated by the SVR model, where each feature is a sliding window of the predicted values of the SVR model
        
        """
        _, _, _, y_svr = svr_model.predict(node, clean_dfs, [df])
        y_svr = y_svr.squeeze()  # Convert (N, 1) to (N,)
        
        df_clean = remove_first_x_days(df, 3) # shape of (2401,) x2 = 4802
        
        # add padding because different shape
        if len(y_svr) < len(df_clean):
            pad_size = len(df_clean) - len(y_svr)
            y_svr = np.concatenate([np.zeros(pad_size), y_svr])
        
        features, labels = self.create_labeled_features(df_clean, self.config.disinfectant.value, self.config.contaminants[0].value, window_left=self.config.window_size, window_right=3)
        print("features shape", features.shape)  # Should be (N, window_size + window_right)
        print("labels shape", labels.shape)      # Should be (N,)
        y_svr = self.create_direct_features(y_svr, window_left=self.config.window_size, window_right=3)
        print("y_svr shape", y_svr.shape)      # Should be (N, window_size + window_right)
        
        return df_clean, features, labels, y_svr