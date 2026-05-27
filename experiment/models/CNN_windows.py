
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from data_transformation import calculate_labels_alarm, remove_first_x_days, get_labels
from utils import add_noisy_dfs, detect_change_point
from experiment_config import ContaminationType, ExperimentConfig
from models.SVR import SVRModel
from models.model import AnomalyModel


class CNN(nn.Module):
    def __init__(self, input_size, sequence_length):
        super(CNN, self).__init__()
        self.sequence_length = sequence_length
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # Reduce sequence length by half
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # Reduce sequence length by half
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)  # Reduce sequence length by half
        
        # Calculate size after pooling operations
        # After each MaxPool1d with kernel_size=2: length = floor(length / 2)
        length_after_pool = sequence_length
        length_after_pool1 = length_after_pool // 2  
        length_after_pool2 = length_after_pool1 // 2  
        length_after_pool3 = length_after_pool2 // 2  

        flattened_size = 128 * length_after_pool3
        
        self.fc1 = nn.Linear(flattened_size, 128) 
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu5 = nn.ReLU()
        self.fc3 = nn.Linear(64, 1) # binary classification output
        
    
    def forward(self, x):
        # x shape: (batch, length, channels)
        # Conv1d expects (batch, channels, length)
        x = x.transpose(1, 2)  # -> (batch, channels, length)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)  

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)  
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x) 

        x = x.flatten(start_dim=1) # Flatten the output for the fully connected layers -> (batch, 128*6)

        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.relu5(x)
        x = self.fc3(x)
        
        return x


class CNNWindowsModel(AnomalyModel):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using GPU:", torch.cuda.get_device_name(0))
        else:
            print("GPU not available, using CPU.")


    def _compute_weight(self, labels):
        """ 
        Computes the weight for the positive class (anomalies) based on the imbalance of the dataset. 
        Weights can be used in the loss function to give more importance to the anomalies during training.

        Parameters:
        - labels: a tensor containing the labels for the training set, where 1 corresponds to an anomaly and 0 to a normal point
        
        Returns :
        - a tensor containing the weight for the positive class
        
        """
        labels_np = np.array(labels)
        n_normal = (labels_np == 0).sum()
        n_anomalous = (labels_np == 1).sum()
        
        print(f"Number of normal samples: {n_normal}, Number of anomalous samples: {n_anomalous}")
    
        weights = torch.tensor([n_normal / n_anomalous], dtype=torch.float32).to(self.device)

        return weights
    

    def run_model(self, train_dataloader, val_dataloader, test_dataloader, weights, epochs):
        """ Trains the CNN model and evaluates it on the test set.
        
        Parameters:
        - train_dataloader: DataLoader for the training set
        - val_dataloader: DataLoader for the validation set
        - test_dataloader: DataLoader for the test set
        - weights: tensor containing the weight for the positive class
        - epochs: number of training epochs
        
        Returns:
        - a list containing the predicted labels for each time step in the test set, where -1 corresponds to an anomaly and 1 to a normal point
        """
        model = CNN(input_size=2, sequence_length=self.config.window_size + 3).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=weights) # loss for binary classification
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        node = self.config.nodes[0]
        if os.path.exists(f"cnn_{node}.pth"):
            model.load_state_dict(torch.load(f"cnn_{node}.pth", map_location=self.device, weights_only=True))
        else:
            train_loss = []
            val_loss = []
            best_val_f1 = 0
            for epoch in range(epochs):
                
                losses = []
                train_preds_all = []
                train_labels_all = []
                val_preds_all = []
                val_labels_all = []
                model.train()
                
                for _, data in enumerate(train_dataloader):
                    windows, labels = data  # windows shape of (batch, window_size+3, 2), labels shape (batch, 1)
                    windows = windows.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(windows) # outputs shape of (batch, 1)
                    outputs = outputs.squeeze(1) # (batch,)
                    
                    probs = torch.sigmoid(outputs) # Convert logits to probabilities

                    preds = (probs > 0.5).float() # Threshold at 0.5 to get binary predictions 
                    
                    optimizer.zero_grad()
                    loss = criterion(outputs, labels)
                    losses.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    
                    train_preds_all.append(preds.flatten().cpu().numpy())
                    train_labels_all.append(labels.flatten().cpu().numpy())
                train_loss.append(np.mean(losses))
                train_preds_all = np.concatenate(train_preds_all)
                train_labels_all = np.concatenate(train_labels_all)
                train_f1 = f1_score(train_labels_all, train_preds_all, average="binary", zero_division=1)
                losses = []
                
                model.eval()
                with torch.no_grad():
                    for _, data in enumerate(val_dataloader):
                        windows, labels = data # windows shape (batch, window_size+3, 2), labels shape : (batch, 1)
                        windows = windows.to(self.device)
                        labels = labels.to(self.device)

                        outputs = model(windows)  # outputs shape (batch, 1)
                        outputs = outputs.squeeze(1)   # (batch,)
                        
                        probs = torch.sigmoid(outputs) # Convert logits to probabilities

                        preds = (probs > 0.5).float() # Threshold at 0.5 to get binary predictions 
                        
                        loss = criterion(outputs, labels)
                        losses.append(loss.item())
                        val_preds_all.append(preds.flatten().cpu().numpy())
                        val_labels_all.append(labels.flatten().cpu().numpy())
                        
                val_loss.append(np.mean(losses))
                val_preds_all = np.concatenate(val_preds_all)
                val_labels_all = np.concatenate(val_labels_all)
                val_f1 = f1_score(val_labels_all, val_preds_all, average="binary", zero_division=1)
                losses = []
                    
                
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Training F1: {train_f1:.4f}, Validation F1: {val_f1:.4f}")
                
                # Save model with best validation F1 score
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    torch.save(model.state_dict(), f"cnn_{node}.pth")
                    print(f"  -> Best model saved with validation F1: {best_val_f1:.4f}")
                
                
            plt.figure()
            plt.plot(train_loss, label="train")
            plt.plot(val_loss, label="validation")
            plt.title("Loss evolution over epochs")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.show()
            
            model.load_state_dict(torch.load(f"cnn_{node}.pth", map_location=self.device, weights_only=True))
            
        
        model.eval()
        final_preds = []
        final_labels = []
        
        with torch.no_grad():
            for _, data in enumerate(test_dataloader):
                windows, labels = data # windows shape (batch, window_size+3, 2), labels shape : (batch, 1)
                windows = windows.to(self.device)
                labels = labels.to(self.device)

                outputs = model(windows) # outputs shape (batch, 1)
                outputs = outputs.squeeze(1)   # (batch,)
                
                probs = torch.sigmoid(outputs) # Convert logits to probabilities
                preds = (probs > 0.5).float() # Threshold at 0.5 to get binary predictions 
                
                labels = labels.flatten().detach().cpu().numpy()
                preds = preds.flatten().detach().cpu().numpy()

                final_preds.append(preds)
                final_labels.append(labels)
                
            all_preds = np.concatenate(final_preds)
            all_labels = np.concatenate(final_labels)

            f1 = f1_score(all_labels, all_preds, average="binary", zero_division=1)
            recall = recall_score(all_labels, all_preds, average="binary", zero_division=1)

            print(f"Final F1 score: {f1:.4f}")
            print(f"Final Recall: {recall:.4f}")
        
            y_pred = []
            for batch_preds in final_preds: 
                for element in batch_preds: 
                    if element == 1: 
                        y_pred.append(-1)
                    else:
                        y_pred.append(1)

            return y_pred
    

    def _call_second_model(self, node):
        """
        Calls the second model (a SVR Model) used to generate additional features for the CNN.
   
        Parameters:
        - node: the node id 
        
        Returns:
        - svr_model: an instantiated svr model 
        """
        
        if self.config.contaminants[0] == ContaminationType.ARSENIC:
            config_svr = ExperimentConfig(
                config_name="SVR_arsenic",
                contaminated_files=self.config.contaminated_files,
                example_files=self.config.example_files,
                nodes=[node],
                window_size=48, # 48*30 min = one day
                model_name="SVR",
                model_params={"gamma": "scale", "epsilon": 0.01, "kernel": "rbf", "C": 10},
                contaminants=[ContaminationType.ARSENIC]
            )
        else:
            config_svr = ExperimentConfig(
                config_name="SVR_pathogen",
                contaminated_files=self.config.contaminated_files,
                example_files=self.config.example_files,
                nodes=[node],
                window_size=288, 
                model_name="SVR",
                model_params={"gamma": "scale", "epsilon": 0.05, "kernel": "rbf", "C": 10},
                contaminants=[ContaminationType.PATHOGEN]
            )
        
        svr_model = SVRModel(config_svr)
        return svr_model 


    def get_results(self):
        results = {}
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()
        
        for node, contaminated_dfs in all_contaminated_dfs.items():
            clean_dfs = all_clean_dfs[node]
            
            clean_dfs = add_noisy_dfs(clean_dfs)
            test_contaminated_df = contaminated_dfs[-1]
            contaminated_dfs = add_noisy_dfs(contaminated_dfs[:-1]) + [test_contaminated_df]
            
            print(f"Calculating results for node {node}")
            
            svr_model = self._call_second_model(node)

            data_train = []
            data_svr_train = []
            y_train = []

            # last dataset for testing 
            # train data 
            for df in contaminated_dfs[:-1]:
                _, features, labels, y_svr = self._prepare_data(svr_model, df, clean_dfs, node)
                data_train.extend(features)
                data_svr_train.extend(y_svr)
                y_train.extend(labels)
            
            # test data (last dataset)
            prepared_df_test, features_test, labels_test, y_svr_test = self._prepare_data(svr_model, contaminated_dfs[-1], clean_dfs, node)
            
            y_true = calculate_labels_alarm(prepared_df_test, self.config.contaminants[0].value, self.config.window_size+3)

            # turn data and y into tensors
            data_train = np.array(data_train) # shape of (n_windows, window_size+3)
            data_train = torch.tensor(data_train, dtype=torch.float32) 
            data_test = np.array(features_test) # shape of (n_windows, window_size+3)
            data_test = torch.tensor(data_test, dtype=torch.float32) 
            
            data_svr_train = np.array(data_svr_train) # shape of (n_windows, window_size+3)
            data_svr_train = torch.tensor(data_svr_train, dtype=torch.float32)
            data_svr_test = np.array(y_svr_test)
            data_svr_test = torch.tensor(data_svr_test, dtype=torch.float32) 
            
            # turn into multivarite 
            data_train = torch.stack((data_train, data_svr_train), dim=2) # shape of (n_windows, window_size+3, 2)
            y_train = np.array(y_train) 
            y_train = torch.tensor(y_train, dtype=torch.float32) # shape of (n_windows, 1)
            data_test = torch.stack((data_test, data_svr_test), dim=2) # shape of (n_windows, window_size+3, 2)
            y_test = torch.tensor(labels_test, dtype=torch.float32) # shape of (n_windows, 1)
            
            # split into train, val and test sets
            X_train, X_val, y_train, y_val = train_test_split(data_train, y_train, test_size=0.15, random_state=42)


            weights = self._compute_weight(y_train)
            
            # create DataLoaders
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            test_dataset = TensorDataset(data_test, y_test)
            train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True) 
            val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            
            y_pred = self.run_model(train_dataloader, val_dataloader, test_dataloader, weights, epochs=50)
            
            y_pred = detect_change_point(y_pred, count_required=30)
            
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
        - window_left: the size of the window to the left of the current time step
        - window_right: the size of the window to the right of the current time step
        
        Returns:
        - a numpy array containing the features for each time step (shape (number of windows, window_left + window_right))
        - a numpy array containing the label for the center time step of each window (shape (number of windows,))
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
        label = get_labels(label)
        
        features = []
        labels = []
        for i in range(window_left, len(feature)-window_right):
            row = feature[i-window_left:i+window_right]
            label_value = label[i] # label of the current time step
            
            features.append(row)
            labels.append(label_value)
        
        return np.array(features), np.array(labels)
    

    def create_direct_features(self, time_series, window_left, window_right):
        """ 
        Creates features for anomaly detection using a sliding window approach.
        
        Parameters:
        - time_series: a numpy array containing the time series data
        - window_left: the size of the window to the left of each time step
        - window_right: the size of the window to the right of each time step
        
        Returns:
        - a numpy array containing the features for each time step, where each feature is the values of the time series in the sliding window (shape (number of windows, window_left + window_right))

        """
                
        features = []
        for i in range(window_left, len(time_series)-window_right):
            row = time_series[i-window_left:i+window_right]
            
            features.append(row)
        
        return np.array(features)


    def _prepare_data(self, svr_model, df, clean_dfs, node):
        """ 
        Prepares the data for training and testing the CNN model.
        
        Parameters:
        - svr_model: the SVR model to use for generating features
        - df: the contaminated dataframe to use for training and testing
        - clean_dfs: a list of clean dataframes to use for training the SVR model
        - node: the node id to use for generating features with the SVR model

        Returns:
        - prepared_df: the contaminated dataframe after removing the first 3 days
        - features: the features for training/testing the CNN model, where each feature is a sliding window of the time series data (shape (number of windows, window_left + window_right))
        - labels: the labels for training/testing the CNN model, where each label corresponds to the center time step of the window (shape (number of windows,))
        - y_svr: the features generated by the SVR model, where each feature is a sliding window of the predicted values of the SVR model (shape (number of windows, window_left + window_right))

        """
        _, _, _, y_svr = svr_model.predict(node, clean_dfs, [df])
        y_svr = y_svr.squeeze()  # Convert (N, 1) to (N,)
        
        prepared_df = remove_first_x_days(df, 3) # shape of (2401,) x2 = 4802
        
        # add padding because different shape
        if len(y_svr) < len(prepared_df):
            pad_size = len(prepared_df) - len(y_svr)
            y_svr = np.concatenate([np.zeros(pad_size), y_svr])
        
        features, labels = self.create_labeled_features(prepared_df, self.config.disinfectant.value, self.config.contaminants[0].value, window_left=self.config.window_size, window_right=3)
        y_svr = self.create_direct_features(y_svr, window_left=self.config.window_size, window_right=3)
        
        return prepared_df, features, labels, y_svr
