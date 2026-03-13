from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from data_transformation import remove_first_x_days, calculate_labels_alarm, get_labels
from utils import detect_change_point
from experiment_config import ContaminationType, ExperimentConfig
from models.SVR import SVRModel 
from models.model import AnomalyModel


class CNN(nn.Module):
    def __init__(self, input_size):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.2)
        
        self.conv_out = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1)
        
    def forward(self, x):
        x = x.transpose(1, 2)  # -> (batch, 2, 48)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)  
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout4(x)

        x = self.conv_out(x)

        
        return x


class CNNModel(AnomalyModel):
    """ Class for CNN multivariate model. It takes into account the raw signal and the signal given by another model (by default, model)."""

    def _get_input_size(self):
        """
        Returns the number of input channels for the CNN model.
        2 is for multivariate and 1 for univariate.
        """
        return 2 


    def _compute_weight(self, labels):
        """ 
        Computes the weight for the positive class (anomalies) based on the imbalance of the dataset.
        Weights can be used in the loss function to give more importance to the anomalies during training.
        
        Parameters:
        - labels: a tensor containing the labels for the training set, where 1 corresponds to an anomaly and 0 to a normal point
        
        Returns :
        - weights: a tensor containing the weight for the positive class
        
        """

        labels_np = np.array(labels).flatten()
        n_normal = (labels_np == 0).sum()
        n_anomalous = (labels_np == 1).sum()
        
        print(f"Number of normal samples: {n_normal}, Number of anomalous samples: {n_anomalous}")
        
        weights = torch.tensor([n_normal / n_anomalous], dtype=torch.float32)

        return weights
    

    def run_model(self, train_dataloader, val_dataloader, test_dataloader, weights, epochs=10):
        """ 
        Trains the CNN model and evaluates it on the test set.
        The model predicts a label for each point in each window. 
        For each time step, labels are given by majority vote: it is an anomaly (-1) if more than 50% of the windows covering it predict an anomaly.

        Parameters:
        - train_dataloader: DataLoader for the training set
        - val_dataloader: DataLoader for the validation set
        - test_dataloader: DataLoader for the test set
        - weights: tensor containing the weight for the positive class (anomalies)
        - epochs: number of epochs 
   
        Returns:
        - a list containing the predicted labels for each time step in the test set, where -1 corresponds to an anomaly and 1 to a normal point
        """

        model = CNN(input_size=self._get_input_size())

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
        plt.plot(train_loss, label="train")
        plt.plot(val_loss, label="validation")
        plt.title("Loss evolution over epochs")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.show()
        
        model.eval()
        n_corrects = 0
        n_total = 0
        f1_scores = []
        recall_scores = []
        
        results_per_time_step = [[0, 0] for _ in range(len(test_dataloader.dataset) + self.config.window_size)]
        with torch.no_grad():
            i = 0
            for _, data in enumerate(test_dataloader):
                windows, labels = data # windows shape (batch, 48, 2), labels shape (batch, 48)
 
                outputs = model(windows) # outputs shape (batch, 1, 48)
                outputs = outputs.squeeze(1)  # Remove the channel dimension -> (batch, 48)
                
                probs = torch.sigmoid(outputs) # Convert logits to probabilities

                preds = (probs > 0.5).float() # Threshold at 0.5 to get binary predictions 
                

                labels = labels.flatten()
                n_total += len(labels)
                preds = preds.flatten()
                
                j = 0
                for element in preds:
                    results_per_time_step[i+j][0] += int(element) # add the predicted label (0 or 1) to the first element of the list corresponding to the time step i+j
                    results_per_time_step[i+j][1] += 1
                    j += 1
                    
                i += 1
                
                n_corrects += (preds == labels).sum().item()
                f1 = f1_score(labels, preds, average="binary", zero_division=1)
                f1_scores.append(f1)
                recall = recall_score(labels, preds, average="binary", zero_division=1)
                recall_scores.append(recall)
            
            print(f"Final Accuracy: {n_corrects/n_total:.4f}")
            print(f"Final F1 Score: {np.mean(f1_scores):.4f}")
            print(f"Final Recall Score: {np.mean(recall_scores):.4f}")
            
            # get the mean predicted label for each time step across all windows, and label as anomaly (-1) if the mean is greater than 0.5 and normal (1) otherwise
            mean_results_per_time_step = []
            for element, count in results_per_time_step:
                new_element = element / count if count > 0 else 0
                if new_element > 0.5:
                    mean_results_per_time_step.append(-1)
                else:
                    mean_results_per_time_step.append(1)

            return mean_results_per_time_step
    

    def _call_second_model(self, node):
        """
        Calls the secondary model used to generate additional features for the CNN.
        By default, it is a SVR model. 
        
        Parameters:
        - node: the node id 
        
        Returns:
        - an instantiated model 
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
                window_size=288, # 288*5 min = one day
                model_name="SVR",
                model_params={"gamma": "scale", "epsilon": 0.01, "kernel": "rbf", "C": 10},
                contaminants=[ContaminationType.PATHOGEN]
            )
        
        svr_model = SVRModel(config_svr)
        return svr_model 


    def get_results(self):
        results = {}
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()
        
        for node, contaminated_dfs in all_contaminated_dfs.items():
            clean_dfs = all_clean_dfs[node]
            
            print(f"Calculating results for node {node}")
            
            other_model = self._call_second_model(node)
            
            data_train = []
            data_model_train = []
            y_train = []

            # last dataset for testing 
            # train data 
            for df in contaminated_dfs[:-1]:
                df_clean, features, labels, y_model = self._prepare_data(other_model, df, clean_dfs, node)
                data_train.extend(features)
                data_model_train.extend(y_model)
                y_train.extend(labels)
            
            # test data (last dataset)
            df_clean_test, features_test, labels_test, y_model_test = self._prepare_data(other_model, contaminated_dfs[-1], clean_dfs, node)
            y_true = calculate_labels_alarm(df_clean_test, self.config.contaminants[0].value, 0)

            # turn data and y into tensors
            data_train = np.array(data_train) # shape of (4706, 48)
            data_train = torch.tensor(data_train, dtype=torch.float32) # shape of (4706, 48)
            data_test = np.array(features_test) # shape of (2401, 48)
            data_test = torch.tensor(data_test, dtype=torch.float32) # shape of (2401, 48)
            
            data_model_train = np.array(data_model_train) # shape of (4706, 48)
            data_model_train = torch.tensor(data_model_train, dtype=torch.float32) # shape of (4706, 48)
            data_model_test = np.array(y_model_test) # shape of (2401, 48)
            data_model_test = torch.tensor(data_model_test, dtype=torch.float32) # shape of (2401, 48)
            
            # turn into multivariate 
            data_train = torch.stack((data_train, data_model_train), dim=2) # shape of (4706, 48, 2)
            y_train = np.array(y_train) # shape of (4706, 48)
            y_train = torch.tensor(y_train, dtype=torch.float32) # shape of (4706, 48)
            data_test = torch.stack((data_test, data_model_test), dim=2) # shape of (2401, 48, 2)
            y_test = torch.tensor(labels_test, dtype=torch.float32) # shape of (2401, 48)
            
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
        label = get_labels(label)
        
        features = []
        labels = []
        for i in range(window_size, len(feature)):
            row = feature[i-window_size:i]
            label_value = label[i-window_size:i]
            
            features.append(row)
            labels.append(label_value)
        
        return np.array(features), np.array(labels)


    def create_direct_features(self, time_series, window_size: int = 10):
        """ 
        Creates features for anomaly detection using a sliding window approach.
        
        Parameters:
        - time_series: a numpy array containing the time series data
        - window_size: the size of the sliding window
        
        Returns:
        - a numpy array containing the features for each time step, where each feature is the values of the time series in the sliding window
        """
        
        features = []
        for i in range(window_size, len(time_series)):
            row = time_series[i-window_size:i]
            
            features.append(row)
        
        return np.array(features)


    def _prepare_data(self, second_model, df, clean_dfs, node):
        """ Prepares the data for training and testing the CNN model.
        
        Parameters:
        - second_model: the other model to use for generating features
        - df: the contaminated dataframe to use for training and testing
        - clean_dfs: a list of clean dataframes to use for training the second model
        - node: the node id to use for generating features with the second model
        
        Returns:
        - df_clean: the cleaned dataframe after removing the first 3 days
        - features: the features for training/testing the CNN model, where each feature is a sliding window of the time series data
        - labels: the labels for training/testing the CNN model, where each label is a sliding window of the original labels
        - y_model: the features generated by the second model, where each feature is a sliding window of the predicted values of the second model
        """
        _, _, _, y_model = second_model.predict(node, clean_dfs, [df])
        y_model = y_model.squeeze()  # Convert (N, 1) to (N,)
        
        df_clean = remove_first_x_days(df, 3) # shape of (2401,) x2 = 4802
        
        # add padding because different shape
        if len(y_model) < len(df_clean):
            pad_size = len(df_clean) - len(y_model)
            y_model = np.concatenate([np.zeros(pad_size), y_model])
        
        features, labels = self.create_labeled_features(df_clean, self.config.disinfectant.value, self.config.contaminants[0].value, window_size=self.config.window_size)
        y_model = self.create_direct_features(y_model, window_size=self.config.window_size)
        
        return df_clean, features, labels, y_model
        