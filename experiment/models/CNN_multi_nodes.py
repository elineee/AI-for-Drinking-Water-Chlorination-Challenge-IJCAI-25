import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import  TensorDataset, DataLoader
from data_transformation import CONTAMINANT_ID, get_labels
from utils import detect_change_point
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
        
        self.conv_out = nn.Conv1d(in_channels=64, out_channels=input_size, kernel_size=1)
        
    def forward(self, x):
        x = x.transpose(1, 2)  # -> (batch, number_of_features, window_size)
        
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


class CNNMultiNodesModel(AnomalyModel):
    """ Class for CNN multivariate model. It takes into account the raw signal and the signal given by another model (by default, model).
    Note: In the CNN configuration, the last file of contaminated_files si the file for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU not available, using CPU")

    def _get_input_size(self):
        """
        Returns the number of input channels for the CNN model.
        """
        return len(self.config.nodes)


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
        
        if n_anomalous != 0: 
            weights = torch.tensor([n_normal / n_anomalous], dtype=torch.float32, device=self.device)
        else: 
            weights = torch.tensor([n_normal / 1], dtype=torch.float32, device=self.device) 

        return weights
    

    def run_model(self, train_dataloader, val_dataloader, test_dataloader, weights, epochs=10):
        """ 
        Trains the CNN model and evaluates it on the test set.
        The model predicts a label for each point in each window. 
        For each time step, labels are given by majority vote: it is an anomaly (-1) if more than 50% of the windows covering it predict an anomaly.

        Parameters:
        - train_dataloader: DataLoader for the training set.
        - val_dataloader: DataLoader for the validation set
        - test_dataloader: DataLoader for the test set
        - weights: tensor containing the weight for the positive class (anomalies)
        - epochs: number of epochs 
   
        Returns:
        - results_per_node : a dictionary mapping each node to its corresponding predicted labels (numpy array), where -1 corresponds to an anomaly and 1 to a normal point
        """
        model = CNN(input_size=self._get_input_size()).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=weights) # loss for binary classification
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        if os.path.exists("cnn_multi_nodes.pth"):
            model.load_state_dict(torch.load("cnn_multi_nodes.pth", map_location=self.device, weights_only=True))
            print("Model loaded from file.")
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
                    windows, labels = data # windows shape (batch, window_size, number of features), labels shape (batch, window_size)
                    windows = windows.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(windows) # outputs shape (batch, N_nodes, window_size)
                    outputs = outputs.transpose(1, 2) # -> (batch, window_size, N_nodes)
                    
                    optimizer.zero_grad()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                    
                    probs = torch.sigmoid(outputs) # Convert logits to probabilities
                    preds = (probs > 0.5).float() # Threshold at 0.5 to get binary predictions 
        
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
                        windows, labels = data # windows shape (batch, window_size, number of features), labels shape (batch, window_size)
                        windows = windows.to(self.device)
                        labels = labels.to(self.device)

                        outputs = model(windows) # outputs shape (batch, N_nodes, window_size)
                        outputs = outputs.transpose(1, 2) # -> (batch, window_size, N_nodes)
                        loss = criterion(outputs, labels)
                        losses.append(loss.item())
                        
                        probs = torch.sigmoid(outputs) # Convert logits to probabilities

                        preds = (probs > 0.5).float() # Threshold at 0.5 to get binary predictions 
                        
                        val_preds_all.append(preds.flatten().cpu().numpy())
                        val_labels_all.append(labels.flatten().cpu().numpy())
                val_loss.append(np.mean(losses))
                val_preds_all = np.concatenate(val_preds_all)
                val_labels_all = np.concatenate(val_labels_all)
                val_f1 = f1_score(val_labels_all, val_preds_all, average="binary", zero_division=1)
                losses = []
                    
                
                print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss[-1]:.4f}, Training F1: {train_f1:.4f}, Validation F1: {val_f1:.4f}")
                
                # Save model with best validation F1 score
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    torch.save(model.state_dict(), f"cnn_multi_nodes.pth")
                    print(f" Best model saved with validation F1: {best_val_f1:.4f}")
                
            # plt.figure()
            # plt.plot(train_loss, label="train")
            # plt.plot(val_loss, label="validation")
            # plt.title("Loss evolution over epochs")
            # plt.xlabel("epoch")
            # plt.ylabel("loss")
            # plt.legend()
            # plt.show()
            
            model.load_state_dict(torch.load(f"cnn_multi_nodes.pth", map_location=self.device, weights_only=True))

        
        model.eval()
        all_preds = []
        all_labels = []
        
        n_nodes = len(self.config.nodes)
        n_test_timesteps = len(test_dataloader.dataset) + self.config.window_size # number of time steps in the test set
        values = np.zeros((n_nodes, n_test_timesteps)) # to store the sum of predicted labels for each time step across all windows and all nodes
        counts = np.zeros((n_nodes, n_test_timesteps)) # to store the count

        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                windows, labels = data # windows shape (batch, window_size, number of features), labels shape (batch, window_size)
                windows = windows.to(self.device)
                labels = labels.to(self.device)

                outputs = model(windows) 
                outputs = outputs.transpose(1, 2)
                
                probs = torch.sigmoid(outputs) # Convert logits to probabilities

                preds = (probs > 0.5).float() # Threshold at 0.5 to get binary predictions 
                
                preds_np = preds.squeeze(0).cpu().numpy() # shape (window_size, number of nodes)
                labels_np = labels.squeeze(0).cpu().numpy()
                
                # for each time step
                for j in range(preds_np.shape[0]):  
                    # calculte the real time step in the original time series corresponding to the j-th time step in the window
                    timestep = i + j
                    # add the values to the corresponding time step
                    values[:, timestep]  += preds_np[j, :] 
                    # count the number of predictions for each time step
                    counts[:, timestep] += 1

                flat_preds  = preds_np.flatten()
                flat_labels = labels_np.flatten()
                all_preds.append(flat_preds)
                all_labels.append(flat_labels)
                
            all_preds = np.concatenate(all_preds)  
            all_labels = np.concatenate(all_labels) 
            f1 = f1_score(all_labels, all_preds, average="binary", zero_division=1)
            recall = recall_score(all_labels, all_preds, average="binary", zero_division=1)

        print(f"Final F1 score: {f1:.4f}")
        print(f"Final Recall: {recall:.4f}")
        
        # For each time step, we calculate the mean predicted label across all windows and all nodes. If the mean is greater than 0.5, we predict an anomaly (-1), otherwise we predict a normal point (1).
        mean_votes = np.divide(values, counts, where=counts > 0)
        
        results_per_node = {}
        for node_idx in range(n_nodes):
            node_labels = []
            for t in range(n_test_timesteps):
                if counts[node_idx, t] == 0:
                    node_labels.append(1)      
                elif mean_votes[node_idx, t] > 0.5:
                    node_labels.append(-1)        
                else:
                    node_labels.append(1)        
            results_per_node[node_idx] = node_labels

        return results_per_node
    

    def get_results(self):
        
        nodes = self.config.nodes
        
        dfs = []
        for file in self.config.contaminated_files[:-1]:
            df = pd.read_csv(file)
            dfs.append(df)
        
        dfs = self.add_noisy_dfs(dfs, nodes)
        
        dfs.append(pd.read_csv(self.config.contaminated_files[-1])) 


        # init dictionaries to store the time series and labels for each node for the train set 
        dict_time_series = {}
        dict_labels = {}
        for node in nodes:
            dict_time_series[node] = []
            dict_labels[node] = []
        
        # get the features and labels for the training set 
        for df in dfs[:-1]: 
            for node in nodes: 
                features, labels = self.prepare_data(df, node)
                dict_time_series[node].append(features)
                dict_labels[node].append(labels)
       
        # init dictionaries to store the time series and labels for each node for the test set
        dict_time_series_test = {}
        dict_labels_test = {}
        for node in nodes:
            dict_time_series_test[node] = []
            dict_labels_test[node] = []
        
        # get the features and labels for the test set
        df = dfs[-1]
        for node in nodes: 
            features, labels = self.prepare_data(df, node)
            dict_time_series_test[node].append(features)
            dict_labels_test[node].append(labels)
        
        # get the true labels for each node   
        y_true = self.get_y_true()

        # concatenate the time series features and labels for each node to create the final training and test sets
        data_train = torch.stack([torch.cat(dict_time_series[node], dim=0) for node in nodes], dim=2) # shape of (number of total train elements, window size, number of nodes)
        y_train = torch.stack([torch.cat(dict_labels[node], dim=0) for node in nodes], dim=2) # shape of (number of total train elements, window size, number of nodes)
        
        data_test = torch.stack([torch.cat(dict_time_series_test[node], dim=0) for node in nodes], dim=2) # shape of (number of total test elements, window size, number of nodes) 
        y_test = torch.stack([torch.cat(dict_labels_test[node], dim=0) for node in nodes], dim=2) # shape of (number of total test elements, window size, number of nodes)
        
        # split the training set into a training set and a validation set
        X_train, X_val, y_train, y_val = train_test_split(data_train, y_train, test_size=0.15, random_state=42)
        
        # create DataLoaders for the training, validation and test sets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(data_test, y_test)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True) # one batch = (batch_size=32, window_size, number of features)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        weights = self._compute_weight(y_train)
        y_pred = self.run_model(train_dataloader, val_dataloader, test_dataloader, weights, epochs=50)
        
        
        dict_pred = {}
        for key, value in y_pred.items(): 
            dict_pred[self.config.nodes[key]] = detect_change_point(value, count_required=50)
            
        # get the results 
        results = {}
        for node in self.config.nodes:
            if y_true[node] is not None:
                results[node] = {
                    "y_true": self.get_label_alarm(y_true[node]),
                    "y_pred": self.get_label_alarm(dict_pred[node])
                }
            else:
                print(f"No true labels for node {node}, skipping evaluation.")

        return results
    

    
    
    def get_label_alarm(self, label):
        """ 
        Calculates labels for anomaly detection. Labels are -1 from the moment the value of the contaminant column becomes an anomaly (> 0) and 1 before that.
        
        Parameters:
        - label: a numpy array containing the values of whether there is an anomaly (-1) or not (1) for each time step in the test set
        
        Returns:
        - labels: a numpy array containing the labels alarm where everything is 1 before the first anomaly and -1 from the first anomaly
        """
        anomaly_started = False
        labels = []
        for element in label:
            if element == -1: 
                anomaly_started = True

            if anomaly_started:
                labels.append(-1)
            else:
                labels.append(1)

        labels = np.array(labels)
        
        return labels
    
    def get_y_true(self): 
        """ 
        Retrieves the true labels for each node in the experiment.
        
        Returns:
        - y_true: a dictionary mapping each node to its corresponding true labels (numpy array), where -1 corresponds to an anomaly and 1 to a normal point
        """
        y_true = {}
        df = pd.read_csv(self.config.contaminated_files[-1])
        for node in self.config.nodes:
             contaminant_id = CONTAMINANT_ID[self.config.contaminants[0]]
             column_label_name = f"bulk_species_node [MG] at {contaminant_id} @ {node}" 
             if column_label_name in df.columns:
                label = df[column_label_name].values.copy()
                for i in range(len(label)):
                    if label[i] > 0:
                        label[i] = -1
                    else:
                        label[i] = 1
                if "dist" in node:
                    label = label[288*3:] # remove first 3 days 
                else: 
                    label = label[48*3:] # remove first 3 days 
                y_true[node] = label
             else:
                print(f"Column {column_label_name} not found in DataFrame.")
                y_true[node] = None
        return y_true
    
    def prepare_data(self, df, node):
        """ Prepares the data for training and testing the CNN model. It retrieves the time series and labels for a given node, removes the first 3 days, and creates windows of the specified size.
        
        Parameters:
        - df: the DataFrame containing the data for a contaminated file
        - node: the node for which to prepare the data
        
        Returns:
        - features: a tensor containing the windows of the time series for the given node, where each window has the shape (window_size, number of features)
        - labels: a tensor containing the corresponding labels for each window, where each label has the shape (window_size, number of features) and indicates whether there is an anomaly (-1) or not (1) for each time step in the window
        """
        column_name_cl = f"bulk_species_node [MG] at Chlorine @ {node}"
        contaminant_id = CONTAMINANT_ID[self.config.contaminants[0]]
        column_label_name = f"bulk_species_node [MG] at {contaminant_id} @ {node}" 
        if column_name_cl in df.columns:
            time_serie = df[column_name_cl].values
        else:
            print(f"Column {column_name_cl} not found in DataFrame.")
        if column_label_name in df.columns:
            label = df[column_label_name].values
            label = get_labels(label)
    
        if "dist" in node:
            time_serie = time_serie[288*3:] # remove first 3 days 
            label = label[288*3:]
        else: 
            time_serie = time_serie[48*3:] # remove first 3 days 
            label = label[48*3:]
        
        features = []
        labels = []
        for i in range(self.config.window_size, len(time_serie)):
            row = time_serie[i-self.config.window_size:i]
            label_value = label[i-self.config.window_size:i]
            
            features.append(row)
            labels.append(label_value)
        features = np.array(features)
        features = torch.tensor(features, dtype=torch.float32)
        labels = np.array(labels)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return features, labels
    
    
    def gaussian_noise(self, x):
        """ Adds gaussian noise to the input array x. The noise has a mean of 0 and a standard deviation randomly chosen from a predefined list. 
        
        Parameters:
        - x: input array to which the noise will be added
        
        Returns:
        - x with added gaussian noise
        """
        mu = 0.0
        std = [0.01, 0.03, 0.05, 0.07]
        noise = np.random.normal(mu, np.random.choice(std), size = x.shape)
        x_noisy = x + noise
        return x_noisy 

    def blank_values(self, x):
        """ Adds blank values to the input array x.
        
        Parameters:
        - x: input array to which blank values will be added

        Returns:
        - x with added blank values
        """
        percentage = [0.01, 0.03, 0.05]
        x_noised = x.copy()
        num_defects = int(np.random.choice(percentage) * len(x))
        defect_indices = np.random.choice(len(x), num_defects, replace=False)
        x_noised[defect_indices] = 0
        return x_noised

    def add_noisy_dfs(self, dfs, nodes):
        """ 
        Adds noisy versions of the dataframes to the original list of dataframes. For each dataframe, a version with gaussian noise and a version with blank values are created with a certain probability.
        
        Parameters:
        - dfs: list of dataframes to which the noisy versions will be added
        - nodes: list of nodes for which to create noisy versions
        
        Returns:
        - a list of dataframes including the original and the noisy versions
        """
        noisy_dfs = []
        proba_gauss = 0.7
        proba_blank = 0.7
        for df in dfs:
            noisy_dfs.append(df) 
            df_copy = df.copy()
            if np.random.rand() < proba_gauss:
                for node in nodes:
                    column_name = f"bulk_species_node [MG] at Chlorine @ {node}"
                    df_copy[column_name] = self.gaussian_noise(df_copy[column_name].values)
            if np.random.rand() < proba_blank:
                for node in nodes:
                    column_name = f"bulk_species_node [MG] at Chlorine @ {node}"
                    df_copy[column_name] = self.blank_values(df_copy[column_name].values)
            noisy_dfs.append(df_copy)
        return noisy_dfs
        