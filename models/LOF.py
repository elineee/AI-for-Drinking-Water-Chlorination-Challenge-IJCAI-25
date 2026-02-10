from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from exploration.data_transformation import change_data_format, get_data_for_one_node

clean_file = ".\data\data_arsenic\scada_data_no_contamination.csv"
contaminated_file = ".\data\data_arsenic\scada_data_conta_22.csv"
NODE = 22
WINDOW_SIZE = 10


df_cleaned = change_data_format(clean_file, to_csv=False)
df_cleaned_node = get_data_for_one_node(df_cleaned, NODE, to_csv=False)

df_contaminated = change_data_format(contaminated_file, to_csv=False)
df_contaminated_node = get_data_for_one_node(df_contaminated, NODE, to_csv=False)


    


def create_features(df, feature_col, window_size=10):
    """ create features for anomaly detection using a sliding window approach
    Parameters:
    - df: a pandas DataFrame containing the data
    - feature_col: the name of the column to use as feature
    - window_size: the size of the sliding window
    
    Returns:
    - a numpy array containing the features for each time step
    """
    feature = df[feature_col].values
    print(feature)
    
    features = []
    
    for i in range(window_size, len(feature)):
        
        window = feature[i-window_size:i]
           
        features.append([
            feature[i],
            window.mean(),
            window.std(),
            window.min(),
            window.max()
        ])
    
    return np.array(features)

def calculate_labels(df, feature_col, window_size=10):
    """ calculate labels for anomaly detection
    Parameters:
    - df: a pandas DataFrame containing the data
    - feature_col: the name of the column to use as feature
    
    Returns:
    - a numpy array containing the labels for each time step (1 if anomaly, 0 otherwise)
    """
    feature = df[feature_col].values
    labels = []
    
    for i in range(window_size, len(feature)):
        if feature[i] > 0: 
            labels.append(-1)
        else:
            labels.append(1)
    
    return np.array(labels)

        
        
X_train = create_features(df_cleaned_node, "chlorine_concentration", WINDOW_SIZE)

X_test = create_features(df_contaminated_node, "chlorine_concentration", WINDOW_SIZE)

y_true = calculate_labels(df_contaminated_node, "arsenic_concentration")
    

lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)

lof.fit(X_train)

y_pred = lof.predict(X_test)

print("accuracy:", accuracy_score(y_true, y_pred))