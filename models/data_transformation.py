import numpy as np
import pandas as pd

# TODO : refactor to make it more general and only keep colums with chlorine and contaminants in the name  + add error if no columns for contaminant
def change_data_format(file_name, to_csv=True):
    "transform the data to have one row = one time step, the chlorine value, the arsenic value and the number of the node"
    df = pd.read_csv(file_name)
    
    df_cleaned = df.drop(columns=["pressure [meter] at 10","pressure [meter] at 11","pressure [meter] at 12","pressure [meter] at 13","pressure [meter] at 21","pressure [meter] at 22","pressure [meter] at 23","pressure [meter] at 31","pressure [meter] at 32","pressure [meter] at 9","pressure [meter] at 2","flow [cubic meter/hr] at 10","flow [cubic meter/hr] at 11","flow [cubic meter/hr] at 12","flow [cubic meter/hr] at 21","flow [cubic meter/hr] at 22","flow [cubic meter/hr] at 31","flow [cubic meter/hr] at 110","flow [cubic meter/hr] at 111","flow [cubic meter/hr] at 112","flow [cubic meter/hr] at 113","flow [cubic meter/hr] at 121","flow [cubic meter/hr] at 122","flow [cubic meter/hr] at 9"])

    new_data = {
        "timestep": [],
        "node": [],
        "chlorine_concentration": [],
        "arsenic_concentration": []
    }
    
    nodes = set()
    
    for column in df_cleaned.columns:
        number = get_node_number(column)
        nodes.add(number)
        
    for timestep, row in df_cleaned.iterrows():
        for node in nodes:
            chlorine_col = f'bulk_species_node [MG] at Chlorine @ {node}'
            arsenic_col = f'bulk_species_node [MG] at AsIII @ {node}'
            
            if chlorine_col in df_cleaned.columns and arsenic_col in df_cleaned.columns:
                new_data["timestep"].append(timestep)
                new_data["node"].append(node)
                new_data["chlorine_concentration"].append(row[chlorine_col])
                new_data["arsenic_concentration"].append(row[arsenic_col])
        
    new_df = pd.DataFrame(new_data)
    
    if to_csv:
        new_df.to_csv(file_name.replace(".csv", "_cleaned.csv"), index=False)
    
    return new_df
    

def get_node_number(column_name):
    "extract the number from the column name to get the node number"
    return int(column_name.split(" @ ")[1].split(" ")[0])

# TODO add error if no nodes found in the data
def get_data_for_one_node(data, node_number, to_csv=True):
    """ extract the data for one node
    Parameters:
    - data: a file path (str) or a pandas DataFrame containing the data
    - node_number: the number of the node to extract
    - to_csv: whether to save the extracted data to a csv file
    
    Returns:
    - a pandas DataFrame containing the data for the specified node
    """
    
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("`data` must be a file path (str) or a pandas DataFrame")

    new_data = {
        "timestep": [],
        "node": [],
        "chlorine_concentration": [],
        "arsenic_concentration": []
    }
    
    for _, row in df.iterrows():
        if row["node"] == node_number:
            new_data["timestep"].append(int(row["timestep"]))
            new_data["node"].append(int(row["node"]))
            new_data["chlorine_concentration"].append(row["chlorine_concentration"])
            new_data["arsenic_concentration"].append(row["arsenic_concentration"])
    
    new_df = pd.DataFrame(new_data)
    
    if to_csv:
        new_df.to_csv(f"node_{node_number}.csv", index=False)
        
    return new_df

def create_features(df, feature_col, window_size=10):
    """ create features for anomaly detection using a sliding window approach
    Parameters:
    - df: a pandas DataFrame containing the data
    - feature_col: the name of the column to use as feature
    - window_size: the size of the sliding window
    
    Returns:
    - a numpy array containing the features for each time step
    """
    # get the col name containing the chlorine concentration for the node
    for column in df.columns:
        if feature_col in column:
            feature_col = column
            break
    
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

# TODO : gère que pour une feature pour l'instant, à faire pour plusieurs contaminants
def calculate_labels(df, feature_col, window_size=10):
    """ calculate labels for anomaly detection
    Parameters:
    - df: a pandas DataFrame containing the data
    - feature_col: the name of the column to use as feature
    
    Returns:
    - a numpy array containing the labels for each time step (1 if anomaly, 0 otherwise)
    """
    for column in df.columns:
        if feature_col in column:
            feature_col = column
            break

    feature = df[feature_col].values
    labels = []
    
    for i in range(window_size, len(feature)):
        if feature[i] > 0: 
            labels.append(-1)
        else:
            labels.append(1)
    
    return np.array(labels)

    