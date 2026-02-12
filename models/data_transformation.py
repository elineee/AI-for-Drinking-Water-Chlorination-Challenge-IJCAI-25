import numpy as np
import pandas as pd

from experiment_config import ContaminationType


def change_data_format(file_name: str, contaminants: list[ContaminationType], to_csv: bool = False):
    """ 
    Changes data format to have 1 row per node and timestep, with columns for chlorine concentration and contaminant concentration (e.g., arsenic) 

    Parameters:
    - file_name: the path to the data file (csv) to transform
    - contaminants: list of ContaminationType to specify which contaminants to extract from the data
    - to_csv: whether to save the transformed data to a csv file
    
    Returns:
    - new_df : a pandas DataFrame containing the transformed data with columns: timestep, node, chlorine_concentration, contaminant_concentration (e.g., arsenic_concentration)
    """
    df = pd.read_csv(file_name)
    
    elements_to_keep = ["Chlorine"]
    
    contaminant_column_names = []
    
    # get id of contaminant to keep the right columns (e.g., AsIII for arsenic)
    for contaminant in contaminants:
        contaminant_id = get_contaminant_id(contaminant)
        elements_to_keep.append(contaminant_id)
        contaminant_column_names.append(f'{contaminant.value}_concentration') 
    
    # keep columns containing chlorine or the contaminant in their name
    df_cleaned = df[[column for column in df.columns if any(element in column for element in elements_to_keep)]]

    new_data = {
        "timestep": [],
        "node": [],
        "chlorine_concentration": [],    
    }
    
    for column in contaminant_column_names:
        new_data[column] = []
    
    
    nodes = {get_node_number(column) for column in df_cleaned.columns}
        
    for timestep, row in df_cleaned.iterrows():
        for node in nodes:
            chlorine_column = f'bulk_species_node [MG] at Chlorine @ {node}'
            for contaminant in contaminants:
                contaminant_id = get_contaminant_id(contaminant)
                contaminant_column = f'bulk_species_node [MG] at {contaminant_id} @ {node}'
                column_name = f'{contaminant.value}_concentration'
                new_data[column_name].append(row[contaminant_column])
            
            new_data["timestep"].append(timestep)
            new_data["node"].append(node)
                                        
            if chlorine_column in df_cleaned.columns:
                new_data["chlorine_concentration"].append(row[chlorine_column])
            else:
                new_data["chlorine_concentration"].append(np.nan)
    
    new_df = pd.DataFrame(new_data)
    
    if to_csv:
        new_df.to_csv(file_name.replace(".csv", "_cleaned.csv"), index=False)
    
    return new_df
    

def get_node_number(column_name: str):
    """
    Get the node number from a column name

    Parameters:
    - column_name: the name of the column to extract the node number from
    
    Returns:
    - the node number extracted from the column name
    """
    return int(column_name.split(" @ ")[1].split(" ")[0])


def get_data_for_one_node(data: str, node_number: int, to_csv: bool = False):
    """ 
    Extracts data for one node and returns it as a pandas DataFrame.

    Parameters:
    - data: a file path (str) or a pandas DataFrame containing the data
    - node_number: the number of the node to extract
    - to_csv: whether to save the extracted data to a csv file
    
    Returns:
    - new_df: a pandas DataFrame containing the data for the specified node
    """
    
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("`data` must be a file path (str) or a pandas DataFrame")
    
    new_df = df[df["node"] == node_number].copy()

    if new_df.empty:
        raise ValueError(f"No data found for node {node_number}")

    if to_csv:
        new_df.to_csv(f"node_{node_number}.csv", index=False)
        
    return new_df


def create_features(df: pd.DataFrame, feature_column: str, window_size: int = 10):
    """ 
    Creates features for anomaly detection using a sliding window approach. 
    For each time step, features are: current value, mean, std, min and max of the values in the sliding window.

    Parameters:
    - df: a pandas DataFrame containing the data
    - feature_col: the name of the column to use as feature
    - window_size: the size of the sliding window
    
    Returns:
    - a numpy array containing the features for each time step
    """
    for column in df.columns:
        if feature_column in column:
            feature_column = column
            break
    
    feature = df[feature_column].values
    
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
def calculate_labels(df: pd.DataFrame, feature_column: str, window_size: int): 
    """ 
    Calculates labels for anomaly detection

    Parameters:
    - df: a pandas DataFrame containing the data
    - feature_col: the name of the column to use as feature
    
    Returns:
    - labels: a list containing the labels for each time step (1 if anomaly, 0 otherwise)
    """
    
    # get the column name containing the chlorine concentration for the node
    for column in df.columns:
        if feature_column in column:
            feature_column = column
            break

    feature = df[feature_column].values
    labels = []
    
    for i in range(window_size, len(feature)):
        if feature[i] > 0: 
            labels.append(-1)
        else:
            labels.append(1)

    labels = np.array(labels)
    
    return labels


CONTAMINANT_ID = {
    ContaminationType.ARSENIC: "AsIII",
}

def get_contaminant_id(contaminant: ContaminationType):
    """
    Returns the ID of a contaminant given its ContaminationType.

    Parameters:
    - contaminant: a ContaminationType enum value representing the contaminant

    Returns: 
    - the ID of the contaminant as a string (e.g., "AsIII" for arsenic)

    """
    return CONTAMINANT_ID[contaminant]

