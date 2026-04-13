import pickle

from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def plot_prediction(timestamps , actual, pred, title, figsize=(15,6)):
    """
    Plots the actual vs predicted signal over time.

    Parameters:
    - timestamps: timestamps for the x-axis
    - actual:  actual signal values
    - pred: predicted/reconstructed signal values
    - title: title of the plot
    - figsize: figure size (default: (15, 6))
    """

    plt.figure(figsize=figsize)
    plt.plot(timestamps, actual, color = "red", linewidth=2.0, alpha=0.6)
    plt.plot(timestamps, pred, color = "blue", linewidth=0.8)
    plt.legend(['Actual', 'Predicted'])
    plt.xlabel('Timestamp')
    plt.title(title)
    plt.show()

def build_timestamps(datasets, window_size):
    """
    Builds a list of timestamps for each sample produced by a sliding window. It doesn't return real timestamps of the dataset. 

    Parameters:
    - datasets: list of DataFrames used 
    - window_size: size of the sliding window

    Returns:
    - a list of timestamps
    """
    timestamps = []
    for dataset in datasets:
        timestamps += list(range(len(dataset) - window_size))
    return timestamps

    
def cusum_detection(data, reference_mean, reference_std, k, threshold, counter=20):
    """
    Detects anomalies using CUSUM applied on reconstruction errors. Since reconstruction errors are always >= 0, only increases are monitored.
    Once the alarm is triggered, it remains active until the end.
    
    Parameters:
    - data: array of reconstruction errors
    - reference_mean: mean of the training reconstruction errors (normal behavior)
    - reference_std: std of the training reconstruction errors (normal behavior)
    - k: parameter controlling sensitivity
    - threshold: alarm threshold
    - counter: number of consecutive points to consider for anomaly detection
    
    Returns:
    - anomalies: numpy array of 1 (normal) and -1 (anomaly)
    - cusum : CUSUM scores over time
    """    
    
    threshold = threshold*25

    n = len(data)
    cusum = np.zeros(n)
    
    for i in range(1, n):
        cusum[i] = max(0, cusum[i-1] + data[i] - reference_mean - k * reference_std)
     
    anomalies = []
    alarm = False
    count = 0
    for c in cusum:
        if c > threshold or alarm:
            count += 1
            if count >= counter:
                alarm = True
                anomalies.append(-1)
            else: 
                anomalies.append(1)
        else:
            anomalies.append(1)
            count = 0

    return np.array(anomalies), cusum


def detect_change_point(predictions: np.array, count_required=20):
        """Detects the change point and returns an array of 1 until the change point and -1 after the change point """
        """
        Scans predictions sequentially and triggers an alarm when detecting a change point. All remaining predictions after the alarm are set to -1.

        Parameters:
        - predictions: array of 1 (normal) and -1 (anomaly)
        - count_required: number of consecutive anomalies required to trigger the alarm (default=20)

        Returns:
        - numpy array of 1 (normal) and -1 (anomaly), with permanent alarm after change point
        """
        
        y_pred = []
        counter = 0
        for i in range(len(predictions)):
            element = predictions[i]
            if element == -1:
                counter += 1
                if counter >= count_required:
                    y_pred.append(-1)
                    y_pred.extend([-1] * (len(predictions) - i - 1))
                    return np.array(y_pred)
                else:
                    y_pred.append(1)
            else:
                counter = 0
                y_pred.append(1)
        return np.array(y_pred)

def get_interesting_and_similar_nodes(df, contaminated_node, dependencies_dict_path, cluster_number):
    """ Returns a cluster of nodes with similar behavior and interesting to monitor using k_means clustering.
    
    Parameters:
    - df: the dataframe containing the time series data
    - contaminated_node: the node where the contamination event occurs
    - dependencies_dict_path: path to the dependencies dict that contains the interesting nodes to monitor for each contaminated node
    - cluster: the cluster number to select among the different cluster of interesting nodes (can be 0, 1 or 2)
    
    Returns:
    - a list of interesting nodes behaving similarly
    
    """
    
    with open(dependencies_dict_path, 'rb') as f:
        dependencies_dict = pickle.load(f)
    interesting_nodes = dependencies_dict[contaminated_node]
    
    time_series_data = get_times_series(df, interesting_nodes)
    
    X = np.array(list(time_series_data.values()))
    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit_predict(X)
    print(kmeans)

    cluster_nodes = {}
    for i in range(len(kmeans)):
        cluster = kmeans[i]
        node = interesting_nodes[i]
        if cluster not in cluster_nodes:
            cluster_nodes[cluster] = []
        cluster_nodes[cluster].append(node)
        
    return cluster_nodes[cluster_number]

def get_times_series(df, nodes): 
    """ 
    Extracts the time series data for the specified nodes from the dataframe.
    
    Parameters:
    - df: the dataframe containing the time series data
    - nodes: list of nodes to extract the time series for
    
    Returns:
    - a dictionary with node as key and time series as value
    """
    
    time_series_data = {}
    for node in nodes:
        column_name_cl = f"bulk_species_node [MG] at Chlorine @ {node}"
        if column_name_cl in df.columns:
            time_series_data[node] = df[column_name_cl].values
    return time_series_data

def gaussian_noise(x):
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

def blank_values(x):
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

def add_noisy_dfs(dfs):
    """ 
    Adds noisy versions of the dataframes to the original list of dataframes. For each dataframe, a version with gaussian noise and a version with blank values are created with a certain probability.
    
    Parameters:
    - dfs: list of dataframes to which the noisy versions will be added
    
    Returns:
    - a list of dataframes including the original and the noisy versions
    """
    noisy_dfs = []
    column_name = "chlorine_concentration"
    proba_gauss = 0.7
    proba_blank = 0.7
    for df in dfs:
        noisy_dfs.append(df) 
        df_copy = df.copy()
        if np.random.rand() < proba_gauss:
            df_copy[column_name] = gaussian_noise(df_copy[column_name].values)
        if np.random.rand() < proba_blank:
            df_copy[column_name] = blank_values(df_copy[column_name].values)
        noisy_dfs.append(df_copy)
    return noisy_dfs

