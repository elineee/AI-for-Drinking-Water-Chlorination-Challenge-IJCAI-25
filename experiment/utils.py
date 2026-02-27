from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import f1_score


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

    
def cusum_detection(data, reference_mean, reference_std, k, threshold):
    """
    Detects anomalies using the CUSUM algorithm.
    
    Parameters:
    - data: array of reconstruction errors
    - reference_mean: mean of the reconstruction errors on the training set (normal behavior)
    - reference_std: standard deviation of the reconstruction errors on the training set (normal behavior)
    - k: integer that represents the noise
    - threshold: alarm threshold
    
    Returns:
    - anomalies: array of 1 (normal) and -1 (anomaly)
    - cusum : CUSUM scores over time
    """    

    n = len(data)
    cusum = np.zeros(n)
    
    for i in range(1, n):
        cusum[i] = max(0, cusum[i-1] + data[i] - reference_mean - k * reference_std)
    
    # When cusum > threshold, then alarm 
    anomalies = []
    alarm = False
    for c in cusum:
        if c > threshold or alarm:
            alarm = True
            anomalies.append(-1)
        else:
            anomalies.append(1)

    return np.array(anomalies), cusum

def detect_change_point(predictions: np.array, count_required=20):
        """Detects the change point and returns an array of 1 until the change point and -1 after the change point """
        y_pred = []
        counter = 0
        for i in range(len(predictions)):
            element = predictions[i]
            if element == -1:
                y_pred.append(-1)
                counter += 1
                if counter >= count_required:
                    y_pred.extend([-1] * (len(predictions) - i - 1))
                    return np.array(y_pred)
            else:
                counter = 0
                y_pred.append(1)
        return np.array(y_pred)

