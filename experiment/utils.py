from matplotlib import pyplot as plt

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