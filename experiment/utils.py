from matplotlib import pyplot as plt

def plot_prediction(timestamps, actual, pred, title, figsize=(15,6)):
    plt.figure(figsize=figsize)
    plt.plot(timestamps, actual, color = "red", linewidth=2.0, alpha=0.6)
    plt.plot(timestamps, pred, color = "blue", linewidth=0.8)
    plt.legend(['Actual', 'Predicted'])
    plt.xlabel('Timestamp')
    plt.title(title)
    plt.show()

def build_timestamps(datasets, window_size):
    timestamps = []
    for dataset in datasets:
        timestamps += list(range(len(dataset) - window_size))
    return timestamps