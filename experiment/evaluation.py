from enum import Enum
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, recall_score, f1_score
import pickle
import matplotlib.pyplot as plt
import numpy as np

class Metrics(Enum):
    """ Enumeration of evaluation metrics"""
    CONFUSION_MATRIX = "confusion_matrix"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    DELAY = "delay"    


class ScalarMetrics(Enum):
    """ Enumeration of scalar evaluation metrics (metrics that can be averaged across nodes)"""
    RECALL = "recall"   
    F1_SCORE = "f1_score"

class Evaluation:
    """
    Class with evaluation methods.
    """

    def __init__(self):
        super().__init__()

    def evaluate(self, results_file: str):
        """
        Evaluates the results of the experiments by calculating metrics for each experiment configuration. 
        The results are loaded from a pickle file containing the results of the experiments.

        Parameters:
        - results_file: the path to the pickle file. The file should contain a list of dictionaries containing the true labels (y_true) and predicted labels (y_pred) for each experiment configuration.

        Returns:
        - evaluation_results: a dictionary with the evaluation metrics for each experiment configuration.
        """

        evaluation_results = {}
        results = pickle.load(open(results_file, 'rb'))

        for result in results:
            for config_name in result: 
                nodes_dict = result[config_name]

                evaluation_results[config_name] = {}

                for node in nodes_dict: 
                    values = nodes_dict[node]

                    y_true = values["y_true"]
                    y_pred = values["y_pred"]
                    
                    delay = 0
                    for i in range(len(y_true)):
                        if y_true[i] == -1 and y_pred[i] == 1:
                            delay += 1
                        elif y_true[i] == -1 and y_pred[i] == -1:
                            break
                    

                    evaluation_results[config_name][node] = {
                        Metrics.CONFUSION_MATRIX.value: confusion_matrix(y_true, y_pred, labels=[1, -1]),
                        Metrics.RECALL.value: recall_score(y_true, y_pred, pos_label=-1, zero_division=0),
                        Metrics.F1_SCORE.value: f1_score(y_true, y_pred, pos_label=-1, zero_division=0),
                        Metrics.DELAY.value: delay
                    }

        return evaluation_results

    def plot_confusion_matrices(self, config_name: str, evaluation_results: dict):
        """
        Plots the confusion matrices for a specific configuration. If the configuration contains multiple nodes, it plots a confusion matrix for each node.
        This allows to visualize the performance of the model in terms of true positives, true negatives, false positives and false negatives for each configuration and node.

        Parameters:
        - config_name: the name of the configuration for which to plot the confusion matrices.
        - evaluation_results: the dictionary containing the evaluation results for each experiment configuration and node, as returned by the evaluate method.
        """

        if config_name not in evaluation_results:
            raise ValueError(f"{config_name} not found in evaluation results.")

        nodes_dict = evaluation_results[config_name]

        for node in nodes_dict:
            cm = nodes_dict[node][Metrics.CONFUSION_MATRIX.value]
            display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal (1)', 'Anomaly (-1)'])
            display.plot(cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix for {config_name} - Node {node}')
            plt.show()

