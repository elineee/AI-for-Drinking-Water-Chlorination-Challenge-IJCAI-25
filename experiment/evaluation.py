from enum import Enum
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, recall_score, f1_score
import pickle
import matplotlib.pyplot as plt
import numpy as np

class Metrics(Enum):
    """ Enumeration of evaluation metrics"""
    ACCURACY = "accuracy"
    CONFUSION_MATRIX = "confusion_matrix"
    RECALL = "recall"
    F1_SCORE = "f1_score"


class ScalarMetrics(Enum):
    """ Enumeration of scalar evaluation metrics (metrics that can be averaged across nodes)"""
    ACCURACY = "accuracy"
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
               
                    evaluation_results[config_name][node] = {
                        Metrics.ACCURACY.value : accuracy_score(y_true, y_pred),
                        Metrics.CONFUSION_MATRIX.value: confusion_matrix(y_true, y_pred, labels=[1, -1]),
                        Metrics.RECALL.value: recall_score(y_true, y_pred, pos_label=-1, zero_division=0),
                        Metrics.F1_SCORE.value: f1_score(y_true, y_pred, pos_label=-1, zero_division=0)
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


    def plot_results_nodes_within_configurations(self, metric_name: ScalarMetrics, evaluation_results: dict):
        """
        Plots the results for each configuration of the different nodes for the specified metric. 
        It allows to compare the performance of the model across different nodes for the same configuration.

        Parameters:
        - metric_name: the name of the metric to plot (ScalarMetrics)
        - evaluation_results: the dictionary containing the evaluation results for each experiment configuration and node, as returned by the evaluate method.
        """

        for config_name in evaluation_results:
            nodes_dict = evaluation_results[config_name]

            metric_values = [nodes_dict[node][metric_name.value] for node in nodes_dict]
            nodes = list(nodes_dict.keys())

            plt.bar(nodes, metric_values)
            plt.xlabel('Nodes')
            plt.ylabel(metric_name.value)
            plt.title(f'{metric_name.value} for each node in configuration {config_name}')
            plt.tight_layout()
            plt.show()

    def plot_mean_configuration(self, metric_name: ScalarMetrics, evaluation_results: dict):  
        """
        Plots the performance of each configuration by giving the mean of nodes for the specified metric.
        This allows to compare the overall performance of different models on the same metric, by aggregating the results across all nodes.

        Parameters:
        - metric_name: the name of the metric to plot (ScalarMetrics)
        - evaluation_results: the dictionary containing the evaluation results for each experiment configuration and node, as returned by the evaluate method.
        """

        config_names = []
        mean_metric_values = []

        for config_name in evaluation_results:
            nodes_dict = evaluation_results[config_name]

            metric_values = [nodes_dict[node][metric_name.value] for node in nodes_dict]
            mean_metric_value = np.mean(metric_values)

            config_names.append(config_name)
            mean_metric_values.append(mean_metric_value)

        plt.bar(config_names, mean_metric_values)
        plt.xlabel('Configurations')
        plt.ylabel(f'Mean {metric_name.value}')
        plt.title(f'Mean {metric_name.value} for each configuration')
        plt.tight_layout()
        plt.show()

        print(f"Mean of {metric_name.value} for each configuration:")
        for i, config_name in enumerate(config_names):
            print(f"  {config_name}: {mean_metric_values[i]}")