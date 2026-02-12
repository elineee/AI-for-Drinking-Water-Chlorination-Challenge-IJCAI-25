from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score
import pickle


class Evaluation:
    """
    Class with evaluation methods.
    """

    def __init__(self):
        super().__init__()

    def evaluate(self, results_file: str):
        """
        Evaluates the results of the experiments by calculating accuracy, confusion matrix, recall and F1 score for each experiment configuration. 
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
                values = result[config_name]
                y_true = values["y_true"]
                y_pred = values["y_pred"]

                evaluation_results[config_name] = {
                    "accuracy": accuracy_score(y_true, y_pred),
                    "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[1, -1]),
                    "recall": recall_score(y_true, y_pred, pos_label=-1),
                    "f1_score": f1_score(y_true, y_pred, pos_label=-1)
                }

        return evaluation_results


        
  