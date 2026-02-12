from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score, roc_auc_score
import pickle
class Evaluation:
    """
    Class with evaluation methods 
    """

    def __init__(self):
        super().__init__()

    def evaluate(self, results_file: str):

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
                }

        return evaluation_results


            # "roc_auc": roc_auc_score(loaded["y_true"], loaded["y_pred"]),
            # "pr_auc": average_precision_score(loaded["y_true"], loaded["y_pred"]),
  