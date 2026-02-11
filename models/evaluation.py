from sklearn.metrics import average_precision_score, roc_auc_score

"Class with evaluation methods "
class Evaluator:

    def evaluate(self, y_true, scores):
        return {
            "roc_auc": roc_auc_score(y_true, scores),
            "pr_auc": average_precision_score(y_true, scores),
        }