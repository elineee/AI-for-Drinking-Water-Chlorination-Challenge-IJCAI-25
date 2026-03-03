from models.isolation_forest import IsolationForestModel
from data_transformation import calculate_labels_alarm
from utils import detect_change_point

class IsolationForestAlarmModel(IsolationForestModel):
    """Class for IsolationForest Model with change point detection."""

    def _calculate_labels(self, df, contaminant, window_size):
        return calculate_labels_alarm(df, contaminant, window_size)

    def _post_predictions(self, y_pred):
        count_required = self.config.model_params.get("count_required", 20)
        return detect_change_point(y_pred, count_required)