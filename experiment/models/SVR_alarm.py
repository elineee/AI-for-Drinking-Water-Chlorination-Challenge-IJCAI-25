from data_transformation import calculate_labels_alarm
from experiment_config import ContaminationType
from utils import detect_change_point
from models.SVR import SVRModel

class SVRAlarmModel(SVRModel):
    """ Class for SVR model with alarm"""
    
    def _get_threshold_multiplier(self):
        if self.config.contaminants[0] == ContaminationType.ARSENIC:
            return 30
        else:
            return 10

    def _calculate_labels(self, df, contaminant, window_size):
        return calculate_labels_alarm(df, contaminant, window_size)
    
    def _post_predictions(self, y_pred):
        if self.config.contaminants[0] == ContaminationType.ARSENIC:
            return detect_change_point(y_pred, count_required=5)
        else:
            return detect_change_point(y_pred, 5)
