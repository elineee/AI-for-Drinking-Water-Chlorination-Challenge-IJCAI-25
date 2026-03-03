from data_transformation import calculate_labels_alarm
from utils import detect_change_point
from models.SVR import SVRModel

class SVRAlarmModel(SVRModel):
    """ Class for SVR model with alarm"""
    
    def _get_threshold_multiplier(self):
        return 25

    def _calculate_labels(self, df, contaminant, window_size):
        return calculate_labels_alarm(df, contaminant, window_size)
    
    def _post_predictions(self, y_pred):
        return detect_change_point(y_pred)
