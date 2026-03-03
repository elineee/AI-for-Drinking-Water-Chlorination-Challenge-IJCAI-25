from models.LOF import LOFModel
from utils import detect_change_point
from data_transformation import calculate_labels_alarm

class LOFAlarmModel(LOFModel):
    """ Class for Local Outlier Factor (LOF) model"""
    
    def _calculate_labels(self, df, contaminant, window_size):
        return calculate_labels_alarm(df, contaminant, window_size)

    def _post_predictions(self, y_pred):
        return detect_change_point(y_pred)