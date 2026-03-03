from data_transformation import calculate_labels_alarm
from utils import detect_change_point
from models.one_class_SVM import OneClassSVMModel

class OneClassSVMAlarmModel(OneClassSVMModel):
    """ Class for One Class SVM with alarm model"""

    def _calculate_labels(self, df, contaminant, window_size):
        return calculate_labels_alarm(df, contaminant, window_size)

    def _post_predictions(self, y_pred):
        return detect_change_point(y_pred)