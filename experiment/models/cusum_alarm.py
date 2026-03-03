
from models.cusum import CusumModel
from data_transformation import calculate_labels_alarm
from utils import detect_change_point

class CusumAlarmModel(CusumModel):
    """ Class for CUSUM with alarm model"""

    def _get_threshold_multiplier(self):
        return 6

    def _calculate_labels(self, df, contaminant, window_size):
        return calculate_labels_alarm(df, contaminant, window_size)

    def _post_predictions(self, y_pred):
        return detect_change_point(y_pred, 15)

