from data_transformation import calculate_labels_alarm
from utils import detect_change_point
from models.LSTM_VAE import LSTMVAEModel

class LSTMVAEAlarmModel(LSTMVAEModel):
    """ Class for LSTM VAE with alarm model"""

    def _calculate_labels(self, df, contaminant, window_size):
        return calculate_labels_alarm(df, contaminant, window_size)
    
    def _post_predictions(self, y_pred):
        return detect_change_point(y_pred)