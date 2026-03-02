from models.LSTM_AE import LSTMAutoEncoderModel
from utils import detect_change_point

class LSTMAutoEncoderAlarmModel(LSTMAutoEncoderModel):

    def get_results(self):

        results = super().get_results()

        for node in results:
            anomalies = results[node]["y_pred"]

            y_pred = detect_change_point(anomalies)

            results[node]["y_pred"] = y_pred

        return results