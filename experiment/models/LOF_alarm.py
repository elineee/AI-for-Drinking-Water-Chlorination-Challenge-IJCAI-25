from models.LOF import LOFModel
from utils import detect_change_point

class LOFAlarmModel(LOFModel):

    def get_results(self):

        results = super().get_results()

        for node in results:
            y_pred_temp = results[node]["y_pred"]
            results[node]["y_pred"] = detect_change_point(y_pred_temp)

        return results