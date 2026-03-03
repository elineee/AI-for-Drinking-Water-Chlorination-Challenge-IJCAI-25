from data_transformation import calculate_labels_alarm
from utils import detect_change_point
from models.SVR import SVRModel

class SVRAlarmModel(SVRModel):
    """ Class for SVR model that alarms when start of contamination is detected"""
    
    def _get_threshold_multiplier(self):
        return 25

    def _calculate_labels(self, df, contaminant, window_size):
        return calculate_labels_alarm(df, contaminant, window_size)

    def get_results(self):
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()
        results = {}

        for node, clean_dfs in all_clean_dfs.items():
            contaminated_dfs = all_contaminated_dfs[node] 

            y_true, y_pred, _, _ = self.predict(node, clean_dfs, contaminated_dfs)
            y_pred = detect_change_point(y_pred, 3) 

            results[node] = {"y_true": y_true, "y_pred": y_pred}

        return results