import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from data_transformation import calculate_labels_alarm
from models.isolation_forest import IsolationForestModel
from utils import detect_change_point

class IsolationForestAlarmModel(IsolationForestModel):
    """Class for IsolationForest Model with change point detection."""

    def get_results(self):
        results = {}
        _, all_contaminated_dfs = self.load_datasets_as_dict()

        for node, contaminated_dfs in all_contaminated_dfs.items():
            contaminated_df = pd.concat(contaminated_dfs)

            feature_col = None
            for col in contaminated_df.columns:
                if self.config.disinfectant.value in col:
                    feature_col = col
                    break

            if feature_col is None:
                raise ValueError(f"No column matching '{self.config.disinfectant.value}' found")

            X = contaminated_df[[feature_col]].values

            contamination = self.config.model_params.get("contamination", "auto")
            count_required = self.config.model_params.get("count_required", 20)

            model = IsolationForest(contamination=contamination, random_state=42)
            raw_pred = model.fit_predict(X)

            y_pred = detect_change_point(raw_pred, count_required)
            y_true = calculate_labels_alarm(contaminated_df, self.config.contaminants[0].value, 0)

            results[node] = {"y_true": y_true, "y_pred": y_pred}

        return results