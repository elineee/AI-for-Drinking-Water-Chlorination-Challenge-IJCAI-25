import pandas as pd
from sklearn.ensemble import IsolationForest
from data_transformation import calculate_labels
from models.model import AnomalyModel


class IsolationForestModel(AnomalyModel):
    """Class for IsolationForest Model."""

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
            model = IsolationForest(contamination=contamination, random_state=42)
            y_pred = model.fit_predict(X)
            y_pred = self._post_predictions(y_pred)

            y_true = calculate_labels(contaminated_df, self.config.contaminants[0].value, 0)

            results[node] = {"y_true": y_true, "y_pred": y_pred}

        return results