import pandas as pd
from sklearn.ensemble import IsolationForest
from models.model import AnomalyModel


class IsolationForestModel(AnomalyModel):
    """Class for IsolationForest Model."""

    def get_results(self):
        results = {}
        _, all_contaminated_dfs = self.load_datasets_as_dict()

        for node, contaminated_dfs in all_contaminated_dfs.items():
            prepared_contaminated_dfs, X = self._prepare_dataset(contaminated_dfs)
            prepared_contaminated_df = pd.concat(prepared_contaminated_dfs)
            
            y_true = self._calculate_labels(prepared_contaminated_df, self.config.contaminants[0].value, self.config.window_size)

            contamination = self.config.model_params.get("contamination", 0.1)
            model = IsolationForest(contamination=contamination, random_state=42)
            y_pred = model.fit_predict(X)
            y_pred = self._post_predictions(y_pred)

            results[node] = {"y_true": y_true, "y_pred": y_pred}

        return results