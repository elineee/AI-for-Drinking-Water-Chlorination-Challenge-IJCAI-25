import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from models.model import AnomalyModel


class LOFModel(AnomalyModel):
    """ Class for Local Outlier Factor (LOF) model"""
    
    def _prepare_data(self, clean_dfs: list[pd.DataFrame], contaminated_dfs: list[pd.DataFrame]):
        """
        Prepares train/test data for the LOF model.

        Parameters:
        - clean_dfs: dataframes with training data (clean data)
        - contaminated_dfs: dataframes with testing data (contaminated data)

        Returns:
        - X_train: training features
        - X_test: test features
        - prepared_contaminated_dfs: contaminated dataframes after preprocessing
        """
        _, X_train = self._prepare_dataset(clean_dfs)
        prepared_contaminated_dfs, X_test = self._prepare_dataset(contaminated_dfs)

        return X_train, X_test, prepared_contaminated_dfs


    def get_results(self):
        all_clean_dfs, all_contaminated_dfs = self.load_datasets_as_dict()
        results = {}

        for node, clean_dfs in all_clean_dfs.items():
            contaminated_dfs = all_contaminated_dfs[node]
            X_train, X_test, prepared_contaminated_dfs = self._prepare_data(clean_dfs, contaminated_dfs)
            prepared_contaminated_df = pd.concat(prepared_contaminated_dfs)

            y_true = self._calculate_labels(prepared_contaminated_df, self.config.contaminants[0].value, self.config.window_size)

            n_neighbors = self.config.model_params.get("n_neighbors", 20)
            contamination = self.config.model_params.get("contamination", 0.1)

            lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=contamination)
            lof.fit(X_train)

            y_pred = lof.predict(X_test)
            y_pred = self._post_predictions(y_pred)

            results[node] = {"y_true": y_true, "y_pred": y_pred}

        return results