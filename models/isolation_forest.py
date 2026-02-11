from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
from traitlets import List, Tuple

    
from data_transformation import *
from model import AnomalyModel


class IsolationForestModel(AnomalyModel):
    def __init__(self, config):
        super().__init__(config)
    
    
    def get_results(self) -> dict:
        contaminated_dfs = self.load_datasets()
        
        # TODO : handle multiple contaminated files, for now only one of each is handled
        X = contaminated_dfs[0][['chlorine_concentration']]
        
        # TODO : voir pour mettre + de paramètres 
        contamination = self.config.model_params.get("contamination", "auto")
        
        model = IsolationForest(contamination=contamination, random_state=42)
        y_pred = model.fit_predict(X)
        
        y_true = calculate_labels(contaminated_dfs[0], self.config.contaminants[0].value, self.config.window_size)

        
        return {
            "y_true": y_true,
            "y_pred": y_pred,
        }
    
    def load_and_filter(self, file_path: str, nodes: List[int]) -> pd.DataFrame:
        # TODO : add parameter contaminants when changed in function 
        df_all = change_data_format(file_path, self.config.contaminants, to_csv=False)  # returns rows with columns: timestep, node, chlorine_concentration, arsenic_concentration
        
        dfs = []
        
        if self.config.aggregate_method is None:
            dfs = []
            for node in nodes:
                df_node = get_data_for_one_node(df_all, node, to_csv=False)
                dfs.append(df_node)
        
        else: # TODO : handle if aggregation
            pass
        
        return dfs

    def load_datasets(self):
        """Return lists of dataframes: (contaminated_dfs) aligned with provided file lists."""     
        contaminated_dfs = []
        for fp in self.config.contaminated_files:
            contaminated_dfs.extend(self.load_and_filter(fp, self.config.nodes))
        
        return contaminated_dfs