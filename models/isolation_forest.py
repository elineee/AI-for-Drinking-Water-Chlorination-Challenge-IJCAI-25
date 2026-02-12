from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
from traitlets import List, Tuple

    
from data_transformation import *
from model import AnomalyModel

class IsolationForestModel(AnomalyModel):
    """ Class for Isolation Forest model"""
    def __init__(self, config):
        super().__init__(config)
    
    
    def get_results(self):
        
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
    
    def load_and_filter(self, file_path: str, nodes: List[int]):
        """
        Loads the dataset from the given file path and filter it based on the specified nodes. Return a list of dataframes corresponding to each node.
        
        Parameters:
        - file_path: the path to the data file (csv) to load
        - nodes: a list of node numbers to filter the data by
        
        Returns:
        - dfs: a list of pandas DataFrames, each containing the data for one of the specified nodes
        """

        df_all = change_data_format(file_path, self.config.contaminants, to_csv=False)  
        
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
        """
        Loads the datasets for the experiment based on the contaminated files specified in the configuration. 
        For each contaminated file, it loads the data and filters it based on the specified nodes.
        
        Returns: 
        contamined_dfs : list of dataframes for each contaminated file.
        """     
        contaminated_dfs = []
        for fp in self.config.contaminated_files:
            contaminated_dfs.extend(self.load_and_filter(fp, self.config.nodes))
        
        return contaminated_dfs