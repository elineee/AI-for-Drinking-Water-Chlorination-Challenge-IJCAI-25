from abc import ABC, abstractmethod
import numpy as np
from typing import List
from experiment_config import ExperimentConfig
from data_transformation import get_data_for_one_node, aggregate_data_for_several_nodes, change_data_format
class AnomalyModel(ABC):
    """ 
    Abstract class for anomaly detection models. 
    All specific models should inherit from this class and implement the "get_results" method.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def apply_threshold(self, scores, threshold=0):
        """
        Converts scores to -1/1 labels if needed.
        """
        return np.where(scores < threshold, -1, 1)

    def load_and_filter(self, file_path: str, nodes: List[int]):
        """
        Loads the dataset from the given file path and filters it based on the specified nodes. It returns a list of dataframes corresponding to each node.
        
        Parameters:
        - file_path: the path to the data file (csv) to load
        - nodes: a list of node numbers to filter the data by
        
        Returns:
        - a list of pandas DataFrames, each containing the data for one of the specified nodes
        """
        # TODO : add parameters contaminants when changed in function 
        df_all = change_data_format(file_path, self.config.contaminants, to_csv=False)  # returns rows with columns: timestep, node, chlorine_concentration, arsenic_concentration
        
        dfs = []
        
        if self.config.aggregate_method is None:
            for node in nodes:
                df_node = get_data_for_one_node(df_all, node, to_csv=False)
                dfs.append(df_node)
        
        else:
            df = aggregate_data_for_several_nodes(df_all, nodes, method=self.config.aggregate_method, to_csv=False)
            dfs.append(df)
        
        return dfs

    def load_datasets(self):
        """
        Loads the datasets for the experiment based on the examples files and contamined files specified in the configuration.
        Return lists of dataframes for each contaminated and example (clean) file. 

        Returns:
        example_dfs: list of dataframes for each example file (clean data)
        contamined_dfs: list of dataframes for each contaminated file
        """ 
        example_dfs = []
        if self.config.example_files is not None:
            for fp in self.config.example_files:
                example_dfs.extend(self.load_and_filter(fp, self.config.nodes))

        contaminated_dfs = []
        for fp in self.config.contaminated_files:
            contaminated_dfs.extend(self.load_and_filter(fp, self.config.nodes))

        return example_dfs, contaminated_dfs

    @abstractmethod
    def get_results(self):
        """ 
        Returns the results of the experiment as a dictionary containing the true labels and predicted labels for each node. 
    
        Returns:
          y_true: true labels 
          y_pred: predicted labels
         """
        pass