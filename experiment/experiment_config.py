from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum

class ContaminationType(Enum):
    """Enumeration of contamination types"""
    ARSENIC = "arsenic"
    CHLORINE = "chlorine"

class ModelName(Enum):
    """Enumeration of model names"""
    LOF = "LOF"
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_SVM"

@dataclass
class ExperimentConfig:
    """ Configuration class for experiments. Contains all the parameters needed to run an experiment, including:
    - model_name: a string with the ModelName to use 
    - config_name: a name for the configuration (used for storing results)
    - window_size: the size of the sliding window for time series (30 by default, 0 means no window) 
    - disinfectant: the type of disinfectant used in the water (chlorine by default)
    - contaminated_files: a list of file paths to the contaminated data files (used for training and testing if no example files)
    - example_files: a list of file paths to the example data files (used for training if need of examples of clean data)
    - nodes: a list of node numbers on which model will be trained/tested
    - contaminants: a list of ContaminationType to specify which contaminants to use (arsenic by default)
    - model_params: a dictionary of parameters to pass to the model (e.g., n_neighbors for LOF)
    - aggregate_method: whether to train models on each node separately or on aggregated nodes (e.g., mean/sum) #TODO
    """

    model_name: ModelName
    config_name: str
    window_size: int = 30
    disinfectant: ContaminationType = ContaminationType.CHLORINE
    contaminated_files: List[str] = field(default_factory=list) 
    example_files: Optional[List[str]] = None
    nodes: List[str] = field(default_factory=list)       
    contaminants: List[ContaminationType] = field(default_factory=lambda: [ContaminationType.ARSENIC])                  
    model_params: Dict = field(default_factory=dict)
    aggregate_method: Optional[str] = None 
    