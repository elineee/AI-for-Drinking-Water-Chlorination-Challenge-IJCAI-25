from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum

""" Contaminations types accepted """
class ContaminationType(Enum):
    ARSENIC = "arsenic"
    CHLORINE = "chlorine"

""" Configuration class for experiments. Contains all the parameters needed to run an experiment, including:
- model_name: the name of the model to use (e.g., "LOF", "isolation_forest")
- config_name: a name for the configuration (used for storing results)
- window_size: the size of the sliding window for time series
- desinfectant: the type of desinfectant used in the water (probably always chlorine)
- contaminated_files: a list of file paths to the contaminated data files
- nodes: a list of node numbers on which model will be trained/tested
- contaminants: a list of ContaminationType to specify which contaminants to use
- model_params: a dictionary of parameters to pass to the model (e.g., n_neighbors for LOF)
- example_files: a list of file paths to the example data files (used for training if need of examples of clean data)
- aggregate_method: whether to train models on each node separately or on aggregated nodes (e.g., mean/sum)
"""
@dataclass
class ExperimentConfig:
    model_name: str
    config_name: str
    window_size: int = 30
    desinfectant: ContaminationType = ContaminationType.CHLORINE
    contaminated_files: List[str] = field(default_factory=list) # data files (used for train and test if no example file)
    nodes: List[str] = field(default_factory=list)          # list of nodes on which models are trained/tested
    contaminants: List[ContaminationType] = field(default_factory=lambda: [ContaminationType.ARSENIC])                  
    model_params: Optional[Dict] = None
    example_files: Optional[List[str]] = field(default_factory=list) # example files of normal behavior if need of examples of clean data
    aggregate_method: Optional[str] = None    # whether models are trained on each node separately or on aggregated nodes (e.g., mean/sum)
    

    