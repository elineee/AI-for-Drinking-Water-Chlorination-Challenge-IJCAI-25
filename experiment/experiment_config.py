from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum

class ContaminationType(Enum):
    """Enumeration of contamination types"""
    ARSENIC = "arsenic"
    CHLORINE = "chlorine"
    PATHOGEN = "pathogen"

class ModelName(Enum):
    """Enumeration of model names"""
    LOF = "LOF"
    LOF_ALARM = "LOF_alarm"
    ISOLATION_FOREST = "isolation_forest"
    ISOLATION_FOREST_ALARM = "isolation_forest_alarm"
    ONE_CLASS_SVM = "one_class_SVM"
    ONE_CLASS_SVM_ALARM = "one_class_SVM_alarm"
    SVR = "SVR"
    SVR_ALARM = "SVR_alarm"
    AUTOENCODER = "Autoencoder"
    AUTOENCODER_ALARM = "Autoencoder_alarm"
    LSTM_AUTOENCODER = "LSTM_Autoencoder"
    LSTM_AUTOENCODER_ALARM = "LSTM_Autoencoder_alarm"
    VAE = "VAE"
    VAE_ALARM = "VAE_ALARM"
    LSTM_VAE = "LSTM_VAE"
    LSTM_VAE_ALARM = "LSTM_VAE_ALARM"
    GAN = "GAN"
    CUSUM = "CUSUM"
    CUSUM_ALARM = "CUSUM_alarm"
    CNN = "CNN"
    CNN_UNIVARIATE = "CNN_univariate"
    CNN_WINDOWS = "CNN_windows"
    CNN_VAE  = "CNN_VAE"

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
    - aggregate_method: whether to train models on each node separately or on aggregated nodes (e.g., mean/sum) or on all nodes alltogether
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
    