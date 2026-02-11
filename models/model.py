""" General abstract model class for the project. All models should inherit from this class and implement the `fit` and `predict` methods. """

from abc import ABC, abstractmethod

from experiment_config import ExperimentConfig

class AnomalyModel(ABC):
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
    @abstractmethod
    def get_results(self):
        pass

    
    
    