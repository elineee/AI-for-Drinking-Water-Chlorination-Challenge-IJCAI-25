from abc import ABC, abstractmethod
from experiment_config import ExperimentConfig

class AnomalyModel(ABC):
    """ 
    Abstract class for anomaly detection models. 
    All specific models should inherit from this class and implement the `get_results` method.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        
    @abstractmethod
    def get_results(self):
        """ 
        Returns the results of the experiment 

        Returns:
          y_true: true labels 
          y_pred: predicted labels
         """
        pass

    
    
    