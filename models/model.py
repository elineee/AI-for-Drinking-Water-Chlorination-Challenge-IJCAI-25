""" General abstract model class for the project. All models should inherit from this class and implement the `fit` and `predict` methods. """

from abc import ABC, abstractmethod

from experiment_config import ExperimentConfig

""" Abstract class for anomaly detection models. All specific models (e.g., LOF, Isolation Forest) should inherit from this class and implement the `get_results` method, which should return the true labels and predicted labels for the experiment."""
class AnomalyModel(ABC):
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
    @abstractmethod
    def get_results(self):
        """ Return y_true and y_pred for the experiment, where y_true are the true labels and y_pred are the predicted labels by the model. """
        pass

    
    
    