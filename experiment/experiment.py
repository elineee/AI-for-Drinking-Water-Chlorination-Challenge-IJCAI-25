from experiment_config import ExperimentConfig, ModelName
from models.LOF import LOFModel
from models.isolation_forest import IsolationForestModel
from models.one_class_SVM import OneClassSVMModel

AVAILABLE_MODELS = {
    ModelName.LOF: LOFModel,
    ModelName.ISOLATION_FOREST: IsolationForestModel,
    ModelName.ONE_CLASS_SVM: OneClassSVMModel
}

class ExperimentRunner:
    """ 
    Class to run an experiment based on a given configuration. It initializes the appropriate model based on the configuration and runs it to get the results.
    The model is selected from the AVAILABLE_MODELS dictionary using the model_name specified in the configuration. 
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def run(self):
        model_type = AVAILABLE_MODELS[self.config.model_name]
        model = model_type(self.config)

        results = model.get_results()
        return {self.config.config_name: results}
