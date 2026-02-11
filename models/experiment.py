from typing import Dict
from LOF import LOFModel
from experiment_config import ExperimentConfig
from isolation_forest import IsolationForestModel


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(self) -> Dict:
        model_name = self.config.model_name
        
        if model_name == "LOF":
            model = LOFModel(self.config)
        elif model_name == "isolation_forest":
            model = IsolationForestModel(self.config)
        
        results = model.get_results()
        return {self.config.config_name: results}