from experiment_config import ExperimentConfig, ModelName
from models.LOF import LOFModel
from models.LOF_alarm import LOFAlarmModel
from models.isolation_forest import IsolationForestModel
from models.isolation_forest_alarm import IsolationForestAlarmModel
from models.one_class_SVM import OneClassSVMModel
from models.one_class_SVM_alarm import OneClassSVMAlarmModel
from models.SVR import SVRModel
from models.SVR_alarm import SVRAlarmModel
from models.autoencoder import AutoencoderModel
from models.autoencoder_alarm import AutoencoderAlarmModel 
from models.LSTM_AE import LSTMAutoencoderModel
from models.LSTM_AE_alarm import LSTMAutoencoderAlarmModel
from models.VAE import VAEModel
from models.LSTM_VAE import LSTMVAEModel
from models.GAN import GANModel
from models.cusum import CusumModel
from models.cusum_alarm import CusumAlarmModel
from models.cnn import CNNModel
from models.cnn_windows import CNNWindowsModel


AVAILABLE_MODELS = {
    ModelName.LOF: LOFModel,
    ModelName.LOF_ALARM: LOFAlarmModel,
    ModelName.ISOLATION_FOREST: IsolationForestModel,
    ModelName.ISOLATION_FOREST_ALARM : IsolationForestAlarmModel, 
    ModelName.ONE_CLASS_SVM: OneClassSVMModel,
    ModelName.ONE_CLASS_SVM_ALARM: OneClassSVMAlarmModel,
    ModelName.SVR: SVRModel,
    ModelName.SVR_ALARM: SVRAlarmModel,
    ModelName.AUTOENCODER: AutoencoderModel,
    ModelName.AUTOENCODER_ALARM : AutoencoderAlarmModel,
    ModelName.LSTM_AUTOENCODER: LSTMAutoencoderModel,
    ModelName.LSTM_AUTOENCODER_ALARM: LSTMAutoencoderAlarmModel,
    ModelName.VAE : VAEModel, 
    ModelName.LSTM_VAE : LSTMVAEModel, 
    ModelName.GAN: GANModel,
    ModelName.CUSUM: CusumModel,
    ModelName.CUSUM_ALARM: CusumAlarmModel,
    ModelName.CNN: CNNModel,
    ModelName.CNN_windows: CNNWindowsModel
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
