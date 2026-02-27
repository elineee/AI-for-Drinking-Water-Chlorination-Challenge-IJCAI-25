from experiment import ExperimentRunner
from experiment_config import ExperimentConfig, ModelName
from evaluation import Evaluation, Metrics

import pickle

if __name__ == "__main__":
    CLEAN_FILES = [".\\data\\data_arsenic\\scada_data_no_conta_53_days_1.csv", ".\\data\\data_arsenic\\scada_data_no_conta_53_days_2.csv", ".\\data\\data_arsenic\\scada_data_no_conta_53_days_3.csv", ".\\data\\data_arsenic\\scada_data_no_conta_53_days_4.csv"]
    CONTAMINATED_FILES = [".\\data\\data_arsenic\\scada_data_conta_node_13.csv"]

    configs = [
    # ExperimentConfig(
    #     config_name="AUTOENCODER",
    #     example_files=CLEAN_FILES,
    #     contaminated_files=CONTAMINATED_FILES,
    #     nodes=[11, 12, 21, 22, 31, 32],
    #     window_size=50,
    #     model_name=ModelName.AUTOENCODER,
    #     model_params={}
    # ),
    # ExperimentConfig(
    #     config_name="LOF_20_neighbors",
    #     example_files=CLEAN_FILES,
    #     contaminated_files=CONTAMINATED_FILES,
    #     nodes=[11, 12, 13, 21, 22, 31, 32],
    #     window_size=20,
    #     model_name=ModelName.LOF,
    #     model_params={"n_neighbors": 20, "contamination": 0.1}
    # ),
    # ExperimentConfig(
    #     config_name="IsolationForest",
    #     contaminated_files=CONTAMINATED_FILES,
    #     nodes=[22, 31],
    #     window_size=0,
    #     model_name=ModelName.ISOLATION_FOREST,
    #     model_params={"contamination": 0.1}
    # ),
    # ExperimentConfig(
    #     config_name="OneClassSVM",
    #     contaminated_files=CONTAMINATED_FILES,
    #     example_files=CLEAN_FILES,
    #     nodes=[11, 12, 21, 22, 31, 32],
    #     window_size=10,
    #     model_name=ModelName.ONE_CLASS_SVM,
    #     model_params={"gamma": "scale", "nu": 0.05, "kernel": "rbf"},
    # ),
    
        ExperimentConfig(
            config_name="OneClassSVM_ALARM",
            contaminated_files=CONTAMINATED_FILES,
            example_files=CLEAN_FILES,
            nodes=[11, 12, 13, 21, 22, 31, 32],
            window_size=10,
            model_name=ModelName.ONE_CLASS_SVM_ALARM,
            model_params={"gamma": "scale", "nu": 0.05, "kernel": "rbf"},
        ),
        # ExperimentConfig(
        #             config_name="SVR",
        #             contaminated_files=CONTAMINATED_FILES,
        #             example_files=CLEAN_FILES,
        #             nodes=[32],
        #             window_size=48, # 48 correspond à 48*30 min donc 1 jour
        #             model_name=ModelName.SVR,
        #             model_params={"gamma": "scale", "epsilon": 0.01, "kernel": "rbf", "C": 10},
        #         ),
        
        # ExperimentConfig(
        #             config_name="SVR_ALARM",
        #             contaminated_files=CONTAMINATED_FILES,
        #             example_files=CLEAN_FILES,
        #             nodes=[13, 22, 31, 32],
        #             window_size=48, # 48 correspond à 48*30 min donc 1 jour
        #             model_name=ModelName.SVR_ALARM,
        #             model_params={"gamma": 0.01, "epsilon": 0.01, "kernel": "rbf", "C": 1},
        #         ),
        
    #     ExperimentConfig(
    #         config_name="LSTM_AUTOENCODER",
    #         example_files=CLEAN_FILES,
    #         contaminated_files=CONTAMINATED_FILES,
    #         nodes=[13],
    #         window_size=10,
    #         model_name=ModelName.LSTM_AUTOENCODER,
    #         model_params={}
    # ),
    ]
    all_results = []


    for cfg in configs:
        runner = ExperimentRunner(cfg)
        res = runner.run()
        all_results.append(res)

    pickle.dump(all_results, open("all_results.pkl", "wb"))

    evaluation = Evaluation()
    evaluation_results = evaluation.evaluate("all_results.pkl")
    print(evaluation_results)

    print("Evaluation results:")

    evaluation.plot_confusion_matrices("OneClassSVM_ALARM", evaluation_results)
    # evaluation.plot_results_nodes_within_configurations(Metrics.ACCURACY, evaluation_results)
    # evaluation.plot_mean_configuration(Metrics.ACCURACY, evaluation_results)

