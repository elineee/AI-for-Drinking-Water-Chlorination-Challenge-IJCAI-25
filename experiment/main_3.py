from experiment import ExperimentRunner
from experiment_config import ContaminationType, ExperimentConfig, ModelName
from evaluation import Evaluation, Metrics

import pickle

if __name__ == "__main__":
    
    nodes = ["dist225", "dist485", "dist631", "dist1332", "dist1459", "dist1702", "dist1975"]
    
    for node in nodes: 
        print(f"Running experiments for node {node}...")
        
        CLEAN_FILES = [".\\data\\data_compet\\scada_data_clean.csv", ".\\data\\data_compet\\scada_data_clean_2.csv"]
        
        CONTAMINATED_FILES1 = [".\\data\\data_compet\\scada_data_clean.csv", ".\\data\\data_compet\\scada_data_clean_2.csv", ".\\data\\data_compet\\scada_data_clean_3.csv", ".\\data\\data_compet\\scada_data_conta_node_423_train_1.csv", ".\\data\\data_compet\\scada_data_conta_node_423_train_2.csv", ".\\data\\data_compet\\scada_data_conta_node_423_train_3.csv", ".\\data\\data_compet\\scada_data_conta_node_423_train_4.csv", ".\\data\\data_compet\\scada_data_conta_node_423_train_5.csv", ".\\data\\data_compet\\scada_data_conta_node_423_train_6.csv", ".\\data\\data_compet\\scada_data_conta_node_423_train_7.csv", ".\\data\\data_compet\\scada_data_conta_node_423_test_1.csv"]
        CONTAMINATED_FILES2 = [".\\data\\data_compet\\scada_data_conta_node_423_train_1.csv", ".\\data\\data_compet\\scada_data_conta_node_423_test_2.csv"]
        CONTAMINATED_FILES3 = [".\\data\\data_compet\\scada_data_conta_node_423_train_1.csv", ".\\data\\data_compet\\scada_data_conta_node_423_test_3.csv"]

        configs = [

        # essayer avec de plus grandes fenêtres
        ExperimentConfig(
                        config_name="CNN",
                        contaminated_files=CONTAMINATED_FILES1,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=500, 
                        model_name=ModelName.CNN,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        ), 
        
        ExperimentConfig(
                        config_name="CNN2",
                        contaminated_files=CONTAMINATED_FILES2,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=500, 
                        model_name=ModelName.CNN,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        ), 
        
        ExperimentConfig(
                        config_name="CNN3",
                        contaminated_files=CONTAMINATED_FILES3,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=500, 
                        model_name=ModelName.CNN,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        )
        

        
        
        ]
        all_results = []


        for cfg in configs:
            runner = ExperimentRunner(cfg)
            res = runner.run()
            all_results.append(res)
            print(all_results)

        pickle.dump(all_results, open(f"all_results_{node}_CNN.pkl", "wb"))

        evaluation = Evaluation()
        evaluation_results = evaluation.evaluate(f"all_results_{node}_CNN.pkl")
        print(evaluation_results)

        print("Evaluation results:")

        evaluation.plot_confusion_matrices("CNN", evaluation_results)
        # evaluation.plot_results_nodes_within_configurations(Metrics.ACCURACY, evaluation_results)
        # evaluation.plot_mean_configuration(Metrics.ACCURACY, evaluation_results)
        
    
    for node in nodes: 
        print(f"Running experiments for node {node}...")
        
        CLEAN_FILES = [".\\data\\data_compet\\scada_data_clean.csv", ".\\data\\data_compet\\scada_data_clean_2.csv", ".\\data\\data_compet\\scada_data_clean_3.csv"]

        CONTAMINATED_FILES1 = [".\\data\\data_compet\\scada_data_conta_node_423_test_1.csv"]
        CONTAMINATED_FILES2 = [".\\data\\data_compet\\scada_data_conta_node_423_test_2.csv"]
        CONTAMINATED_FILES3 = [".\\data\\data_compet\\scada_data_conta_node_423_test_3.csv"]
        
        
        configs = [

        # essayer avec de plus grandes fenêtres
        ExperimentConfig(
                        config_name="SVR_ALARM_1",
                        contaminated_files=CONTAMINATED_FILES1,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=288, 
                        model_name=ModelName.CNN,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        ), 
        
        ExperimentConfig(
                        config_name="SVR_ALARM_2",
                        contaminated_files=CONTAMINATED_FILES2,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=288, 
                        model_name=ModelName.CNN,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        ), 
        
        ExperimentConfig(
                        config_name="SVR_ALARM_3",
                        contaminated_files=CONTAMINATED_FILES3,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=288, 
                        model_name=ModelName.CNN,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        )
        

        
        
        ]
        all_results = []


        for cfg in configs:
            runner = ExperimentRunner(cfg)
            res = runner.run()
            all_results.append(res)
            print(all_results)

        pickle.dump(all_results, open(f"all_results_{node}_SVR.pkl", "wb"))

        evaluation = Evaluation()
        evaluation_results = evaluation.evaluate(f"all_results_{node}_SVR.pkl")
        print(evaluation_results)

        print("Evaluation results:")

        evaluation.plot_confusion_matrices("SVR", evaluation_results)
        # evaluation.plot_results_nodes_within_configurations(Metrics.ACCURACY, evaluation_results)
        # evaluation.plot_mean_configuration(Metrics.ACCURACY, evaluation_results)

