from experiment import ExperimentRunner
from experiment_config import ContaminationType, ExperimentConfig, ModelName
from evaluation import Evaluation, Metrics

import pickle

if __name__ == "__main__":
    nodes = ["dist356", "dist399", "dist485", "dist606", "dist1028", "dist1332", "dist1363", "dist1459", "dist1464", "dist1915"]
    
    for node in nodes: 
        print(f"Running experiments for node {node}...")
        
        CLEAN_FILES = [".\\data\\data_small_chlorine\\scada_data_clean_1.csv", ".\\data\\data_small_chlorine\\scada_data_clean_3.csv", ".\\data\\data_small_chlorine\\scada_data_clean_4.csv"]
        
        CONTAMINATED_FILES2 = [".\\data\\data_small_chlorine\\scada_data_test_2.csv"]
        CONTAMINATED_FILES3 = [".\\data\\data_small_chlorine\\scada_data_test_5.csv"]
        CONTAMINATED_FILES4 = [".\\data\\data_small_chlorine\\scada_data_test_6.csv"]

        configs = [

        
        ExperimentConfig(
                        config_name="LOF2",
                        contaminated_files=CONTAMINATED_FILES2,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=20, 
                        model_name=ModelName.LOF_ALARM,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        ), 
        
        ExperimentConfig(
                        config_name="LOF3",
                        contaminated_files=CONTAMINATED_FILES3,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=20, 
                        model_name=ModelName.LOF_ALARM,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        ),
        
        ExperimentConfig(
                        config_name="LOF4",
                        contaminated_files=CONTAMINATED_FILES4,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=20, 
                        model_name=ModelName.LOF_ALARM,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        ),    
        
        ]
        all_results = []


        for cfg in configs:
            runner = ExperimentRunner(cfg)
            res = runner.run()
            all_results.append(res)
            # print(all_results)

        pickle.dump(all_results, open(f"all_results_{node}_LOF_ALARM.pkl", "wb"))

        evaluation = Evaluation()
        evaluation_results = evaluation.evaluate(f"all_results_{node}_LOF_ALARM.pkl")
        print(evaluation_results)

        print("Evaluation results:")

        #evaluation.plot_confusion_matrices("LOF_ALARM", evaluation_results)
        # evaluation.plot_results_nodes_within_configurations(Metrics.ACCURACY, evaluation_results)
        # evaluation.plot_mean_configuration(Metrics.ACCURACY, evaluation_results)
    
    
    for node in nodes: 
        print(f"Running experiments for node {node}...")
        
        CLEAN_FILES = [".\\data\\data_small_chlorine\\scada_data_clean_1.csv", ".\\data\\data_small_chlorine\\scada_data_clean_3.csv", ".\\data\\data_small_chlorine\\scada_data_clean_4.csv"]
        
        CONTAMINATED_FILES2 = [".\\data\\data_small_chlorine\\scada_data_test_2.csv"]
        CONTAMINATED_FILES3 = [".\\data\\data_small_chlorine\\scada_data_test_5.csv"]
        CONTAMINATED_FILES4 = [".\\data\\data_small_chlorine\\scada_data_test_6.csv"]

        configs = [
        
        ExperimentConfig(
                        config_name="OneClassSVM2",
                        contaminated_files=CONTAMINATED_FILES2,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=20, 
                        model_name=ModelName.ONE_CLASS_SVM_ALARM,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        ), 
        
        ExperimentConfig(
                        config_name="OneClassSVM3",
                        contaminated_files=CONTAMINATED_FILES3,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=20, 
                        model_name=ModelName.ONE_CLASS_SVM_ALARM,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        ),
    
        ExperimentConfig(
                        config_name="OneClassSVM4",
                        contaminated_files=CONTAMINATED_FILES4,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=20, 
                        model_name=ModelName.ONE_CLASS_SVM_ALARM,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        )
    
        
        ]
        all_results = []


        for cfg in configs:
            runner = ExperimentRunner(cfg)
            res = runner.run()
            all_results.append(res)
            # print(all_results)

        pickle.dump(all_results, open(f"all_results_{node}_ONE_CLASS_SVM_ALARM.pkl", "wb"))

        evaluation = Evaluation()
        evaluation_results = evaluation.evaluate(f"all_results_{node}_ONE_CLASS_SVM_ALARM.pkl")
        print(evaluation_results)

        print("Evaluation results:")

        #evaluation.plot_confusion_matrices("ONE_CLASS_SVM_ALARM", evaluation_results)
        # evaluation.plot_results_nodes_within_configurations(Metrics.ACCURACY, evaluation_results)
        # evaluation.plot_mean_configuration(Metrics.ACCURACY, evaluation_results)
    


    for node in nodes: 
        print(f"Running experiments for node {node}...")
        
        CLEAN_FILES = [".\\data\\data_small_chlorine\\scada_data_clean_1.csv", ".\\data\\data_small_chlorine\\scada_data_clean_3.csv", ".\\data\\data_small_chlorine\\scada_data_clean_4.csv"]
        
        CONTAMINATED_FILES2 = [".\\data\\data_small_chlorine\\scada_data_test_2.csv"]
        CONTAMINATED_FILES3 = [".\\data\\data_small_chlorine\\scada_data_test_5.csv"]
        CONTAMINATED_FILES4 = [".\\data\\data_small_chlorine\\scada_data_test_6.csv"]

        configs = [
        
        ExperimentConfig(
                        config_name="IsoForest2",
                        contaminated_files=CONTAMINATED_FILES2,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=50, 
                        model_name=ModelName.ISOLATION_FOREST_ALARM,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        ), 
        
        ExperimentConfig(
                        config_name="IsoForest3",
                        contaminated_files=CONTAMINATED_FILES3,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=50, 
                        model_name=ModelName.ISOLATION_FOREST_ALARM,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        ),
        
        ExperimentConfig(
                        config_name="IsoForest4",
                        contaminated_files=CONTAMINATED_FILES4,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=50, 
                        model_name=ModelName.ISOLATION_FOREST_ALARM,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        )
    
        
        ]
        all_results = []


        for cfg in configs:
            runner = ExperimentRunner(cfg)
            res = runner.run()
            all_results.append(res)
            # print(all_results)

        pickle.dump(all_results, open(f"all_results_{node}_ISO_FOREST_ALARM.pkl", "wb"))

        evaluation = Evaluation()
        evaluation_results = evaluation.evaluate(f"all_results_{node}_ISO_FOREST_ALARM.pkl")
        print(evaluation_results)

        print("Evaluation results:")

        #evaluation.plot_confusion_matrices("ISO_FOREST_ALARM", evaluation_results)
        # evaluation.plot_results_nodes_within_configurations(Metrics.ACCURACY, evaluation_results)
        # evaluation.plot_mean_configuration(Metrics.ACCURACY, evaluation_results)
    
    
    for node in nodes: 
        print(f"Running experiments for node {node}...")
        
        CLEAN_FILES = [".\\data\\data_small_chlorine\\scada_data_clean_1.csv", ".\\data\\data_small_chlorine\\scada_data_clean_3.csv", ".\\data\\data_small_chlorine\\scada_data_clean_4.csv"]
        
        CONTAMINATED_FILES2 = [".\\data\\data_small_chlorine\\scada_data_test_2.csv"]
        CONTAMINATED_FILES3 = [".\\data\\data_small_chlorine\\scada_data_test_5.csv"]
        CONTAMINATED_FILES4 = [".\\data\\data_small_chlorine\\scada_data_test_6.csv"]

        configs = [
        
        ExperimentConfig(
                        config_name="SVR2",
                        contaminated_files=CONTAMINATED_FILES2,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=48, 
                        model_name=ModelName.SVR_ALARM,
                        model_params={"gamma": "scale", "epsilon": 0.01, "kernel": "rbf", "C": 10},
                        contaminants=[ContaminationType.PATHOGEN]
    
        ), 
        
        ExperimentConfig(
                        config_name="SVR3",
                        contaminated_files=CONTAMINATED_FILES3,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=48, 
                        model_name=ModelName.SVR_ALARM,
                        model_params={"gamma": "scale", "epsilon": 0.01, "kernel": "rbf", "C": 10},
                        contaminants=[ContaminationType.PATHOGEN]
    
        ),
    
        ExperimentConfig(
                        config_name="SVR4",
                        contaminated_files=CONTAMINATED_FILES4,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=48, 
                        model_name=ModelName.SVR_ALARM,
                        model_params={"gamma": "scale", "epsilon": 0.01, "kernel": "rbf", "C": 10},
                        contaminants=[ContaminationType.PATHOGEN]
    
        )
    
        
        ]
        all_results = []


        for cfg in configs:
            runner = ExperimentRunner(cfg)
            res = runner.run()
            all_results.append(res)
            # print(all_results)

        pickle.dump(all_results, open(f"all_results_{node}_SVR_ALARM.pkl", "wb"))

        evaluation = Evaluation()
        evaluation_results = evaluation.evaluate(f"all_results_{node}_SVR_ALARM.pkl")
        print(evaluation_results)

        print("Evaluation results:")

        #evaluation.plot_confusion_matrices("SVR_ALARM", evaluation_results)
        # evaluation.plot_results_nodes_within_configurations(Metrics.ACCURACY, evaluation_results)
        # evaluation.plot_mean_configuration(Metrics.ACCURACY, evaluation_results)
    
    for node in nodes: 
        print(f"Running experiments for node {node}...")
        
        CLEAN_FILES = [".\\data\\data_small_chlorine\\scada_data_clean_1.csv", ".\\data\\data_small_chlorine\\scada_data_clean_3.csv", ".\\data\\data_small_chlorine\\scada_data_clean_4.csv", ".\\data\\data_small_chlorine\\scada_data_clean_7.csv", ".\\data\\data_small_chlorine\\scada_data_clean_9.csv"]

        CONTAMINATED_FILES2 = [".\\data\\data_small_chlorine\\scada_data_train_1_1.csv", ".\\data\\data_small_chlorine\\scada_data_train_1_2.csv", ".\\data\\data_small_chlorine\\scada_data_train_1_3.csv", ".\\data\\data_small_chlorine\\scada_data_train_3_1.csv", ".\\data\\data_small_chlorine\\scada_data_train_3_2.csv", ".\\data\\data_small_chlorine\\scada_data_train_3_3.csv", ".\\data\\data_small_chlorine\\scada_data_train_4_1.csv", ".\\data\\data_small_chlorine\\scada_data_train_4_2.csv", ".\\data\\data_small_chlorine\\scada_data_train_4_3.csv", ".\\data\\data_small_chlorine\\scada_data_test_2.csv"]
        CONTAMINATED_FILES3 = [".\\data\\data_small_chlorine\\scada_data_train_1_1.csv", ".\\data\\data_small_chlorine\\scada_data_test_5.csv"]
        CONTAMINATED_FILES4 = [".\\data\\data_small_chlorine\\scada_data_train_1_1.csv", ".\\data\\data_small_chlorine\\scada_data_test_6.csv"]
    
        configs = [

        ExperimentConfig(
                        config_name="CNN2",
                        contaminated_files=CONTAMINATED_FILES2,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=150, 
                        model_name=ModelName.CNN,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]

        ), 
        
        ExperimentConfig(
                        config_name="CNN3",
                        contaminated_files=CONTAMINATED_FILES3,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=150, 
                        model_name=ModelName.CNN,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        ),
        ExperimentConfig(
                        config_name="CNN4",
                        contaminated_files=CONTAMINATED_FILES4,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=150, 
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
            #print(all_results)

        pickle.dump(all_results, open(f"all_results_{node}_CNN.pkl", "wb"))

        evaluation = Evaluation()
        evaluation_results = evaluation.evaluate(f"all_results_{node}_CNN.pkl")
        print(evaluation_results)

        print("Evaluation results:")

        #evaluation.plot_confusion_matrices("CNN", evaluation_results)
        # evaluation.plot_results_nodes_within_configurations(Metrics.ACCURACY, evaluation_results)
        # evaluation.plot_mean_configuration(Metrics.ACCURACY, evaluation_results)
    
    for node in nodes: 
        print(f"Running experiments for node {node}...")
        
        CLEAN_FILES = [".\\data\\data_small_chlorine\\scada_data_clean_1.csv", ".\\data\\data_small_chlorine\\scada_data_clean_3.csv", ".\\data\\data_small_chlorine\\scada_data_clean_4.csv", ".\\data\\data_small_chlorine\\scada_data_clean_7.csv", ".\\data\\data_small_chlorine\\scada_data_clean_9.csv"]

        CONTAMINATED_FILES2 = [".\\data\\data_small_chlorine\\scada_data_train_1_1.csv", ".\\data\\data_small_chlorine\\scada_data_train_1_2.csv", ".\\data\\data_small_chlorine\\scada_data_train_1_3.csv", ".\\data\\data_small_chlorine\\scada_data_train_3_1.csv", ".\\data\\data_small_chlorine\\scada_data_train_3_2.csv", ".\\data\\data_small_chlorine\\scada_data_train_3_3.csv", ".\\data\\data_small_chlorine\\scada_data_train_4_1.csv", ".\\data\\data_small_chlorine\\scada_data_train_4_2.csv", ".\\data\\data_small_chlorine\\scada_data_train_4_3.csv", ".\\data\\data_small_chlorine\\scada_data_test_2.csv"]
        CONTAMINATED_FILES3 = [".\\data\\data_small_chlorine\\scada_data_train_1_1.csv", ".\\data\\data_small_chlorine\\scada_data_test_5.csv"]
        CONTAMINATED_FILES4 = [".\\data\\data_small_chlorine\\scada_data_train_1_1.csv", ".\\data\\data_small_chlorine\\scada_data_test_6.csv"]
    
    
        configs = [
            
            ExperimentConfig(
                    config_name="CNN_multi_nodes2",
                    contaminated_files=CONTAMINATED_FILES2,
                    example_files=CLEAN_FILES,
                    nodes=["dist64", "dist420"],
                    window_size=150,
                    model_name=ModelName.CNN_MULTI_NODES,
                    model_params={},
                    contaminants=[ContaminationType.PATHOGEN]
            ),
            
            ExperimentConfig(
                    config_name="CNN_multi_nodes3",
                    contaminated_files=CONTAMINATED_FILES3,
                    example_files=CLEAN_FILES,
                    nodes=["dist64", "dist420"],
                    window_size=150,
                    model_name=ModelName.CNN_MULTI_NODES,
                    model_params={},
                    contaminants=[ContaminationType.PATHOGEN]
            ),
            ExperimentConfig(
                    config_name="CNN_multi_nodes4",
                    contaminated_files=CONTAMINATED_FILES4,
                    example_files=CLEAN_FILES,
                    nodes=["dist64", "dist420"],
                    window_size=150,
                    model_name=ModelName.CNN_MULTI_NODES,
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

        pickle.dump(all_results, open(f"all_results_CNN_MULTI_NODES.pkl", "wb"))

        evaluation = Evaluation()
        evaluation_results = evaluation.evaluate(f"all_results_CNN_MULTI_NODES.pkl")
        print(evaluation_results)

        print("Evaluation results:")

        #evaluation.plot_confusion_matrices("LOF_ALARM", evaluation_results)
        # evaluation.plot_results_nodes_within_configurations(Metrics.ACCURACY, evaluation_results)
        # evaluation.plot_mean_configuration(Metrics.ACCURACY, evaluation_results)
    


    for node in nodes: 
        print(f"Running experiments for node {node}...")
        
        CLEAN_FILES = [".\\data\\data_small_chlorine\\scada_data_clean_1.csv", ".\\data\\data_small_chlorine\\scada_data_clean_3.csv", ".\\data\\data_small_chlorine\\scada_data_clean_4.csv", ".\\data\\data_small_chlorine\\scada_data_clean_7.csv", ".\\data\\data_small_chlorine\\scada_data_clean_9.csv"]

        CONTAMINATED_FILES2 = [".\\data\\data_small_chlorine\\scada_data_train_1_1.csv", ".\\data\\data_small_chlorine\\scada_data_train_1_2.csv", ".\\data\\data_small_chlorine\\scada_data_train_1_3.csv", ".\\data\\data_small_chlorine\\scada_data_train_3_1.csv", ".\\data\\data_small_chlorine\\scada_data_train_3_2.csv", ".\\data\\data_small_chlorine\\scada_data_train_3_3.csv", ".\\data\\data_small_chlorine\\scada_data_train_4_1.csv", ".\\data\\data_small_chlorine\\scada_data_train_4_2.csv", ".\\data\\data_small_chlorine\\scada_data_train_4_3.csv", ".\\data\\data_small_chlorine\\scada_data_test_2.csv"]
        CONTAMINATED_FILES3 = [".\\data\\data_small_chlorine\\scada_data_train_1_1.csv", ".\\data\\data_small_chlorine\\scada_data_test_5.csv"]
        CONTAMINATED_FILES4 = [".\\data\\data_small_chlorine\\scada_data_train_1_1.csv", ".\\data\\data_small_chlorine\\scada_data_test_6.csv"]
    
        configs = [
        
        ExperimentConfig(
                        config_name="CNN_VAE2",
                        contaminated_files=CONTAMINATED_FILES2,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=100, 
                        model_name=ModelName.CNN_VAE,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        ), 
        
        ExperimentConfig(
                        config_name="CNN_VAE3",
                        contaminated_files=CONTAMINATED_FILES3,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=100, 
                        model_name=ModelName.CNN_VAE,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        ),
    
        ExperimentConfig(
                        config_name="CNN_VAE4",
                        contaminated_files=CONTAMINATED_FILES4,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=100, 
                        model_name=ModelName.CNN_VAE,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        )
        

        
        
        ]
        all_results = []


        for cfg in configs:
            runner = ExperimentRunner(cfg)
            res = runner.run()
            all_results.append(res)
            # print(all_results)

        pickle.dump(all_results, open(f"all_results_{node}_CNN_VAE.pkl", "wb"))

        evaluation = Evaluation()
        evaluation_results = evaluation.evaluate(f"all_results_{node}_CNN_VAE.pkl")
        print(evaluation_results)

        print("Evaluation results:")

        evaluation.plot_confusion_matrices("CNN", evaluation_results)
        evaluation.plot_results_nodes_within_configurations(Metrics.ACCURACY, evaluation_results)
        evaluation.plot_mean_configuration(Metrics.ACCURACY, evaluation_results)
        
    
    for node in nodes: 
        print(f"Running experiments for node {node}...")

        CLEAN_FILES = [".\\data\\data_small_chlorine\\scada_data_clean_1.csv", ".\\data\\data_small_chlorine\\scada_data_clean_3.csv", ".\\data\\data_small_chlorine\\scada_data_clean_4.csv", ".\\data\\data_small_chlorine\\scada_data_clean_7.csv", ".\\data\\data_small_chlorine\\scada_data_clean_9.csv"]

        CONTAMINATED_FILES2 = [".\\data\\data_small_chlorine\\scada_data_train_1_1.csv", ".\\data\\data_small_chlorine\\scada_data_train_1_2.csv", ".\\data\\data_small_chlorine\\scada_data_train_1_3.csv", ".\\data\\data_small_chlorine\\scada_data_train_3_1.csv", ".\\data\\data_small_chlorine\\scada_data_train_3_2.csv", ".\\data\\data_small_chlorine\\scada_data_train_3_3.csv", ".\\data\\data_small_chlorine\\scada_data_train_4_1.csv", ".\\data\\data_small_chlorine\\scada_data_train_4_2.csv", ".\\data\\data_small_chlorine\\scada_data_train_4_3.csv", ".\\data\\data_small_chlorine\\scada_data_test_2.csv"]
        CONTAMINATED_FILES3 = [".\\data\\data_small_chlorine\\scada_data_train_1_1.csv", ".\\data\\data_small_chlorine\\scada_data_test_5.csv"]
        CONTAMINATED_FILES4 = [".\\data\\data_small_chlorine\\scada_data_train_1_1.csv", ".\\data\\data_small_chlorine\\scada_data_test_6.csv"]
    
          

        configs = [
        
        ExperimentConfig(
                        config_name="CNN_Window2",
                        contaminated_files=CONTAMINATED_FILES2,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=50, 
                        model_name=ModelName.CNN_WINDOWS,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        ), 
        
        ExperimentConfig(
                        config_name="CNN_Window3",
                        contaminated_files=CONTAMINATED_FILES3,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=50, 
                        model_name=ModelName.CNN_WINDOWS,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        ),
        ExperimentConfig(
                        config_name="CNN_Window4",
                        contaminated_files=CONTAMINATED_FILES4,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=50, 
                        model_name=ModelName.CNN_WINDOWS,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        )
        
        ]
        all_results = []


        for cfg in configs:
            runner = ExperimentRunner(cfg)
            res = runner.run()
            all_results.append(res)
            # print(all_results)

        pickle.dump(all_results, open(f"all_results_{node}_CNN_WINDOWS.pkl", "wb"))

        evaluation = Evaluation()
        evaluation_results = evaluation.evaluate(f"all_results_{node}_CNN_WINDOWS.pkl")
        print(evaluation_results)

        print("Evaluation results:")

        #evaluation.plot_confusion_matrices("CNN", evaluation_results)
        # evaluation.plot_results_nodes_within_configurations(Metrics.ACCURACY, evaluation_results)
        # evaluation.plot_mean_configuration(Metrics.ACCURACY, evaluation_results)




    for node in nodes: 
        print(f"Running experiments for node {node}...")
        
        CLEAN_FILES = [".\\data\\data_small_chlorine\\scada_data_clean_1.csv", ".\\data\\data_small_chlorine\\scada_data_clean_3.csv", ".\\data\\data_small_chlorine\\scada_data_clean_4.csv", ".\\data\\data_small_chlorine\\scada_data_clean_7.csv", ".\\data\\data_small_chlorine\\scada_data_clean_9.csv"]

        CONTAMINATED_FILES2 = [".\\data\\data_small_chlorine\\scada_data_train_1_1.csv", ".\\data\\data_small_chlorine\\scada_data_train_1_2.csv", ".\\data\\data_small_chlorine\\scada_data_train_1_3.csv", ".\\data\\data_small_chlorine\\scada_data_train_3_1.csv", ".\\data\\data_small_chlorine\\scada_data_train_3_2.csv", ".\\data\\data_small_chlorine\\scada_data_train_3_3.csv", ".\\data\\data_small_chlorine\\scada_data_train_4_1.csv", ".\\data\\data_small_chlorine\\scada_data_train_4_2.csv", ".\\data\\data_small_chlorine\\scada_data_train_4_3.csv", ".\\data\\data_small_chlorine\\scada_data_test_2.csv"]
        CONTAMINATED_FILES3 = [".\\data\\data_small_chlorine\\scada_data_train_1_1.csv", ".\\data\\data_small_chlorine\\scada_data_test_5.csv"]
        CONTAMINATED_FILES4 = [".\\data\\data_small_chlorine\\scada_data_train_1_1.csv", ".\\data\\data_small_chlorine\\scada_data_test_6.csv"]
    
    
        configs = [

        
        ExperimentConfig(
                        config_name="CNN_Window_VAE2",
                        contaminated_files=CONTAMINATED_FILES2,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=50, 
                        model_name=ModelName.CNN_WINDOWS_VAE,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        ), 
        
        ExperimentConfig(
                        config_name="CNN_Window_VAE3",
                        contaminated_files=CONTAMINATED_FILES3,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=50, 
                        model_name=ModelName.CNN_WINDOWS_VAE,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        ),
        ExperimentConfig(
                        config_name="CNN_Window_VAE4",
                        contaminated_files=CONTAMINATED_FILES4,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=50, 
                        model_name=ModelName.CNN_WINDOWS_VAE,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        )
        

        
        
        ]
        all_results = []


        for cfg in configs:
            runner = ExperimentRunner(cfg)
            res = runner.run()
            all_results.append(res)
            # print(all_results)

        pickle.dump(all_results, open(f"all_results_{node}_CNN_WINDOWS_VAE.pkl", "wb"))

        evaluation = Evaluation()
        evaluation_results = evaluation.evaluate(f"all_results_{node}_CNN_WINDOWS.pkl")
        print(evaluation_results)

        print("Evaluation results:")

        #evaluation.plot_confusion_matrices("CNN", evaluation_results)
        # evaluation.plot_results_nodes_within_configurations(Metrics.ACCURACY, evaluation_results)
        # evaluation.plot_mean_configuration(Metrics.ACCURACY, evaluation_results)
        



    for node in nodes: 
        print(f"Running experiments for node {node}...")
        
        CLEAN_FILES = [".\\data\\data_small_chlorine\\scada_data_clean_1.csv", ".\\data\\data_small_chlorine\\scada_data_clean_3.csv", ".\\data\\data_small_chlorine\\scada_data_clean_4.csv", ".\\data\\data_small_chlorine\\scada_data_clean_7.csv", ".\\data\\data_small_chlorine\\scada_data_clean_9.csv"]

        CONTAMINATED_FILES2 = [".\\data\\data_small_chlorine\\scada_data_train_1_1.csv", ".\\data\\data_small_chlorine\\scada_data_train_1_2.csv", ".\\data\\data_small_chlorine\\scada_data_train_1_3.csv", ".\\data\\data_small_chlorine\\scada_data_train_3_1.csv", ".\\data\\data_small_chlorine\\scada_data_train_3_2.csv", ".\\data\\data_small_chlorine\\scada_data_train_3_3.csv", ".\\data\\data_small_chlorine\\scada_data_train_4_1.csv", ".\\data\\data_small_chlorine\\scada_data_train_4_2.csv", ".\\data\\data_small_chlorine\\scada_data_train_4_3.csv", ".\\data\\data_small_chlorine\\scada_data_test_2.csv"]
        CONTAMINATED_FILES3 = [".\\data\\data_small_chlorine\\scada_data_train_1_1.csv", ".\\data\\data_small_chlorine\\scada_data_test_5.csv"]
        CONTAMINATED_FILES4 = [".\\data\\data_small_chlorine\\scada_data_train_1_1.csv", ".\\data\\data_small_chlorine\\scada_data_test_6.csv"]
    


        configs = [

        
        ExperimentConfig(
                        config_name="CNN_Univariate2",
                        contaminated_files=CONTAMINATED_FILES2,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=150, 
                        model_name=ModelName.CNN_UNIVARIATE,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        ), 
        
        ExperimentConfig(
                        config_name="CNN_Univariate3",
                        contaminated_files=CONTAMINATED_FILES3,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=150, 
                        model_name=ModelName.CNN_UNIVARIATE,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        ),
        ExperimentConfig(
                        config_name="CNN_Univariate4",
                        contaminated_files=CONTAMINATED_FILES4,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=150, 
                        model_name=ModelName.CNN_UNIVARIATE,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        )
        
        ]
        all_results = []


        for cfg in configs:
            runner = ExperimentRunner(cfg)
            res = runner.run()
            all_results.append(res)
            #print(all_results)

        pickle.dump(all_results, open(f"all_results_{node}_CNN_Univariate.pkl", "wb"))

        evaluation = Evaluation()
        evaluation_results = evaluation.evaluate(f"all_results_{node}_CNN_Univariate.pkl")
        print(evaluation_results)

        print("Evaluation results:")

        evaluation.plot_confusion_matrices("CNN", evaluation_results)
        evaluation.plot_results_nodes_within_configurations(Metrics.ACCURACY, evaluation_results)
        evaluation.plot_mean_configuration(Metrics.ACCURACY, evaluation_results)
    


    for node in nodes: 
        print(f"Running experiments for node {node}...")
        
        CLEAN_FILES = [".\\data\\data_small_chlorine\\scada_data_clean_1.csv", ".\\data\\data_small_chlorine\\scada_data_clean_3.csv", ".\\data\\data_small_chlorine\\scada_data_clean_4.csv", ".\\data\\data_small_chlorine\\scada_data_clean_7.csv", ".\\data\\data_small_chlorine\\scada_data_clean_9.csv"]

        CONTAMINATED_FILES2 = [".\\data\\data_small_chlorine\\scada_data_train_1_1.csv", ".\\data\\data_small_chlorine\\scada_data_train_1_2.csv", ".\\data\\data_small_chlorine\\scada_data_train_1_3.csv", ".\\data\\data_small_chlorine\\scada_data_train_3_1.csv", ".\\data\\data_small_chlorine\\scada_data_train_3_2.csv", ".\\data\\data_small_chlorine\\scada_data_train_3_3.csv", ".\\data\\data_small_chlorine\\scada_data_train_4_1.csv", ".\\data\\data_small_chlorine\\scada_data_train_4_2.csv", ".\\data\\data_small_chlorine\\scada_data_train_4_3.csv", ".\\data\\data_small_chlorine\\scada_data_test_2.csv"]
        CONTAMINATED_FILES3 = [".\\data\\data_small_chlorine\\scada_data_train_1_1.csv", ".\\data\\data_small_chlorine\\scada_data_test_5.csv"]
        CONTAMINATED_FILES4 = [".\\data\\data_small_chlorine\\scada_data_train_1_1.csv", ".\\data\\data_small_chlorine\\scada_data_test_6.csv"]
    
        configs = [

        
        ExperimentConfig(
                        config_name="Embedding_CNN2",
                        contaminated_files=CONTAMINATED_FILES2,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=100, 
                        model_name=ModelName.VAE_CNN,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        ), 
        
        ExperimentConfig(
                        config_name="Embedding_CNN3",
                        contaminated_files=CONTAMINATED_FILES3,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=100, 
                        model_name=ModelName.VAE_CNN,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        ), 
        ExperimentConfig(
                        config_name="Embedding_CNN4",
                        contaminated_files=CONTAMINATED_FILES4,
                        example_files=CLEAN_FILES,
                        nodes=[node],
                        window_size=100, 
                        model_name=ModelName.VAE_CNN,
                        model_params={},
                        contaminants=[ContaminationType.PATHOGEN]
        )
        
        ]
        all_results = []


        for cfg in configs:
            runner = ExperimentRunner(cfg)
            res = runner.run()
            all_results.append(res)
            # print(all_results)

        pickle.dump(all_results, open(f"all_results_{node}_Embedding_CNN.pkl", "wb"))

        evaluation = Evaluation()
        evaluation_results = evaluation.evaluate(f"all_results_{node}_Embedding_CNN.pkl")
        print(evaluation_results)

        # print("Evaluation results:")
