from experiment import ExperimentRunner
from experiment_config import ContaminationType, ExperimentConfig, ModelName
from evaluation import Evaluation, Metrics

import pickle

# #################
# # DONT REMOVE FIRST 3 DAYS HERE IN THE CODE BC ALREADY DONE IN THE DATA !!!!!!
# ##################

if __name__ == "__main__":
    
    # node = "dist606"
    # print(f"Running experiments for node {node}...")
    
    # CLEAN_FILES = ["../data/data_ageing_network/scada_data_train_dist1332_5.csv", "../data/data_global_warming/scada_data_train_dist1332_93.csv", "../data/data_global_warming/scada_data_train_dist1915_48.csv","../data/data_global_warming/scada_data_train_dist1915_53.csv"]

    # CONTAMINATED_FILES2 = ["../data/data_ageing_network/scada_data_train_dist606_8.csv", "../data/data_ageing_network/scada_data_train_dist606_10.csv", "../data/data_ageing_network/scada_data_train_dist606_11.csv", "../data/data_ageing_network/scada_data_train_dist606_14.csv", "../data/data_ageing_network/scada_data_train_dist606_17.csv", "../data/data_ageing_network/scada_data_train_dist606_32.csv", "../data/data_ageing_network/scada_data_train_dist606_36.csv", "../data/data_ageing_network/scada_data_test_dist606_26.csv"]
    # CONTAMINATED_FILES3 = ["../data/data_ageing_network/scada_data_train_dist606_8.csv", "../data/data_ageing_network/scada_data_test_dist606_79.csv"]
    # CONTAMINATED_FILES4 = ["../data/data_ageing_network/scada_data_train_dist606_8.csv", "../data/data_ageing_network/scada_data_test_dist606_114.csv"]

    # configs = [

    # ExperimentConfig(
    #                 config_name="CNN2",
    #                 contaminated_files=CONTAMINATED_FILES2,
    #                 example_files=CLEAN_FILES,
    #                 nodes=[node],
    #                 window_size=288, 
    #                 model_name=ModelName.CNN,
    #                 model_params={},
    #                 contaminants=[ContaminationType.PATHOGEN]

    # ), 
    
    # ExperimentConfig(
    #                 config_name="CNN3",
    #                 contaminated_files=CONTAMINATED_FILES3,
    #                 example_files=CLEAN_FILES,
    #                 nodes=[node],
    #                 window_size=288, 
    #                 model_name=ModelName.CNN,
    #                 model_params={},
    #                 contaminants=[ContaminationType.PATHOGEN]
    # ),
    # ExperimentConfig(
    #                 config_name="CNN4",
    #                 contaminated_files=CONTAMINATED_FILES4,
    #                 example_files=CLEAN_FILES,
    #                 nodes=[node],
    #                 window_size=288, 
    #                 model_name=ModelName.CNN,
    #                 model_params={},
    #                 contaminants=[ContaminationType.PATHOGEN]
    # )
    
    
    # ]
    # all_results = []


    # for cfg in configs:
    #     runner = ExperimentRunner(cfg)
    #     res = runner.run()
    #     all_results.append(res)
    #     #print(all_results)

    # pickle.dump(all_results, open(f"all_results_{node}_CNN.pkl", "wb"))

    # evaluation = Evaluation()
    # evaluation_results = evaluation.evaluate(f"all_results_{node}_CNN.pkl")
    # print(evaluation_results)



    
    # node = "dist1332"
    # print(f"Running experiments for node {node}...")
    
    # CLEAN_FILES = ["../data/data_ageing_network/scada_data_train_dist606_8.csv", "../data/data_ageing_network/scada_data_train_dist606_10.csv", "../data/data_ageing_network/scada_data_train_dist606_11.csv", "../data/data_ageing_network/scada_data_train_dist606_17.csv", "../data/data_ageing_network/scada_data_train_dist606_32.csv", "../data/data_ageing_network/scada_data_train_dist606_36.csv"]

    # CONTAMINATED_FILES2 = ["../data/data_ageing_network/scada_data_train_dist1332_5.csv", "../data/data_ageing_network/scada_data_train_dist1332_9.csv", "../data/data_ageing_network/scada_data_train_dist1332_13.csv", "../data/data_ageing_network/scada_data_train_dist1332_20.csv", "../data/data_ageing_network/scada_data_train_dist1332_24.csv", "../data/data_ageing_network/scada_data_train_dist1332_65.csv", "../data/data_ageing_network/scada_data_train_dist1332_82.csv", "../data/data_ageing_network/scada_data_train_dist1332_93.csv", "../data/data_ageing_network/scada_data_train_dist1915_15.csv", "../data/data_ageing_network/scada_data_train_dist1915_25.csv", "../data/data_ageing_network/scada_data_test_dist1332_38.csv"]
    # CONTAMINATED_FILES3 = ["../data/data_ageing_network/scada_data_train_dist1332_5.csv", "../data/data_ageing_network/scada_data_test_dist1332_102.csv"]
    # CONTAMINATED_FILES4 = ["../data/data_ageing_network/scada_data_train_dist1332_5.csv", "../data/data_ageing_network/scada_data_test_dist1332_129.csv"]

    # configs = [

    # ExperimentConfig(
    #                 config_name="CNN2",
    #                 contaminated_files=CONTAMINATED_FILES2,
    #                 example_files=CLEAN_FILES,
    #                 nodes=[node],
    #                 window_size=288, 
    #                 model_name=ModelName.CNN,
    #                 model_params={},
    #                 contaminants=[ContaminationType.PATHOGEN]

    # ), 
    
    # ExperimentConfig(
    #                 config_name="CNN3",
    #                 contaminated_files=CONTAMINATED_FILES3,
    #                 example_files=CLEAN_FILES,
    #                 nodes=[node],
    #                 window_size=288, 
    #                 model_name=ModelName.CNN,
    #                 model_params={},
    #                 contaminants=[ContaminationType.PATHOGEN]
    # ),
    # ExperimentConfig(
    #                 config_name="CNN4",
    #                 contaminated_files=CONTAMINATED_FILES4,
    #                 example_files=CLEAN_FILES,
    #                 nodes=[node],
    #                 window_size=288, 
    #                 model_name=ModelName.CNN,
    #                 model_params={},
    #                 contaminants=[ContaminationType.PATHOGEN]
    # )
    
    
    # ]
    # all_results = []


    # for cfg in configs:
    #     runner = ExperimentRunner(cfg)
    #     res = runner.run()
    #     all_results.append(res)
    #     #print(all_results)

    # pickle.dump(all_results, open(f"all_results_{node}_CNN.pkl", "wb"))

    # evaluation = Evaluation()
    # evaluation_results = evaluation.evaluate(f"all_results_{node}_CNN.pkl")
    # print(evaluation_results)


    
    
    # node = "dist1915"
    # print(f"Running experiments for node {node}...")
    
    # CLEAN_FILES = ["../data/data_ageing_network/scada_data_train_dist606_8.csv", "../data/data_ageing_network/scada_data_train_dist606_10.csv", "../data/data_ageing_network/scada_data_train_dist606_11.csv", "../data/data_ageing_network/scada_data_train_dist606_17.csv", "../data/data_ageing_network/scada_data_train_dist606_32.csv", "../data/data_ageing_network/scada_data_train_dist606_36.csv"]

    # CONTAMINATED_FILES2 = ["../data/data_ageing_network/scada_data_train_dist1915_15.csv", "../data/data_ageing_network/scada_data_train_dist1915_19.csv", "../data/data_ageing_network/scada_data_train_dist1915_21.csv", "../data/data_ageing_network/scada_data_train_dist1915_25.csv", "../data/data_ageing_network/scada_data_train_dist1915_44.csv", "../data/data_ageing_network/scada_data_train_dist1915_48.csv", "../data/data_ageing_network/scada_data_train_dist1915_53.csv", "../data/data_ageing_network/scada_data_train_dist1915_107.csv", "../data/data_ageing_network/scada_data_test_dist1915_3.csv"]
    # CONTAMINATED_FILES3 = ["../data/data_ageing_network/scada_data_train_dist1332_5.csv", "../data/data_ageing_network/scada_data_test_dist1915_12.csv"]
    # CONTAMINATED_FILES4 = ["../data/data_ageing_network/scada_data_train_dist1332_5.csv", "../data/data_ageing_network/scada_data_test_dist1915_44.csv"]

    # configs = [

    # ExperimentConfig(
    #                 config_name="CNN2",
    #                 contaminated_files=CONTAMINATED_FILES2,
    #                 example_files=CLEAN_FILES,
    #                 nodes=[node],
    #                 window_size=288, 
    #                 model_name=ModelName.CNN,
    #                 model_params={},
    #                 contaminants=[ContaminationType.PATHOGEN]

    # ), 
    
    # ExperimentConfig(
    #                 config_name="CNN3",
    #                 contaminated_files=CONTAMINATED_FILES3,
    #                 example_files=CLEAN_FILES,
    #                 nodes=[node],
    #                 window_size=288, 
    #                 model_name=ModelName.CNN,
    #                 model_params={},
    #                 contaminants=[ContaminationType.PATHOGEN]
    # ),
    # ExperimentConfig(
    #                 config_name="CNN4",
    #                 contaminated_files=CONTAMINATED_FILES4,
    #                 example_files=CLEAN_FILES,
    #                 nodes=[node],
    #                 window_size=288, 
    #                 model_name=ModelName.CNN,
    #                 model_params={},
    #                 contaminants=[ContaminationType.PATHOGEN]
    # )
    
    
    # ]
    # all_results = []


    # for cfg in configs:
    #     runner = ExperimentRunner(cfg)
    #     res = runner.run()
    #     all_results.append(res)
    #     #print(all_results)

    # pickle.dump(all_results, open(f"all_results_{node}_CNN.pkl", "wb"))

    # evaluation = Evaluation()
    # evaluation_results = evaluation.evaluate(f"all_results_{node}_CNN.pkl")
    # print(evaluation_results)





# #############################################################################################################################################################################################################

    # node = "dist606"
    # print(f"Running experiments for node {node}...")
    
    # CLEAN_FILES = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_global_warming/scada_data_train_dist1332_93.csv", "./data/data_global_warming/scada_data_train_dist1915_48.csv","./data/data_global_warming/scada_data_train_dist1915_53.csv"]

    # CONTAMINATED_FILES2 = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_train_dist606_10.csv", "./data/data_ageing_network/scada_data_train_dist606_11.csv", "./data/data_ageing_network/scada_data_train_dist606_14.csv", "./data/data_ageing_network/scada_data_train_dist606_17.csv", "./data/data_ageing_network/scada_data_train_dist606_32.csv", "./data/data_ageing_network/scada_data_train_dist606_36.csv", "./data/data_ageing_network/scada_data_test_dist606_26.csv"]
    # CONTAMINATED_FILES3 = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_test_dist606_79.csv"]
    # CONTAMINATED_FILES4 = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_test_dist606_114.csv"]

    # configs = [
    
    # ExperimentConfig(
    #                 config_name="CNN_VAE2",
    #                 contaminated_files=CONTAMINATED_FILES2,
    #                 example_files=CLEAN_FILES,
    #                 nodes=[node],
    #                 window_size=288, 
    #                 model_name=ModelName.CNN_VAE,
    #                 model_params={},
    #                 contaminants=[ContaminationType.PATHOGEN]
    # ), 
    
    # ExperimentConfig(
    #                 config_name="CNN_VAE3",
    #                 contaminated_files=CONTAMINATED_FILES3,
    #                 example_files=CLEAN_FILES,
    #                 nodes=[node],
    #                 window_size=288, 
    #                 model_name=ModelName.CNN_VAE,
    #                 model_params={},
    #                 contaminants=[ContaminationType.PATHOGEN]
    # ),

    # ExperimentConfig(
    #                 config_name="CNN_VAE4",
    #                 contaminated_files=CONTAMINATED_FILES4,
    #                 example_files=CLEAN_FILES,
    #                 nodes=[node],
    #                 window_size=288, 
    #                 model_name=ModelName.CNN_VAE,
    #                 model_params={},
    #                 contaminants=[ContaminationType.PATHOGEN]
    # )
    
    
    # ]
    # all_results = []


    # for cfg in configs:
    #     runner = ExperimentRunner(cfg)
    #     res = runner.run()
    #     all_results.append(res)
    #     # print(all_results)

    # pickle.dump(all_results, open(f"all_results_{node}_CNN_VAE.pkl", "wb"))

    # evaluation = Evaluation()
    # evaluation_results = evaluation.evaluate(f"all_results_{node}_CNN_VAE.pkl")
    # print(evaluation_results)

    # print("Evaluation results:")



    
    # node = "dist1332"
    # print(f"Running experiments for node {node}...")
    
    # CLEAN_FILES = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_train_dist606_10.csv", "./data/data_ageing_network/scada_data_train_dist606_11.csv", "./data/data_ageing_network/scada_data_train_dist606_17.csv", "./data/data_ageing_network/scada_data_train_dist606_32.csv", "./data/data_ageing_network/scada_data_train_dist606_36.csv"]

    # CONTAMINATED_FILES2 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_train_dist1332_9.csv", "./data/data_ageing_network/scada_data_train_dist1332_13.csv", "./data/data_ageing_network/scada_data_train_dist1332_20.csv", "./data/data_ageing_network/scada_data_train_dist1332_24.csv", "./data/data_ageing_network/scada_data_train_dist1332_65.csv", "./data/data_ageing_network/scada_data_train_dist1332_82.csv", "./data/data_ageing_network/scada_data_train_dist1332_93.csv", "./data/data_ageing_network/scada_data_train_dist1915_15.csv", "./data/data_ageing_network/scada_data_train_dist1915_25.csv", "./data/data_ageing_network/scada_data_test_dist1332_38.csv"]
    # CONTAMINATED_FILES3 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_test_dist1332_102.csv"]
    # CONTAMINATED_FILES4 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_test_dist1332_129.csv"]
    # configs = [
    
    # ExperimentConfig(
    #                 config_name="CNN_VAE2",
    #                 contaminated_files=CONTAMINATED_FILES2,
    #                 example_files=CLEAN_FILES,
    #                 nodes=[node],
    #                 window_size=288, 
    #                 model_name=ModelName.CNN_VAE,
    #                 model_params={},
    #                 contaminants=[ContaminationType.PATHOGEN]
    # ), 
    
    # ExperimentConfig(
    #                 config_name="CNN_VAE3",
    #                 contaminated_files=CONTAMINATED_FILES3,
    #                 example_files=CLEAN_FILES,
    #                 nodes=[node],
    #                 window_size=288, 
    #                 model_name=ModelName.CNN_VAE,
    #                 model_params={},
    #                 contaminants=[ContaminationType.PATHOGEN]
    # ),

    # ExperimentConfig(
    #                 config_name="CNN_VAE4",
    #                 contaminated_files=CONTAMINATED_FILES4,
    #                 example_files=CLEAN_FILES,
    #                 nodes=[node],
    #                 window_size=288, 
    #                 model_name=ModelName.CNN_VAE,
    #                 model_params={},
    #                 contaminants=[ContaminationType.PATHOGEN]
    # )
    

    
    
    # ]
    # all_results = []


    # for cfg in configs:
    #     runner = ExperimentRunner(cfg)
    #     res = runner.run()
    #     all_results.append(res)
    #     # print(all_results)

    # pickle.dump(all_results, open(f"all_results_{node}_CNN_VAE.pkl", "wb"))

    # evaluation = Evaluation()
    # evaluation_results = evaluation.evaluate(f"all_results_{node}_CNN_VAE.pkl")
    # print(evaluation_results)

    # print("Evaluation results:")


    
    
    # node = "dist1915"
    # print(f"Running experiments for node {node}...")
    
    # CLEAN_FILES = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_train_dist606_10.csv", "./data/data_ageing_network/scada_data_train_dist606_11.csv", "./data/data_ageing_network/scada_data_train_dist606_17.csv", "./data/data_ageing_network/scada_data_train_dist606_32.csv", "./data/data_ageing_network/scada_data_train_dist606_36.csv"]

    # CONTAMINATED_FILES2 = ["./data/data_ageing_network/scada_data_train_dist1915_15.csv", "./data/data_ageing_network/scada_data_train_dist1915_19.csv", "./data/data_ageing_network/scada_data_train_dist1915_21.csv", "./data/data_ageing_network/scada_data_train_dist1915_25.csv", "./data/data_ageing_network/scada_data_train_dist1915_44.csv", "./data/data_ageing_network/scada_data_train_dist1915_48.csv", "./data/data_ageing_network/scada_data_train_dist1915_53.csv", "./data/data_ageing_network/scada_data_train_dist1915_107.csv", "./data/data_ageing_network/scada_data_test_dist1915_3.csv"]
    # CONTAMINATED_FILES3 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_test_dist1915_12.csv"]
    # CONTAMINATED_FILES4 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_test_dist1915_44.csv"]
    # configs = [
    
    # ExperimentConfig(
    #                 config_name="CNN_VAE2",
    #                 contaminated_files=CONTAMINATED_FILES2,
    #                 example_files=CLEAN_FILES,
    #                 nodes=[node],
    #                 window_size=288, 
    #                 model_name=ModelName.CNN_VAE,
    #                 model_params={},
    #                 contaminants=[ContaminationType.PATHOGEN]
    # ), 
    
    # ExperimentConfig(
    #                 config_name="CNN_VAE3",
    #                 contaminated_files=CONTAMINATED_FILES3,
    #                 example_files=CLEAN_FILES,
    #                 nodes=[node],
    #                 window_size=288, 
    #                 model_name=ModelName.CNN_VAE,
    #                 model_params={},
    #                 contaminants=[ContaminationType.PATHOGEN]
    # ),

    # ExperimentConfig(
    #                 config_name="CNN_VAE4",
    #                 contaminated_files=CONTAMINATED_FILES4,
    #                 example_files=CLEAN_FILES,
    #                 nodes=[node],
    #                 window_size=288, 
    #                 model_name=ModelName.CNN_VAE,
    #                 model_params={},
    #                 contaminants=[ContaminationType.PATHOGEN]
    # )

    
    # ]
    # all_results = []


    # for cfg in configs:
    #     runner = ExperimentRunner(cfg)
    #     res = runner.run()
    #     all_results.append(res)
    #     # print(all_results)

    # pickle.dump(all_results, open(f"all_results_{node}_CNN_VAE.pkl", "wb"))

    # evaluation = Evaluation()
    # evaluation_results = evaluation.evaluate(f"all_results_{node}_CNN_VAE.pkl")
    # print(evaluation_results)

    # print("Evaluation results:")


# ############################################################################################################################################################################################################


#     node = "dist606"
#     print(f"Running experiments for node {node}...")
    
#     CLEAN_FILES = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_global_warming/scada_data_train_dist1332_93.csv", "./data/data_global_warming/scada_data_train_dist1915_48.csv","./data/data_global_warming/scada_data_train_dist1915_53.csv"]

#     CONTAMINATED_FILES2 = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_train_dist606_10.csv", "./data/data_ageing_network/scada_data_train_dist606_11.csv", "./data/data_ageing_network/scada_data_train_dist606_14.csv", "./data/data_ageing_network/scada_data_train_dist606_17.csv", "./data/data_ageing_network/scada_data_train_dist606_32.csv", "./data/data_ageing_network/scada_data_train_dist606_36.csv", "./data/data_ageing_network/scada_data_test_dist606_26.csv"]
#     CONTAMINATED_FILES3 = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_test_dist606_79.csv"]
#     CONTAMINATED_FILES4 = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_test_dist606_114.csv"]
        

#     configs = [
    
#     ExperimentConfig(
#                     config_name="CNN_Window2",
#                     contaminated_files=CONTAMINATED_FILES2,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=150, 
#                     model_name=ModelName.CNN_WINDOWS,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     ), 
    
#     ExperimentConfig(
#                     config_name="CNN_Window3",
#                     contaminated_files=CONTAMINATED_FILES3,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=150, 
#                     model_name=ModelName.CNN_WINDOWS,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     ),
#     ExperimentConfig(
#                     config_name="CNN_Window4",
#                     contaminated_files=CONTAMINATED_FILES4,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=150, 
#                     model_name=ModelName.CNN_WINDOWS,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     )
    
#     ]
#     all_results = []


#     for cfg in configs:
#         runner = ExperimentRunner(cfg)
#         res = runner.run()
#         all_results.append(res)
#         # print(all_results)

#     pickle.dump(all_results, open(f"all_results_{node}_CNN_WINDOWS.pkl", "wb"))

#     evaluation = Evaluation()
#     evaluation_results = evaluation.evaluate(f"all_results_{node}_CNN_WINDOWS.pkl")
#     print(evaluation_results)




#     node = "dist1332"
#     print(f"Running experiments for node {node}...")
    
#     CLEAN_FILES = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_train_dist606_10.csv", "./data/data_ageing_network/scada_data_train_dist606_11.csv", "./data/data_ageing_network/scada_data_train_dist606_17.csv", "./data/data_ageing_network/scada_data_train_dist606_32.csv", "./data/data_ageing_network/scada_data_train_dist606_36.csv"]

#     CONTAMINATED_FILES2 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_train_dist1332_9.csv", "./data/data_ageing_network/scada_data_train_dist1332_13.csv", "./data/data_ageing_network/scada_data_train_dist1332_20.csv", "./data/data_ageing_network/scada_data_train_dist1332_24.csv", "./data/data_ageing_network/scada_data_train_dist1332_65.csv", "./data/data_ageing_network/scada_data_train_dist1332_82.csv", "./data/data_ageing_network/scada_data_train_dist1332_93.csv", "./data/data_ageing_network/scada_data_train_dist1915_15.csv", "./data/data_ageing_network/scada_data_train_dist1915_25.csv", "./data/data_ageing_network/scada_data_test_dist1332_38.csv"]
#     CONTAMINATED_FILES3 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_test_dist1332_102.csv"]
#     CONTAMINATED_FILES4 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_test_dist1332_129.csv"]
        

#     configs = [
    
#     ExperimentConfig(
#                     config_name="CNN_Window2",
#                     contaminated_files=CONTAMINATED_FILES2,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=150, 
#                     model_name=ModelName.CNN_WINDOWS,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     ), 
    
#     ExperimentConfig(
#                     config_name="CNN_Window3",
#                     contaminated_files=CONTAMINATED_FILES3,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=150, 
#                     model_name=ModelName.CNN_WINDOWS,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     ),
#     ExperimentConfig(
#                     config_name="CNN_Window4",
#                     contaminated_files=CONTAMINATED_FILES4,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=150, 
#                     model_name=ModelName.CNN_WINDOWS,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     )
    
#     ]
#     all_results = []


#     for cfg in configs:
#         runner = ExperimentRunner(cfg)
#         res = runner.run()
#         all_results.append(res)
#         # print(all_results)

#     pickle.dump(all_results, open(f"all_results_{node}_CNN_WINDOWS.pkl", "wb"))

#     evaluation = Evaluation()
#     evaluation_results = evaluation.evaluate(f"all_results_{node}_CNN_WINDOWS.pkl")
#     print(evaluation_results)

    


#     node = "dist1915"
#     print(f"Running experiments for node {node}...")
    
#     CLEAN_FILES = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_train_dist606_10.csv", "./data/data_ageing_network/scada_data_train_dist606_11.csv", "./data/data_ageing_network/scada_data_train_dist606_17.csv", "./data/data_ageing_network/scada_data_train_dist606_32.csv", "./data/data_ageing_network/scada_data_train_dist606_36.csv"]

#     CONTAMINATED_FILES2 = ["./data/data_ageing_network/scada_data_train_dist1915_15.csv", "./data/data_ageing_network/scada_data_train_dist1915_19.csv", "./data/data_ageing_network/scada_data_train_dist1915_21.csv", "./data/data_ageing_network/scada_data_train_dist1915_25.csv", "./data/data_ageing_network/scada_data_train_dist1915_44.csv", "./data/data_ageing_network/scada_data_train_dist1915_48.csv", "./data/data_ageing_network/scada_data_train_dist1915_53.csv", "./data/data_ageing_network/scada_data_train_dist1915_107.csv", "./data/data_ageing_network/scada_data_test_dist1915_3.csv"]
#     CONTAMINATED_FILES3 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_test_dist1915_12.csv"]
#     CONTAMINATED_FILES4 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_test_dist1915_44.csv"]

#     configs = [
    
#     ExperimentConfig(
#                     config_name="CNN_Window2",
#                     contaminated_files=CONTAMINATED_FILES2,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=150, 
#                     model_name=ModelName.CNN_WINDOWS,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     ), 
    
#     ExperimentConfig(
#                     config_name="CNN_Window3",
#                     contaminated_files=CONTAMINATED_FILES3,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=150, 
#                     model_name=ModelName.CNN_WINDOWS,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     ),
#     ExperimentConfig(
#                     config_name="CNN_Window4",
#                     contaminated_files=CONTAMINATED_FILES4,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=150, 
#                     model_name=ModelName.CNN_WINDOWS,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     )
    
#     ]
#     all_results = []


#     for cfg in configs:
#         runner = ExperimentRunner(cfg)
#         res = runner.run()
#         all_results.append(res)
#         # print(all_results)

#     pickle.dump(all_results, open(f"all_results_{node}_CNN_WINDOWS.pkl", "wb"))

#     evaluation = Evaluation()
#     evaluation_results = evaluation.evaluate(f"all_results_{node}_CNN_WINDOWS.pkl")
#     print(evaluation_results)



# ############################################################################################################################################################################################################



#     node = "dist606"
#     print(f"Running experiments for node {node}...")
    
#     CLEAN_FILES = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_global_warming/scada_data_train_dist1332_93.csv", "./data/data_global_warming/scada_data_train_dist1915_48.csv","./data/data_global_warming/scada_data_train_dist1915_53.csv"]

#     CONTAMINATED_FILES2 = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_train_dist606_10.csv", "./data/data_ageing_network/scada_data_train_dist606_11.csv", "./data/data_ageing_network/scada_data_train_dist606_14.csv", "./data/data_ageing_network/scada_data_train_dist606_17.csv", "./data/data_ageing_network/scada_data_train_dist606_32.csv", "./data/data_ageing_network/scada_data_train_dist606_36.csv", "./data/data_ageing_network/scada_data_test_dist606_26.csv"]
#     CONTAMINATED_FILES3 = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_test_dist606_79.csv"]
#     CONTAMINATED_FILES4 = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_test_dist606_114.csv"]

#     configs = [

    
#     ExperimentConfig(
#                     config_name="CNN_Window_VAE2",
#                     contaminated_files=CONTAMINATED_FILES2,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=100, 
#                     model_name=ModelName.CNN_WINDOWS_VAE,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     ), 
    
#     ExperimentConfig(
#                     config_name="CNN_Window_VAE3",
#                     contaminated_files=CONTAMINATED_FILES3,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=100, 
#                     model_name=ModelName.CNN_WINDOWS_VAE,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     ),
#     ExperimentConfig(
#                     config_name="CNN_Window_VAE4",
#                     contaminated_files=CONTAMINATED_FILES4,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=100, 
#                     model_name=ModelName.CNN_WINDOWS_VAE,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     )
    

    
    
#     ]
#     all_results = []


#     for cfg in configs:
#         runner = ExperimentRunner(cfg)
#         res = runner.run()
#         all_results.append(res)
#         # print(all_results)

#     pickle.dump(all_results, open(f"all_results_{node}_CNN_WINDOWS_VAE.pkl", "wb"))

#     evaluation = Evaluation()
#     evaluation_results = evaluation.evaluate(f"all_results_{node}_CNN_WINDOWS_VAE.pkl")
#     print(evaluation_results)



    
#     node = "dist1332"
#     print(f"Running experiments for node {node}...")
    
#     CLEAN_FILES = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_train_dist606_10.csv", "./data/data_ageing_network/scada_data_train_dist606_11.csv", "./data/data_ageing_network/scada_data_train_dist606_17.csv", "./data/data_ageing_network/scada_data_train_dist606_32.csv", "./data/data_ageing_network/scada_data_train_dist606_36.csv"]

#     CONTAMINATED_FILES2 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_train_dist1332_9.csv", "./data/data_ageing_network/scada_data_train_dist1332_13.csv", "./data/data_ageing_network/scada_data_train_dist1332_20.csv", "./data/data_ageing_network/scada_data_train_dist1332_24.csv", "./data/data_ageing_network/scada_data_train_dist1332_65.csv", "./data/data_ageing_network/scada_data_train_dist1332_82.csv", "./data/data_ageing_network/scada_data_train_dist1332_93.csv", "./data/data_ageing_network/scada_data_train_dist1915_15.csv", "./data/data_ageing_network/scada_data_train_dist1915_25.csv", "./data/data_ageing_network/scada_data_test_dist1332_38.csv"]
#     CONTAMINATED_FILES3 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_test_dist1332_102.csv"]
#     CONTAMINATED_FILES4 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_test_dist1332_129.csv"]

#     configs = [

    
#     ExperimentConfig(
#                     config_name="CNN_Window_VAE2",
#                     contaminated_files=CONTAMINATED_FILES2,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=100, 
#                     model_name=ModelName.CNN_WINDOWS_VAE,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     ), 
    
#     ExperimentConfig(
#                     config_name="CNN_Window_VAE3",
#                     contaminated_files=CONTAMINATED_FILES3,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=100, 
#                     model_name=ModelName.CNN_WINDOWS_VAE,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     ),
#     ExperimentConfig(
#                     config_name="CNN_Window_VAE4",
#                     contaminated_files=CONTAMINATED_FILES4,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=100, 
#                     model_name=ModelName.CNN_WINDOWS_VAE,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     )
    

    
    
#     ]
#     all_results = []


#     for cfg in configs:
#         runner = ExperimentRunner(cfg)
#         res = runner.run()
#         all_results.append(res)
#         # print(all_results)

#     pickle.dump(all_results, open(f"all_results_{node}_CNN_WINDOWS_VAE.pkl", "wb"))

#     evaluation = Evaluation()
#     evaluation_results = evaluation.evaluate(f"all_results_{node}_CNN_WINDOWS_VAE.pkl")
#     print(evaluation_results)



    
#     node = "dist1915"
#     print(f"Running experiments for node {node}...")
    
#     CLEAN_FILES = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_train_dist606_10.csv", "./data/data_ageing_network/scada_data_train_dist606_11.csv", "./data/data_ageing_network/scada_data_train_dist606_17.csv", "./data/data_ageing_network/scada_data_train_dist606_32.csv", "./data/data_ageing_network/scada_data_train_dist606_36.csv"]

#     CONTAMINATED_FILES2 = ["./data/data_ageing_network/scada_data_train_dist1915_15.csv", "./data/data_ageing_network/scada_data_train_dist1915_19.csv", "./data/data_ageing_network/scada_data_train_dist1915_21.csv", "./data/data_ageing_network/scada_data_train_dist1915_25.csv", "./data/data_ageing_network/scada_data_train_dist1915_44.csv", "./data/data_ageing_network/scada_data_train_dist1915_48.csv", "./data/data_ageing_network/scada_data_train_dist1915_53.csv", "./data/data_ageing_network/scada_data_train_dist1915_107.csv", "./data/data_ageing_network/scada_data_test_dist1915_3.csv"]
#     CONTAMINATED_FILES3 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_test_dist1915_12.csv"]
#     CONTAMINATED_FILES4 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_test_dist1915_44.csv"]

#     configs = [

    
#     ExperimentConfig(
#                     config_name="CNN_Window_VAE2",
#                     contaminated_files=CONTAMINATED_FILES2,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=100, 
#                     model_name=ModelName.CNN_WINDOWS_VAE,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     ), 
    
#     ExperimentConfig(
#                     config_name="CNN_Window_VAE3",
#                     contaminated_files=CONTAMINATED_FILES3,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=100, 
#                     model_name=ModelName.CNN_WINDOWS_VAE,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     ),
#     ExperimentConfig(
#                     config_name="CNN_Window_VAE4",
#                     contaminated_files=CONTAMINATED_FILES4,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=100, 
#                     model_name=ModelName.CNN_WINDOWS_VAE,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     )
    

    
    
#     ]
#     all_results = []


#     for cfg in configs:
#         runner = ExperimentRunner(cfg)
#         res = runner.run()
#         all_results.append(res)
#         # print(all_results)

#     pickle.dump(all_results, open(f"all_results_{node}_CNN_WINDOWS_VAE.pkl", "wb"))

#     evaluation = Evaluation()
#     evaluation_results = evaluation.evaluate(f"all_results_{node}_CNN_WINDOWS_VAE.pkl")
#     print(evaluation_results)

    
        
# ############################################################################################################################################################################################################


 
#     node = "dist606"
#     print(f"Running experiments for node {node}...")
    
#     CLEAN_FILES = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_global_warming/scada_data_train_dist1332_93.csv", "./data/data_global_warming/scada_data_train_dist1915_48.csv","./data/data_global_warming/scada_data_train_dist1915_53.csv"]

#     CONTAMINATED_FILES2 = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_train_dist606_10.csv", "./data/data_ageing_network/scada_data_train_dist606_11.csv", "./data/data_ageing_network/scada_data_train_dist606_14.csv", "./data/data_ageing_network/scada_data_train_dist606_17.csv", "./data/data_ageing_network/scada_data_train_dist606_32.csv", "./data/data_ageing_network/scada_data_train_dist606_36.csv", "./data/data_ageing_network/scada_data_test_dist606_26.csv"]
#     CONTAMINATED_FILES3 = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_test_dist606_79.csv"]
#     CONTAMINATED_FILES4 = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_test_dist606_114.csv"]


#     configs = [

    
#     ExperimentConfig(
#                     config_name="CNN_Univariate2",
#                     contaminated_files=CONTAMINATED_FILES2,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=400, 
#                     model_name=ModelName.CNN_UNIVARIATE,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     ), 
    
#     ExperimentConfig(
#                     config_name="CNN_Univariate3",
#                     contaminated_files=CONTAMINATED_FILES3,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=400, 
#                     model_name=ModelName.CNN_UNIVARIATE,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     ),
#     ExperimentConfig(
#                     config_name="CNN_Univariate4",
#                     contaminated_files=CONTAMINATED_FILES4,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=400, 
#                     model_name=ModelName.CNN_UNIVARIATE,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     )
    
#     ]
#     all_results = []


#     for cfg in configs:
#         runner = ExperimentRunner(cfg)
#         res = runner.run()
#         all_results.append(res)
#         #print(all_results)

#     pickle.dump(all_results, open(f"all_results_{node}_CNN_Univariate.pkl", "wb"))

#     evaluation = Evaluation()
#     evaluation_results = evaluation.evaluate(f"all_results_{node}_CNN_Univariate.pkl")
#     print(evaluation_results)



    
#     node = "dist1332"
#     print(f"Running experiments for node {node}...")
    
#     CLEAN_FILES = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_train_dist606_10.csv", "./data/data_ageing_network/scada_data_train_dist606_11.csv", "./data/data_ageing_network/scada_data_train_dist606_17.csv", "./data/data_ageing_network/scada_data_train_dist606_32.csv", "./data/data_ageing_network/scada_data_train_dist606_36.csv"]

#     CONTAMINATED_FILES2 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_train_dist1332_9.csv", "./data/data_ageing_network/scada_data_train_dist1332_13.csv", "./data/data_ageing_network/scada_data_train_dist1332_20.csv", "./data/data_ageing_network/scada_data_train_dist1332_24.csv", "./data/data_ageing_network/scada_data_train_dist1332_65.csv", "./data/data_ageing_network/scada_data_train_dist1332_82.csv", "./data/data_ageing_network/scada_data_train_dist1332_93.csv", "./data/data_ageing_network/scada_data_train_dist1915_15.csv", "./data/data_ageing_network/scada_data_train_dist1915_25.csv", "./data/data_ageing_network/scada_data_test_dist1332_38.csv"]
#     CONTAMINATED_FILES3 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_test_dist1332_102.csv"]
#     CONTAMINATED_FILES4 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_test_dist1332_129.csv"]

#     configs = [

    
#     ExperimentConfig(
#                     config_name="CNN_Univariate2",
#                     contaminated_files=CONTAMINATED_FILES2,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=400, 
#                     model_name=ModelName.CNN_UNIVARIATE,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     ), 
    
#     ExperimentConfig(
#                     config_name="CNN_Univariate3",
#                     contaminated_files=CONTAMINATED_FILES3,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=400, 
#                     model_name=ModelName.CNN_UNIVARIATE,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     ),
#     ExperimentConfig(
#                     config_name="CNN_Univariate4",
#                     contaminated_files=CONTAMINATED_FILES4,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=400, 
#                     model_name=ModelName.CNN_UNIVARIATE,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     )
    
#     ]
#     all_results = []


#     for cfg in configs:
#         runner = ExperimentRunner(cfg)
#         res = runner.run()
#         all_results.append(res)
#         #print(all_results)

#     pickle.dump(all_results, open(f"all_results_{node}_CNN_Univariate.pkl", "wb"))

#     evaluation = Evaluation()
#     evaluation_results = evaluation.evaluate(f"all_results_{node}_CNN_Univariate.pkl")
#     print(evaluation_results)

 

    
#     node = "dist1915"
#     print(f"Running experiments for node {node}...")
    
#     CLEAN_FILES = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_train_dist606_10.csv", "./data/data_ageing_network/scada_data_train_dist606_11.csv", "./data/data_ageing_network/scada_data_train_dist606_17.csv", "./data/data_ageing_network/scada_data_train_dist606_32.csv", "./data/data_ageing_network/scada_data_train_dist606_36.csv"]

#     CONTAMINATED_FILES2 = ["./data/data_ageing_network/scada_data_train_dist1915_15.csv", "./data/data_ageing_network/scada_data_train_dist1915_19.csv", "./data/data_ageing_network/scada_data_train_dist1915_21.csv", "./data/data_ageing_network/scada_data_train_dist1915_25.csv", "./data/data_ageing_network/scada_data_train_dist1915_44.csv", "./data/data_ageing_network/scada_data_train_dist1915_48.csv", "./data/data_ageing_network/scada_data_train_dist1915_53.csv", "./data/data_ageing_network/scada_data_train_dist1915_107.csv", "./data/data_ageing_network/scada_data_test_dist1915_3.csv"]
#     CONTAMINATED_FILES3 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_test_dist1915_12.csv"]
#     CONTAMINATED_FILES4 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_test_dist1915_44.csv"]


#     configs = [

    
#     ExperimentConfig(
#                     config_name="CNN_Univariate2",
#                     contaminated_files=CONTAMINATED_FILES2,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=400, 
#                     model_name=ModelName.CNN_UNIVARIATE,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     ), 
    
#     ExperimentConfig(
#                     config_name="CNN_Univariate3",
#                     contaminated_files=CONTAMINATED_FILES3,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=400, 
#                     model_name=ModelName.CNN_UNIVARIATE,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     ),
#     ExperimentConfig(
#                     config_name="CNN_Univariate4",
#                     contaminated_files=CONTAMINATED_FILES4,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=400, 
#                     model_name=ModelName.CNN_UNIVARIATE,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     )
    
#     ]
#     all_results = []


#     for cfg in configs:
#         runner = ExperimentRunner(cfg)
#         res = runner.run()
#         all_results.append(res)
#         #print(all_results)

#     pickle.dump(all_results, open(f"all_results_{node}_CNN_Univariate.pkl", "wb"))

#     evaluation = Evaluation()
#     evaluation_results = evaluation.evaluate(f"all_results_{node}_CNN_Univariate.pkl")
#     print(evaluation_results)



# ############################################################################################################################################################################################################
    


#     node = "dist606"
#     print(f"Running experiments for node {node}...")
    
#     CLEAN_FILES = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_global_warming/scada_data_train_dist1332_93.csv", "./data/data_global_warming/scada_data_train_dist1915_48.csv","./data/data_global_warming/scada_data_train_dist1915_53.csv"]

#     CONTAMINATED_FILES2 = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_train_dist606_10.csv", "./data/data_ageing_network/scada_data_train_dist606_11.csv", "./data/data_ageing_network/scada_data_train_dist606_14.csv", "./data/data_ageing_network/scada_data_train_dist606_17.csv", "./data/data_ageing_network/scada_data_train_dist606_32.csv", "./data/data_ageing_network/scada_data_train_dist606_36.csv", "./data/data_ageing_network/scada_data_test_dist606_26.csv"]
#     CONTAMINATED_FILES3 = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_test_dist606_79.csv"]
#     CONTAMINATED_FILES4 = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_test_dist606_114.csv"]
    
#     configs = [

#     ExperimentConfig(
#                     config_name="Embedding_CNN2",
#                     contaminated_files=CONTAMINATED_FILES2,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=100, 
#                     model_name=ModelName.VAE_CNN,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     ), 
    
#     ExperimentConfig(
#                     config_name="Embedding_CNN3",
#                     contaminated_files=CONTAMINATED_FILES3,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=100, 
#                     model_name=ModelName.VAE_CNN,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     ), 
#     ExperimentConfig(
#                     config_name="Embedding_CNN4",
#                     contaminated_files=CONTAMINATED_FILES4,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=100, 
#                     model_name=ModelName.VAE_CNN,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     )
    
#     ]
#     all_results = []


#     for cfg in configs:
#         runner = ExperimentRunner(cfg)
#         res = runner.run()
#         all_results.append(res)
#         # print(all_results)

#     pickle.dump(all_results, open(f"all_results_{node}_Embedding_CNN.pkl", "wb"))

#     evaluation = Evaluation()
#     evaluation_results = evaluation.evaluate(f"all_results_{node}_Embedding_CNN.pkl")
#     print(evaluation_results)


    

#     node = "dist1332"
#     print(f"Running experiments for node {node}...")
    
#     CLEAN_FILES = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_train_dist606_10.csv", "./data/data_ageing_network/scada_data_train_dist606_11.csv", "./data/data_ageing_network/scada_data_train_dist606_17.csv", "./data/data_ageing_network/scada_data_train_dist606_32.csv", "./data/data_ageing_network/scada_data_train_dist606_36.csv"]

#     CONTAMINATED_FILES2 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_train_dist1332_9.csv", "./data/data_ageing_network/scada_data_train_dist1332_13.csv", "./data/data_ageing_network/scada_data_train_dist1332_20.csv", "./data/data_ageing_network/scada_data_train_dist1332_24.csv", "./data/data_ageing_network/scada_data_train_dist1332_65.csv", "./data/data_ageing_network/scada_data_train_dist1332_82.csv", "./data/data_ageing_network/scada_data_train_dist1332_93.csv", "./data/data_ageing_network/scada_data_train_dist1915_15.csv", "./data/data_ageing_network/scada_data_train_dist1915_25.csv", "./data/data_ageing_network/scada_data_test_dist1332_38.csv"]
#     CONTAMINATED_FILES3 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_test_dist1332_102.csv"]
#     CONTAMINATED_FILES4 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_test_dist1332_129.csv"]

#     configs = [

    
#     ExperimentConfig(
#                     config_name="Embedding_CNN2",
#                     contaminated_files=CONTAMINATED_FILES2,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=100, 
#                     model_name=ModelName.VAE_CNN,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     ), 
    
#     ExperimentConfig(
#                     config_name="Embedding_CNN3",
#                     contaminated_files=CONTAMINATED_FILES3,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=100, 
#                     model_name=ModelName.VAE_CNN,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     ), 
#     ExperimentConfig(
#                     config_name="Embedding_CNN4",
#                     contaminated_files=CONTAMINATED_FILES4,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=100, 
#                     model_name=ModelName.VAE_CNN,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     )
    
#     ]
#     all_results = []


#     for cfg in configs:
#         runner = ExperimentRunner(cfg)
#         res = runner.run()
#         all_results.append(res)
#         # print(all_results)

#     pickle.dump(all_results, open(f"all_results_{node}_Embedding_CNN.pkl", "wb"))

#     evaluation = Evaluation()
#     evaluation_results = evaluation.evaluate(f"all_results_{node}_Embedding_CNN.pkl")
#     print(evaluation_results)




#     node = "dist1915"
#     print(f"Running experiments for node {node}...")
    
#     CLEAN_FILES = ["./data/data_ageing_network/scada_data_train_dist606_8.csv", "./data/data_ageing_network/scada_data_train_dist606_10.csv", "./data/data_ageing_network/scada_data_train_dist606_11.csv", "./data/data_ageing_network/scada_data_train_dist606_17.csv", "./data/data_ageing_network/scada_data_train_dist606_32.csv", "./data/data_ageing_network/scada_data_train_dist606_36.csv"]

#     CONTAMINATED_FILES2 = ["./data/data_ageing_network/scada_data_train_dist1915_15.csv", "./data/data_ageing_network/scada_data_train_dist1915_19.csv", "./data/data_ageing_network/scada_data_train_dist1915_21.csv", "./data/data_ageing_network/scada_data_train_dist1915_25.csv", "./data/data_ageing_network/scada_data_train_dist1915_44.csv", "./data/data_ageing_network/scada_data_train_dist1915_48.csv", "./data/data_ageing_network/scada_data_train_dist1915_53.csv", "./data/data_ageing_network/scada_data_train_dist1915_107.csv", "./data/data_ageing_network/scada_data_test_dist1915_3.csv"]
#     CONTAMINATED_FILES3 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_test_dist1915_12.csv"]
#     CONTAMINATED_FILES4 = ["./data/data_ageing_network/scada_data_train_dist1332_5.csv", "./data/data_ageing_network/scada_data_test_dist1915_44.csv"]

#     configs = [

    
#     ExperimentConfig(
#                     config_name="Embedding_CNN2",
#                     contaminated_files=CONTAMINATED_FILES2,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=100, 
#                     model_name=ModelName.VAE_CNN,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     ), 
    
#     ExperimentConfig(
#                     config_name="Embedding_CNN3",
#                     contaminated_files=CONTAMINATED_FILES3,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=100, 
#                     model_name=ModelName.VAE_CNN,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     ), 
#     ExperimentConfig(
#                     config_name="Embedding_CNN4",
#                     contaminated_files=CONTAMINATED_FILES4,
#                     example_files=CLEAN_FILES,
#                     nodes=[node],
#                     window_size=100, 
#                     model_name=ModelName.VAE_CNN,
#                     model_params={},
#                     contaminants=[ContaminationType.PATHOGEN]
#     )
    
#     ]
#     all_results = []


#     for cfg in configs:
#         runner = ExperimentRunner(cfg)
#         res = runner.run()
#         all_results.append(res)
#         # print(all_results)

#     pickle.dump(all_results, open(f"all_results_{node}_Embedding_CNN.pkl", "wb"))

#     evaluation = Evaluation()
#     evaluation_results = evaluation.evaluate(f"all_results_{node}_Embedding_CNN.pkl")
#     print(evaluation_results)

#     # print("Evaluation results:")
