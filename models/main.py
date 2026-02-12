from experiment import ExperimentRunner
from experiment_config import ExperimentConfig, ModelName
from evaluation import Evaluation

import pickle

if __name__ == "__main__":
    CLEAN_FILES = [".\\data\\data_arsenic\\scada_data_no_contamination.csv"]
    CONTAMINATED_FILES = [".\\data\\data_arsenic\\scada_data_conta_22.csv"]

    configs = [
        ExperimentConfig(
            config_name="LOF_20_neighbors",
            example_files=CLEAN_FILES,
            contaminated_files=CONTAMINATED_FILES,
            nodes=[22],
            window_size=30,
            model_name=ModelName.LOF,
            model_params={"n_neighbors": 20, "contamination": 0.1}
        ),
        ExperimentConfig(
            config_name="IsolationForest",
            contaminated_files=CONTAMINATED_FILES,
            nodes=[22],
            window_size=0,
            model_name=ModelName.ISOLATION_FOREST,
            model_params={"contamination": 0.1}
        )
    ]

    all_results = []

    for cfg in configs:
        runner = ExperimentRunner(cfg)
        res = runner.run()
        all_results.append(res)

    pickle.dump(all_results, open("all_results.pkl", "wb"))
    print(all_results)

    evaluation = Evaluation()
    evaluation_results = evaluation.evaluate("all_results.pkl")

    print("Evaluation results:")
    print(evaluation_results)