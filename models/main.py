from experiment import ExperimentRunner
from experiment_config import ExperimentConfig


if __name__ == "__main__":
    # Example: compare LOF vs IsolationForest on same node / same files
    CLEAN_FILES = [r".\data\data_arsenic\scada_data_no_contamination.csv"]
    CONTAMINATED_FILES = [r".\data\data_arsenic\scada_data_conta_22.csv"]

    configs = [
        ExperimentConfig(
            example_files=CLEAN_FILES,
            contaminated_files=CONTAMINATED_FILES,
            nodes=[22],
            window_size=30,
            model_name="LOF",
            model_params={"n_neighbors": 20, "contamination": 0.1}
        ),
        ExperimentConfig(
            contaminated_files=CONTAMINATED_FILES,
            nodes=[22],
            window_size=0,
            model_name="isolation_forest",
            model_params={"contamination": 0.1}
        )
    ]
    
    
    all_results = []
    for cfg in configs:
        runner = ExperimentRunner(cfg)
        res = runner.run()
        all_results.append(res)

    print(all_results)