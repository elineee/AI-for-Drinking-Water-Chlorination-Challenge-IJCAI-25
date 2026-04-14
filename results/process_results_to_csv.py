import pickle 
import os
import pandas as pd
import numpy as np
from experiment.evaluation import Evaluation

def _create_row_results(model: str, network: str, file: str, config_name: str, node: str, values: dict, evaluation_results: dict):
    """Create a row for the results dataframe.

    Parameters:
    - model: The name of the model.
    - network: The name of the network.
    - file: The name of the file containing the results.
    - config_name: The name of the configuration.
    - node: The name of the node.
    - values: The values containing the true and predicted labels.
    - evaluation_results: The evaluation results containing the metrics for the given file.

    Returns:
    - a dictionary containing the values for the row.
    """
    
    event_missed = 0 
    false_alarm = 0

    y_true = np.array(values["y_true"])
    y_pred = np.array(values["y_pred"])

    if -1 in y_true and -1 not in y_pred:
        event_missed = 1

    metrics = evaluation_results[config_name][node]
    cm = metrics["confusion_matrix"]

    if metrics["delay"] == 0: 
        false_alarm = 1

    return {
        "model": model,
        "network": network,
        "file": file,
        "config": config_name,
        "node": node,
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "delay": metrics["delay"],
        "TP": cm[1, 1],
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "event_missed": event_missed,
        "false_alarm": false_alarm
    }

def process_results_to_csv(folder="results"):
    """
    Process the results from the experiments.
    This function reads the .pkl files containing the experiment results from subfolders (model/network) and processes them.
    Results are then saved in a csv file.

    Parameters:
    folder (str): The folder containing the results. Default is "results".

    """
    rows = []
    evaluation = Evaluation()
    
    for model in os.listdir(folder):
        model_path = os.path.join(folder, model)
        if not os.path.isdir(model_path):
            continue

        for network in os.listdir(model_path):
            network_path = os.path.join(model_path, network)
            if not os.path.isdir(network_path):
                continue

            for file in os.listdir(network_path):
                if file.endswith('.pkl'):
                    path = os.path.join(network_path, file)
                    evaluation_results = evaluation.evaluate(path)
                    
                    with open(path, 'rb') as f:
                        results = pickle.load(f)

                    for result in results:
                        for config_name, nodes_dict in result.items():
                            for node, values in nodes_dict.items():
                                row = _create_row_results(model, network, file, config_name, node, values, evaluation_results)
                                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("results/results_summary.csv", index=False)

process_results_to_csv()
