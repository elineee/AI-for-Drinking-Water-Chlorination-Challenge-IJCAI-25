import pickle 
import os
import pandas as pd
import numpy as np

from experiment.evaluation import Evaluation

def process_results(folder="results"):
    """
    Process the results from the experiments.
    This function reads the .pkl files containing the experiment results, processes them and saves them in a csv file.
    """
    rows = []
    evaluation = Evaluation()

    for file in os.listdir(folder):
        if file.endswith('.pkl'):
            path = os.path.join(folder, file)
            
            evaluation_results = evaluation.evaluate(path)
            
            with open(path, 'rb') as f:
                results = pickle.load(f)

            for result in results:
                for config_name, nodes_dict in result.items():
                    for node, values in nodes_dict.items():
                        
                        event_missed = 0 
                        false_alarm = 0

                        y_true = np.array(values["y_true"])
                        y_pred = np.array(values["y_pred"])
                        if -1 in y_true and -1 not in y_pred:
                            event_missed = 1

                        metrics = evaluation_results[config_name][node]
                        print(metrics)
                        cm = metrics["confusion_matrix"]

                        if metrics["delay"] == 0: 
                            false_alarm = 1

                        rows.append({
                            "file": file,
                            "config": config_name,
                            "node": node,
                            "accuracy": metrics["accuracy"],
                            "recall": metrics["recall"],
                            "f1_score": metrics["f1_score"],
                            "delay": metrics["delay"],
                            "TP": cm[1, 1],
                            "TN": cm[0, 0],
                            "FP": cm[0, 1],
                            "FN": cm[1, 0],
                            "event_missed": event_missed,
                            "false_alarm": false_alarm
                        })

    df = pd.DataFrame(rows)
    df.to_csv("results_summary.csv", index=False)

print(process_results())