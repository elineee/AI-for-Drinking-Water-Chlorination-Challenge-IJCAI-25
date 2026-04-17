import pandas as pd

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_latex.html

def create_results_summary_table(csv_file):
    """
    Create a summary table from the results csv file per network with the average metrics per model.
    Delay average excludes rows where false_alarm=1 or event_missed=1.

    Parameters:
    - csv_file: The path to the csv file containing the results.

    Returns:
    - dict: A dictionary of DataFrames, one per network.
    """

    df = pd.read_csv(csv_file)
    tables = {}

    for network in df["network"].unique():
        network_df = df[df["network"] == network]
        
        rows = []
        for model in network_df["model"].unique():
            model_df = network_df[network_df["model"] == model]
            recall_avg = model_df["recall"].mean()
            f1_score_avg = model_df["f1_score"].mean()
            delay = model_df[(model_df["false_alarm"] == 0) & (model_df["event_missed"] == 0)]["delay"]
            delay_avg = delay.mean()
            total = len(model_df)
            false_alarm_rate = model_df["false_alarm"].sum() / total
            event_missed_rate = model_df["event_missed"].sum() / total

            rows.append({
                "Model": model.replace("_", " "),
                "Recall": recall_avg,
                "F1 Score": f1_score_avg,
                "Delay": delay_avg,
                "False Alarms": false_alarm_rate,
                "Events Missed": event_missed_rate
            })

        table = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
        tables[network] = table

        for network, table in tables.items():
            latex_str = table.to_latex(
                index=False,
                float_format="{:.3f}".format,
                na_rep="-",
                caption=f"Results for network {network}",
                label=f"tab:results-{network}"
            )

        with open(f"results/table_{network}.tex", "w") as f:
            f.write(latex_str)

    return tables


print(create_results_summary_table("results/results_summary.csv"))