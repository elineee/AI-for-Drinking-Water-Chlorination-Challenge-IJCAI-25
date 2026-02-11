from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
from traitlets import List, Tuple

    
from data_transformation import *
from model import AnomalyModel


class LOFModel(AnomalyModel):
    def __init__(self, config):
        super().__init__(config)
    
    
    def get_results(self) -> dict:
        clean_dfs, contaminated_dfs = self.load_datasets()
        
        # TODO : handle multiple clean/contaminated files, for now only one of each is handled
        X_train = create_features(clean_dfs[0], self.config.desinfectant.value, self.config.window_size)

        # TODO : handle multiple contaminants, for now only one contaminant is handled
        X_test = create_features(contaminated_dfs[0], self.config.desinfectant.value, self.config.window_size)

        # TODO : handle multiple contaminants, for now only one contaminant is handled
        y_true = calculate_labels(contaminated_dfs[0], self.config.contaminants[0].value, self.config.window_size)
        
        # TODO : voir pour mettre + de paramètres 
        n_neighbors = self.config.model_params.get("n_neighbors", 20)
        contamination = self.config.model_params.get("contamination", 0.1)
        
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=contamination)

        lof.fit(X_train)

        y_pred = lof.predict(X_test)
        
        return {
            "y_true": y_true,
            "y_pred": y_pred,
        }
    
    def load_and_filter(self, file_path: str, nodes: List[int]) -> pd.DataFrame:
        # TODO : add parameters contaminants when changed in function 
        df_all = change_data_format(file_path, self.config.contaminants, to_csv=False)  # returns rows with columns: timestep, node, chlorine_concentration, arsenic_concentration
        
        dfs = []
        
        if self.config.aggregate_method is None:
            dfs = []
            for node in nodes:
                df_node = get_data_for_one_node(df_all, node, to_csv=False)
                dfs.append(df_node)
        
        else: # TODO : handle if aggregation
            pass
        
        return dfs

    def load_datasets(self):
        """Return lists of dataframes: (clean_dfs, contaminated_dfs) aligned with provided file lists."""
        
        clean_dfs = []
        if self.config.example_files is not None:
            for fp in self.config.example_files:
                clean_dfs.extend(self.load_and_filter(fp, self.config.nodes))
                
        contaminated_dfs = []
        for fp in self.config.contaminated_files:
            contaminated_dfs.extend(self.load_and_filter(fp, self.config.nodes))
        
        return clean_dfs, contaminated_dfs

# clean_file = "..\data\data_arsenic\scada_data_no_contamination.csv"
# contaminated_file = "..\data\data_arsenic\scada_data_conta_22.csv"
# NODE = 22
# WINDOW_SIZE = 30
# # 15 score de 73, si 1, score de 40%, si 20, score de 67%, si 10, score de 70%, si 30, score de 76% (meilleur score)


# df_cleaned = change_data_format(clean_file, to_csv=False)
# df_cleaned_node = get_data_for_one_node(df_cleaned, NODE, to_csv=False)

# df_contaminated = change_data_format(contaminated_file, to_csv=False)
# df_contaminated_node = get_data_for_one_node(df_contaminated, NODE, to_csv=False)



        
# X_train = create_features(df_cleaned_node, "chlorine_concentration", WINDOW_SIZE)

# X_test = create_features(df_contaminated_node, "chlorine_concentration", WINDOW_SIZE)

# y_true = calculate_labels(df_contaminated_node, "arsenic_concentration", WINDOW_SIZE)

# anomalies = 0
# normal = 0
# for i in y_true:
#     if i == -1:
#         anomalies += 1
#     else:
#         normal += 1
# print("number of anomalies:", anomalies)
# print("number of normal samples:", normal)
    

# lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)

# lof.fit(X_train)

# y_pred = lof.predict(X_test)

# print("accuracy:", accuracy_score(y_true, y_pred))

# print("Matrice de confusion :")
# print(confusion_matrix(y_true, y_pred, labels=[1, -1]))