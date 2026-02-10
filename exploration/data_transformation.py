import pandas as pd

def change_data_format(file_name, to_csv=True):
    "transform the data to have one row = one time step, the chlorine value, the arsenic value and the number of the node"
    df = pd.read_csv(file_name)
    
    df_cleaned = df.drop(columns=["pressure [meter] at 10","pressure [meter] at 11","pressure [meter] at 12","pressure [meter] at 13","pressure [meter] at 21","pressure [meter] at 22","pressure [meter] at 23","pressure [meter] at 31","pressure [meter] at 32","pressure [meter] at 9","pressure [meter] at 2","flow [cubic meter/hr] at 10","flow [cubic meter/hr] at 11","flow [cubic meter/hr] at 12","flow [cubic meter/hr] at 21","flow [cubic meter/hr] at 22","flow [cubic meter/hr] at 31","flow [cubic meter/hr] at 110","flow [cubic meter/hr] at 111","flow [cubic meter/hr] at 112","flow [cubic meter/hr] at 113","flow [cubic meter/hr] at 121","flow [cubic meter/hr] at 122","flow [cubic meter/hr] at 9"])

    new_data = {
        "timestep": [],
        "node": [],
        "chlorine_concentration": [],
        "arsenic_concentration": []
    }
    
    nodes = set()
    
    for column in df_cleaned.columns:
        number = get_node_number(column)
        nodes.add(number)
        
    for timestep, row in df_cleaned.iterrows():
        for node in nodes:
            chlorine_col = f'bulk_species_node [MG] at Chlorine @ {node}'
            arsenic_col = f'bulk_species_node [MG] at AsIII @ {node}'
            
            if chlorine_col in df_cleaned.columns and arsenic_col in df_cleaned.columns:
                new_data["timestep"].append(timestep)
                new_data["node"].append(node)
                new_data["chlorine_concentration"].append(row[chlorine_col])
                new_data["arsenic_concentration"].append(row[arsenic_col])
        
    new_df = pd.DataFrame(new_data)
    
    if to_csv:
        new_df.to_csv(file_name.replace(".csv", "_cleaned.csv"), index=False)
    
    return new_df
    

def get_node_number(column_name):
    "extract the number from the column name to get the node number"
    return int(column_name.split(" @ ")[1].split(" ")[0])

def get_data_for_one_node(file_name, node_number, to_csv=True):
    "keep only the data for one node and save it in a new csv file"
    df = pd.read_csv(file_name)
    
    new_data = {
        "timestep": [],
        "node": [],
        "chlorine_concentration": [],
        "arsenic_concentration": []
    }
    
    for _, row in df.iterrows():
        if row["node"] == node_number:
            new_data["timestep"].append(int(row["timestep"]))
            new_data["node"].append(int(row["node"]))
            new_data["chlorine_concentration"].append(row["chlorine_concentration"])
            new_data["arsenic_concentration"].append(row["arsenic_concentration"])
    
    new_df = pd.DataFrame(new_data)
    
    if to_csv:
        new_df.to_csv(file_name.replace(".csv", f"_node_{node_number}.csv"), index=False)
        
    return new_df

if __name__ == "__main__":
    # change_data_format(".\data\data_arsenic\scada_data_node_22.csv")
    get_data_for_one_node(".\data\data_arsenic\scada_data_node_22_cleaned.csv", 22)
    