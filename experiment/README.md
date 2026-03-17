# Types of data
There are three types of data in the folder data: 

- data_arsenic : a small configuration with few nodes 
- data_hanoi : a middle configuration
- data_compet : a big configuration, but where each node is not relevant. 

Note: In the hanoi and the compet data, you can not observe the contamination on the node where it happens. 

# Launch a configuration

To launch a configuration, we need to write it in the configs of the main file. 
You have to specify: 
- a configuration name
- example files (of clean data)
- contaminated files (of contaminated data). 
    In the case of CNN, you need to specify several contaminated files. 
    The last one is the one used for testing. 
- nodes: name of the nodes that you want analyze. 
    In the case of arsenic data and hanoi data, the node id is a string with the number of the node. 
    In the case of competition data, the node id is a string with "dist" followed by the number of the node. For example, "dist11"
- the window size 
- the model name (among the one available in ModelName)
- contaminants: by default, arsenic. For the competition data, you must specify pathogen. 

# Example of config
'''
    ExperimentConfig(
    config_name="CNN_VAE",
    example_files=CLEAN_FILES,
    contaminated_files=CONTAMINATED_FILES,
    nodes=["dist33"],
    window_size=350, 
    model_name=ModelName.CNN_VAE,
    model_params={},
    contaminants=[ContaminationType.PATHOGEN]
)
'''