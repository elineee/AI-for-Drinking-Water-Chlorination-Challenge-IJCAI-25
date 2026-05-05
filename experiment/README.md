# Types of data
There are three types of data in the folder data: 

- `data_net1`: data generated with the Net1 network, a small network with few nodes.
- `data_hanoi`: data generated with the Hanoi network, a medium-sized network.
- `data_cy`: data generated with the CY-DBP network, a large and more complex network, where not all nodes are relevant for anomaly detection.

Notes: 
- In the Hanoi and the CY-DBP network, contamination cannot be directly observed at the injection node itself.
- In the CY-DBP network, some nodes never receive chlorine, making anomaly detection impossible on these nodes.



# Running experiments 

Experiments are launched by defining configurations in the corresponding `main_...` files.
Each configuration needs: 

- `config_name`: name of the experiment configuration.
- `example_files`: list of clean data files.
- `contaminated_files`: list of contaminated data files.
    For CNN models, several contaminated files are required. The last contaminated file is used for testing.
- `nodes`: list of nodes to analyze for contamination detection.
    For `data_net1` and `data_hanoi`, node ids are represented as strings containing only the node number. 
    For `data_cy`, node ids are represented as strings containing "dist" following by the node number.
- `window_size`: size of the window used by the model.
- `model_name`: selected model from the available ones in `ModelName`.
- `contaminants`: contaminant type.
    By default, it is arsenic (for `data_net1` and `data_hanoi`). For `data_cy`, it must be set to `pathogen`.



# Example of a configuration
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



# Experiments 

Several `main_...` files are already provided with all experiments conducted in this work.

To reproduce the experiments:
- Open the corresponding `main_...` file.
- Uncomment the desired experiment configurations.
- Run the file.

