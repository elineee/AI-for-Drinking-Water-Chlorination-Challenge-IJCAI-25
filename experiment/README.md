# Types of data
There are three types of data in the folder `data`: 

- `data_hanoi`: data generated with the Hanoi network, small network whose chlorine concentration pattern is simple and very similar from one node to another.
- `data_net1`: data generated with the Net1 network, a very small network whose chlorine concentration pattern is complex.
- `data_cy`: data generated with the CY-DBP network, a large and more complex network, where not all nodes are relevant for anomaly detection.

Notes: 
- In the Hanoi and the CY-DBP network, contamination cannot be directly observed at the injection node itself.
- In the CY-DBP network, some nodes never receive chlorine, making anomaly detection impossible on these nodes.


# Running experiments 

Several `main_...` files are already provided with all experiments conducted in this work, in the folder `experiment`:  
- `main_hanoi`: experiments conducted on the Hanoi network. 
- `main_net1`: experiments conducted on the Net1 network. 
- `main_cy`: experiments conducted on the CY-DBP network. 
- `main_low_chlorine`: experiments conducted with low chlorine concentration, on the CY-DBP network.  
- `main_noise`: experiments conducted with noisy data, on the CY-DBP network.  
- `main_ageing_network`: experiments conducted with the simulation of an ageing network conditions, on the CY-DBP network.  
- `main_global_warming`: experiments conducted with the simulation of global warming conditions, on the CY-DBP network.  


To reproduce the experiments:
- Open the corresponding `main_...` file.
- Run the file.


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


# Structure of the folder

- `models/`: contains the implementation of all models.
- `data_transformation.py`: handles data preprocessing and transformation.
- `evaluation.py`: contains the evaluation metrics and functions.
- `experiment_config.py`: defines the experiment configuration structure.
- `experiment.py`: experiment runner.
- `utils.py`: utility functions.


# Example of a configuration
```python
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
```

# Names of models

Some model names in the code differ from those used in the thesis:

| Code name | Thesis name |
|---|---|
| CNN | SVR-CNN (variant 1) |
| CNN_windows | SVR-CNN (variant 2) |
| CNN_VAE | VAE-CNN (variant 1) |
| CNN_windows_VAE | VAE-CNN (variant 2) |
| CNN_multi_nodes | CNN multi-node |
| CNN_univariate | CNN univariate |
| VAE_classifier | VAE-Classifier |
| VAE | VAE |
| AE | AE |
| LSTM_AE | LSTM AE |
| SVR | SVR |
| one_class_SVM | One-Class SVM |
| LOF | LOF |
| isolation_forest | Isolation Forest |
