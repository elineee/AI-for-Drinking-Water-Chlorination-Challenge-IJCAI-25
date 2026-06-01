# Generation of data for CY-DBP

For generating the data, two files are needed: a `.inp` file and a corresponding `.msx` file.

For this network, there are a total of **10 `.inp` files** and **10 `.msx` files**, each corresponding to different demand scenarios.

## Data generation under normal conditions

The code to generate the data under normal conditions for the CY-DBP network is in the file **"scenario_generation.ipynb"**.

The main parameters that can be changed are:

- The `.inp` and corresponding `.msx` file.
- The number of days the simulation lasts.
- The duration of the contamination event.
- The number of contamination events occurring during the simulation.
- The rate of the injection.
- The days between which an event can start.
- The node at which the contaminant is injected.

**"scenario_generation.ipynb"** contains further documentation about the generation with already specified values for the above parameters.

## Data generation under extreme conditions

The code to generate the data under extreme conditions for the CY-DBP network is in the file **"future_scenario_generator.py"**.

The main parameters that can be changed are:

- The `.inp` and corresponding `.msx` file.
- The number of days the simulation lasts.
- The duration of the contamination event.
- The number of contamination events occurring during the simulation.
- The rate of the injection.
- The days between which an event can start.
- The node at which the contaminant is injected.
- The `temperature_factor`: it varies between 0 and 1. The higher the number, the higher the temperature.
- The `ageing_factor`: it varies between 0 and 1. The higher the number, the higher the ageing factor of the network.
- The `urban_growth_factor`: it varies between 0 and 1. The higher the number, the higher the urban growth factor.

**"future_scenario_generator.py"** contains further documentation about the generation with already specified values for the above parameters.