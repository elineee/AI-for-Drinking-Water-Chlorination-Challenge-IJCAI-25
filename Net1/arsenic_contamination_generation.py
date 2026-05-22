"""
Add a contamination event representing arsenic injection into a simple water distribution network/
This file is based on the given example: https://epyt-flow.readthedocs.io/en/stable/examples/arsenic_contamination.html. 
"""
import numpy as np
from epyt_flow.data.benchmarks import load_leakdb_scenarios
from epyt_flow.simulation import ScenarioSimulator, EpanetConstants, ScenarioConfig
from epyt_flow.simulation.events import SpeciesInjectionEvent
from epyt_flow.utils import to_seconds
import random


def generate_contamination_event(species: str, contamination_node: str, injection_amount: float, source_type: int, start_day: int, duration_days: int):
    """
    Generates a contamination event, which consists of an injection of a species at a node in the network.
    The contamination event is defined as an injection of a species at a specific node in the network, with a specified profile and duration.

    Parameters:
    species: id of the species being injected
    contamination_node: node id where the contamination occurs
    start_day: day of the start of the contamination event 
    duration_days: duration of the contamination event in days 
    injection_amount: amount of species injected 

    Returns:
    contamination_event: a SpeciesInjectionEvent object that represents the contamination event

    """
    
    contamination_event = SpeciesInjectionEvent(
        species_id=species,
        node_id=contamination_node,
        profile=np.array([injection_amount]),
        source_type=source_type,
        start_time=to_seconds(days=start_day),
        end_time=to_seconds(days=start_day + duration_days)
    )

    return contamination_event


if __name__ == "__main__":
    # Create a new scenario based on the first Net1 LeakDB scenario --
    # we add an additional EPANET-MSX configuration file
    config, = load_leakdb_scenarios(scenarios_id=["1"], use_net1=True)
    config = ScenarioConfig(scenario_config=config,
                            f_msx_in="exploration/arsenic_contamination.msx")

    with ScenarioSimulator(scenario_config=config) as sim:
        # Set simulation duration to 21 days
        sim.set_general_parameters(simulation_duration=to_seconds(days=21))

        # Place some chlorine sensors and also keep track of the contaminant 
        cl_sensor_locations = ["10", "11", "12", "13", "21", "22", "23", "31", "32"]
        all_nodes = sim.sensor_config.nodes
        sim.set_bulk_species_node_sensors({"Chlorine": cl_sensor_locations,
                                           "AsIII": all_nodes})   # Arsenite

        # Create a random contamination event for arsenic (AsIII) at a random node and with random parameters
        species = "AsIII"
        contamination_node = random.choice(cl_sensor_locations)
        start_day = random.randint(0, 7)  
        duration_days = random.randint(1, 5)  
        injection_amount = random.randint(1, 1000000)
        source_type = EpanetConstants.EN_MASS

        contamination_event = generate_contamination_event(species, contamination_node, injection_amount, source_type, start_day, duration_days)
        print(f"Generated contamination event: {contamination_event}")
        sim.add_system_event(contamination_event)

        # Run simulation
        scada_data = sim.run_simulation()

        # Export SCADA results 
        df = scada_data.to_pandas_dataframe(export_raw_data=False)
        df.to_csv("scada_data.csv", index=False)
