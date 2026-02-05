"""
This file contains code for creating new (random) contamination scenarios.
"""

import random
import math
import numpy as np
from epyt_flow.simulation import ScenarioSimulator, EpanetConstants


def create_random_contamination_event(time_window: tuple[int, int],
                                      duration_interval: tuple[int, int], n_time_steps: int):
    """Create a random contamination event profile for three contaminants:
    - Pathogen (P)
    - Carbon Fraction Rapidly Available (C_FRA)
    - Carbon Slowly Readily Available (C_SRA)
    The contamination event is defined by a random start time within the given time window,
    and a random duration within the given duration interval.
    Args:
        time_window (tuple[int, int]): Time window (in time steps) within which the contamination event can start.
        duration_interval (tuple[int, int]): Duration interval (in time steps) for the contamination event.
        n_time_steps (int): Total number of time steps in the simulation.
    Returns:
        tuple: Three tuples, each containing the species ID and its corresponding contamination profile.
    """
    # Random point in time
    start_time = random.randint(time_window[0], time_window[1])

    # Random duration
    end_time = start_time + random.randint(duration_interval[0], duration_interval[1])

    # Random amount of contaminants, don't need to change that
    EV_log_min = math.log10(1.39e6)
    EV_log_max = math.log10(2.08e7)
    EV_conc = 10 ** (EV_log_min + random.uniform(0, 1) * (EV_log_max - EV_log_min))
    TOC = 140 + random.uniform(0, 1) * (250 - 140)
    C_FRA_fraction = 0.4
    C_SRA_fraction = 0.6
    # can change the rate : instensity of the injection
    rate = 100

    injection_conc_P = EV_conc * rate # rate = injection strength, can change it 
    injection_conc_C_FRA = C_FRA_fraction * TOC * rate
    injection_conc_C_SRA = C_SRA_fraction * TOC * rate

    profile_P = np.zeros(n_time_steps) # first, 0 at each time step, then we will add the contamination event
    profile_P[start_time:end_time] = injection_conc_P # add contamination event for corresonding timesteps

    profile_C_FRA = np.zeros(n_time_steps)
    profile_C_FRA[start_time:end_time] = injection_conc_C_FRA

    profile_C_SRA = np.zeros(n_time_steps)
    profile_C_SRA[start_time:end_time] = injection_conc_C_SRA

    # return lists of values for each time step, for each species that correspont to what we will inject at each time step, for each species
    return ("P", profile_P), ("C_FRA", profile_C_FRA), ("C_SRA", profile_C_SRA)


if __name__ == "__main__":
    f_inp_in = "CY-DBP_competition_stream_competition_6days_0.inp"  # 6 days long scenario; file for topology and hydraulics
    f_msx_in = "CY-DBP_competition_stream_competition_6days_0.msx" # file for water quality and species
    #f_inp_in = "CY-DBP_competition_stream_competition_365days.inp"   # 365 days long scenario
    #f_msx_in = "CY-DBP_competition_stream_competition_365days.msx"

    ########################################################################
    # Parameters of the contamination events, what we are going to change 
    duration_interval = (60, 480)    # 60 min - 480 min long contamination : duration of the contamination event 
    n_contamination_events = 2  # Two random contamination events
    time_window = (2, 5)        # Contamination event between the third and sixth day
    #n_contamination_events = 5  # Five random contamination events in the entire year
    #time_window = (5, 350)      # Contamination event between the sixth and 350th day
    ########################################################################

    # Create scenario
    with ScenarioSimulator(f_inp_in=f_inp_in, f_msx_in=f_msx_in) as scenario:
        # Setup time intervals
        hyd_time_step = scenario.get_hydraulic_time_step()  # Usually 5min time steps (so 5*60 seconds )
        steps_per_day = (24 * 60 * 60) / hyd_time_step
        time_window = (time_window[0] * steps_per_day, time_window[1] * steps_per_day) 
        duration_interval = ((duration_interval[0] * 60) / hyd_time_step,
                             (duration_interval[1] * 60) / hyd_time_step)
        n_time_steps = int(scenario.get_simulation_duration() / hyd_time_step)

        # Add random contamination events
        all_junctions = scenario.get_topology().get_all_junctions() # get all nodes where we can add contamination events
        contamination_patterns = [] 
        # can control at specific nodes 
        for _ in range(n_contamination_events):
            node_id = random.choice(all_junctions) # choose random node to add contamination event

            contaminants_profiles = create_random_contamination_event(time_window, duration_interval,
                                                                      n_time_steps) # get contamination profiles for each species
            for species_id, pattern in contaminants_profiles:
                contamination_patterns.append(pattern) # pattern is the list of values at each time step for each species
                scenario.add_species_injection_source(species_id, node_id, pattern, # inject contamination at the chosen node following the pattern
                                                      EpanetConstants.EN_MASS) # different types of injection to simulate, can choose, here use just MASS

        # Compute labels -- for each time step, 1 if a contamination present, 0 otherwise
        y = np.sum(contamination_patterns, axis=0) != 0
        print(y.shape)  # TODO: Export labels

        # TEST: run simulation
        # place sensors at all nodes but can choose specific nodes too
        scenario.place_bulk_species_node_sensors_everywhere(["P", "CL2"]) # choose what to mesure with the sensors (can also choose where to measure it)
        # mesure also chlorine concenteration to detect contamination later 
        # chlorine is used as a proxy to detect contamination 
        scada_data = scenario.run_simulation(verbose=True) # run hydraulics then water quality 
        # resultas stored as scada data
        # plot the species concentration at all nodes, can be more specific (choose the nodes)
        scada_data.plot_bulk_species_node_concentration({"CL2": ["dist71"]}) # here, plot chlorine concentration at node dist71
        
        # Export results to use later 
        scada_data.to_numpy_file("scada_data.npz", export_raw_data=False)
        df = scada_data.to_pandas_dataframe(export_raw_data=False)
        df.to_csv("scada_data.csv", index=False)
    
