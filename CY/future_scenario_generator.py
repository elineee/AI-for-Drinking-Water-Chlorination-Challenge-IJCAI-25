"""
This file contains code for creating new (random) contamination scenarios for more extreme case scenario (e.g, global warming, ageing network)
"""
import random
#random.seed(424242)
import matplotlib.pyplot as plt
import math
import numpy as np
import os
from epyt_flow.simulation import ScenarioSimulator, EpanetConstants, ScadaData
from epyt_flow.utils import to_seconds, plot_timeseries_data
from nrw_model import NRWClass


def create_random_contamination_event(time_window: tuple[int, int],
                                      duration_interval: tuple[int, int], n_time_steps: int,
                                      C_FRA_fraction: float, C_SRA_fraction: float, TOC_bounds: np.ndarray):
    # Random point in time
    start_time = random.randint(time_window[0], time_window[1])

    # Random duration
    end_time = start_time + random.randint(duration_interval[0], duration_interval[1])

    # Random amount of contaminants
    EV_log_min = math.log10(1.39e6)
    EV_log_max = math.log10(2.08e7)
    EV_conc = 10 ** (EV_log_min + random.uniform(0, 1) * (EV_log_max - EV_log_min))
    TOC = TOC_bounds[0] + random.uniform(0, 1) * (TOC_bounds[1] - TOC_bounds[0])
    rate = 100

    injection_conc_P = EV_conc * rate
    injection_conc_C_FRA = C_FRA_fraction * TOC * rate
    injection_conc_C_SRA = C_SRA_fraction * TOC * rate

    profile_P = np.zeros(n_time_steps)
    profile_P[start_time:end_time] = injection_conc_P

    profile_C_FRA = np.zeros(n_time_steps)
    profile_C_FRA[start_time:end_time] = injection_conc_C_FRA

    profile_C_SRA = np.zeros(n_time_steps)
    profile_C_SRA[start_time:end_time] = injection_conc_C_SRA

    return ("P", profile_P), ("C_FRA", profile_C_FRA), ("C_SRA", profile_C_SRA)


def generate_scenario_parameters(temperature_factor: float = 1.e-3, ageing_factor: float = 1.e-3,
                                 urban_growth_factor: float = 1.e-3) -> dict:
    """
    Temperature factor: Daily household water usage icreases (i.e., increase peak demand), and TOC increases as well
    Ageing factor: The aging of pipes increases the concentration of fast-reacting agents, and increases the number of leakages
    Urban population growth: Water cosumption increases, reduces groundwater -> more surface water is used, which has higher TOC and lower Slow:Fast Ratio
    Hydro-climatic extremes: more storms: More and stronger waste water contaminations at multiple points!
    """
    params = {"C_FRA_fraction": 0.4, "C_SRA_fraction": 0.6,
              "TOC_bounds": np.array([1, 5]), "TOC_contamin_bounds": np.array([140, 250]),
              "peak_demand_factor": 1, "nrw_age_class": NRWClass.A}

    # Urban population factor -> ground water / surface water
    # Aging -> more fast-reacting agents
    params["C_FRA_fraction"] = .4 * math.pow((.8 / .4), urban_growth_factor * ageing_factor)
    params["C_SRA_fraction"] = 1. - params["C_FRA_fraction"]

    # Temperature factor & surface water usage (due to urban population growth) -> TOC concentration
    toc_contamin_inc = 10 * math.pow(10, temperature_factor * urban_growth_factor)
    params["TOC_contamin_bounds"] = params["TOC_contamin_bounds"] + toc_contamin_inc

    toc_inc = math.pow(1.25, temperature_factor * urban_growth_factor)
    params["TOC_bounds"] = params["TOC_bounds"] * toc_inc
    #print(params["TOC_bounds"])

    # Temperatur factor -> peak demand increase factor (4% demand increase per 1 degree celcius -- we consider up to 3 degree celcius)
    params["peak_demand_factor"] = math.pow(1.12, temperature_factor)

    # Ageing & urban population growth -> non-revenue water (i.e., leakages)
    params["nrw_age_class"] = NRWClass.determine_class((ageing_factor * urban_growth_factor) * 60)

    # Contmination events
    params["contam_duration_interval"] = (1440, 2880)    # 60 min - 1440 min long contamination
    params["n_contam_events"] = 1
    params["contam_time_window"] = (3, 8)   # Contamination event between the third and eighth day

    return params


class ScenarioGenerator():
    """
    TODO: Docstring
    """
    def __init__(self, f_inp_in: str, f_msx_in: str,
                 temperature_factor: float, # can change,  bteween 0 and 1
                 ageing_factor: float, # can chan ge,  bteween 0 and 1
                 urban_growth_factor: float): # can change,  bteween 0 and 1
        self._f_inp_in = f_inp_in
        self._f_msx_in = f_msx_in
        self._s_params = generate_scenario_parameters(temperature_factor, ageing_factor, urban_growth_factor)

    def _set_toc_reservoir(self, epanet_api):
        toc_bounds = self._s_params["TOC_bounds"]
        reservoir_toc = toc_bounds[0] + (toc_bounds[1] - toc_bounds[0]) * random.random()

        pat_idx = epanet_api.MSXgetindex(EpanetConstants.MSX_PATTERN, "C_SRA_REPAT")
        epanet_api.MSXsetsource(epanet_api.get_node_idx("WTP"),
                                epanet_api.get_msx_species_idx("C_SRA"),
                                EpanetConstants.EN_SETPOINT, reservoir_toc, pat_idx)

    def _adjust_peak_demands(self, epanet_api):
        dmd_pat_factor = self._s_params["peak_demand_factor"]

        for pat_idx, pat_id in enumerate(epanet_api.get_all_patterns_id()):
            if not pat_id.startswith("P-Res"):
                continue

            pat_values = epanet_api.get_pattern(pat_idx)

            # Adjust peak demands (i.e., 75 percentile)
            threshold = np.percentile(pat_values, 75)
            for i in range(len(pat_values)): 
                if pat_values[i] >= threshold:
                    pat_values[i] *= dmd_pat_factor

            epanet_api.set_pattern(pat_idx, pat_values)

    def _set_background_leakages(self, scenario):
        pipes_km = 0
        topo = scenario.get_topology()
        for link_id, _ in topo.get_all_links():
            pipes_km += topo.get_link_info(link_id)["length"] * .001

        nrw_class = self._s_params["nrw_age_class"]
        nrw = pipes_km * nrw_class.sample_demand(n_points=7, RNG=np.random.default_rng()) # m^3/day
        nrw = nrw / 24. # m^3/hour

        n_junctions = len(scenario.epanet_api.get_all_junctions_id())
        nrw_dmd_per_junc = nrw / n_junctions

        # Add NRW demand to base demand
        for junc_id in scenario.epanet_api.get_all_junctions_id():
            node_idx = scenario.epanet_api.get_node_idx(junc_id)
            base_demand = scenario.epanet_api.get_node_base_demand(node_idx) + \
                np.random.choice(nrw_dmd_per_junc)
            scenario.epanet_api.setbasedemand(node_idx, 1, base_demand)

    def _create_random_contamination_event(self, node_id: str, scenario):
        time_window = (4, 7)
        duration_interval = (500, 1000)

        hyd_time_step = scenario.get_hydraulic_time_step()  # Usually 5min time steps
        steps_per_day = (24 * 60 * 60) / hyd_time_step
        time_window = (time_window[0] * steps_per_day, time_window[1] * steps_per_day)
        duration_interval = ((duration_interval[0] * 60) / hyd_time_step,
                             (duration_interval[1] * 60) / hyd_time_step)
        n_time_steps = int(scenario.get_simulation_duration() / hyd_time_step)

        # Random point in time
        start_time = random.randint(time_window[0], time_window[1])

        # Random duration
        end_time = start_time + random.randint(duration_interval[0], duration_interval[1])

        # Random amount of contaminants
        EV_log_min = math.log10(1.39e6)
        EV_log_max = math.log10(2.08e7)
        EV_conc = 10 ** (EV_log_min + random.uniform(0, 1) * (EV_log_max - EV_log_min))

        TOC_bounds = self._s_params["TOC_contamin_bounds"]
        TOC = TOC_bounds[0] + random.uniform(0, 1) * (TOC_bounds[1] - TOC_bounds[0])

        C_FRA_fraction = self._s_params["C_FRA_fraction"]
        C_SRA_fraction = self._s_params["C_SRA_fraction"]

        rate = 1000

        print("ev_conc", EV_conc)
        injection_conc_P = EV_conc * rate
        injection_conc_C_FRA = C_FRA_fraction * TOC * rate
        injection_conc_C_SRA = C_SRA_fraction * TOC * rate

        # Random place of contamination
        all_junctions = scenario.get_topology().get_all_junctions()
        node_id = random.choice(all_junctions)

        profile_P = np.zeros(n_time_steps)
        print("start_time", start_time, "end_time", end_time)
        profile_P[start_time:end_time] = injection_conc_P
        print(profile_P[start_time:end_time])
        scenario.add_species_injection_source("P", node_id, profile_P, EpanetConstants.MSX_MASS)

        profile_C_FRA = np.zeros(n_time_steps)
        profile_C_FRA[start_time:end_time] = injection_conc_C_FRA
        scenario.add_species_injection_source("C_FRA", node_id, profile_C_FRA, EpanetConstants.MSX_MASS)

        profile_C_SRA = np.zeros(n_time_steps)
        profile_C_SRA[start_time:end_time] = injection_conc_C_SRA
        scenario.add_species_injection_source("C_SRA", node_id, profile_C_SRA, EpanetConstants.MSX_MASS)

    def run_simulation(self, verbose: bool = True) -> ScadaData:
        with ScenarioSimulator(f_inp_in=self._f_inp_in, f_msx_in=self._f_msx_in) as scenario:
            skip_days = 3

            # General scenario configuration
            scenario.set_general_parameters(simulation_duration=to_seconds(days=9)) # Note that only demands for 6 days are given, however, since we discard the first 3 days we can stretch the simulation to 9 days in total without having repeating patterns
            self._set_toc_reservoir(scenario.epanet_api)
            self._adjust_peak_demands(scenario.epanet_api)
            self._set_background_leakages(scenario)

            # Sensor placement
            scenario.place_demand_sensors_everywhere()
            scenario.place_flow_sensors_everywhere()
            scenario.place_bulk_species_node_sensors_everywhere(["P", "CL2", "C_FRA", "C_SRA"])
            scenario.place_bulk_species_link_sensors_everywhere(["P", "CL2", "C_FRA", "C_SRA"])

            # Run simulations without any contamination events
            hyd_file = "tmp.hyd" 
            scada_data_clean = scenario.run_simulation(verbose=verbose, hyd_export=hyd_file)
            # scada_data_clean = ""

            # Add contamination events
            all_junctions = scenario.get_topology().get_all_junctions()

            for _ in range(self._s_params["n_contam_events"]):
                node_id = random.choice(all_junctions)
                self._create_random_contamination_event(node_id, scenario)

            # Run final simulation and discard first few days
            scada_data = scenario.run_advanced_quality_simulation(hyd_file_in=hyd_file, verbose=verbose)
            scada_data.join(scada_data_clean)

            return scada_data_clean, scada_data


if __name__ == "__main__":
    for i in range(1000):
        # Get the directory of this script to resolve file paths correctly
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        f_inp_in = os.path.join(script_dir, "CY-DBP_competition_stream_competition_6days_0.inp")  # 6 days long scenario
        f_msx_in = os.path.join(script_dir, "CY-DBP_competition_stream_competition_6days_0.msx")

        # Verify that the files exist before running
        if not os.path.exists(f_msx_in):
            raise FileNotFoundError(f"MSX file not found: {f_msx_in}")
        if not os.path.exists(f_inp_in):
            raise FileNotFoundError(f"INP file not found: {f_inp_in}")

        # use that for robustness and create extreme scenario 
        # gen = ScenarioGenerator(f_inp_in, f_msx_in,
        #                         temperature_factor=0.001, ageing_factor=0.005,
        #                         urban_growth_factor=0.03) # global warming, network not maintained (scenario for robustness)
        
        # Ageing network scenario
        # gen = ScenarioGenerator(f_inp_in, f_msx_in,
        #                         temperature_factor=0.001, ageing_factor=0.99,
        #                         urban_growth_factor=0.03) # global warming, network not maintained (scenario for robustness)
        
        # global warming scenario
        gen = ScenarioGenerator(f_inp_in, f_msx_in,
                                temperature_factor=0.99, ageing_factor=0.005,
                                urban_growth_factor=0.03)

        scada_data_clean, scada_data_contam = gen.run_simulation()

        all_junctions = scada_data_contam.network_topo.get_all_junctions() 
        
        df = scada_data_contam.to_pandas_dataframe(export_raw_data=False)
        df_clean = scada_data_clean.to_pandas_dataframe(export_raw_data=False)
        
        conta = False
        nodes = [ "dist606", "dist1332", "dist1915"]
        # Check if there is a significant increase in chlorine concentration at the nodes we're interested in (i.e., dist606, dist1332, dist1915)
        for column in df.columns: 
            for node in nodes:
                if node in column and "at CL2 @" in column:
                    conta_values = df[column]
                    clean_values = df_clean[column]
                    resulting_values = clean_values - conta_values
                    print("resulting_values", resulting_values)
                    if np.any(resulting_values > 0.05):
                        # scada_data_contam.plot_bulk_species_node_concentration({"CL2": [node]})
                        # scada_data_clean.plot_bulk_species_node_concentration({"CL2": [node]})
                        conta = True
                        conta_node = node
                        print("Contamination event detected in column", column)
            
            
        # Rename columns to match the format of the original data (e.g., replace [CUSTOM UNIT] with [MG], and CL2 with Chlorine)
        for column in df.columns:
            if "[CUSTOM UNIT]" in column:
                new_column_name = column.replace("[CUSTOM UNIT]", "[MG]")
                df.rename(columns={column: new_column_name}, inplace=True)
            if "CL2" in column:
                new_column_name = column.replace("CL2", "Chlorine")
                df.rename(columns={column: new_column_name}, inplace=True)
        
        # Save the contaminated data to a new CSV file if a contamination event was detected at one of the nodes of interest
        if conta:
            df.to_csv(f"scada_data_test_{conta_node}_{i}.csv", index=False)