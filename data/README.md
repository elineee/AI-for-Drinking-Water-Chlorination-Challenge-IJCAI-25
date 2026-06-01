# Information about the data

Each `.csv` file corresponds to one time series of chlorine concentration lasting several days (18 days for Hanoi and Net1, and one week for CY-DBP).

Among the data, there are several types of files:

- The **"clean"** or **"no conta"** files represent time series with no contamination events. It serves for models that need training on uncontaminated data.
- The **"train"** files represent time series that contain one contamination event that has been used for models that need training on contaminated data.
- The **"test"** files represent time series that contain one contamination event that has been used for testing the models.

## Dataset specifications

### Net1

- All contaminated files have a contamination at node **22**.
- A timestep corresponds to **30 minutes**.

### Hanoi

- All contaminated files have a contamination at node **3**.
- A timestep corresponds to **30 minutes**.

### CY-DBP

- All contaminated files have a contamination at node **dist423**.
- A timestep corresponds to **5 minutes**.

### Low chlorine data

- All contaminated files have a contamination at node **dist423**.
- A timestep corresponds to **5 minutes**.

### Global warming and ageing network data

- The contamination was random.
- Each file is named after the node at which we observe a contamination.
- A timestep corresponds to **5 minutes**.

### Noise experiment

- The data for the noise experiment are included with the CY-DBP data.
- These files contain **"noisy"** in the file name.

## Important note

For each time series, the first **3 days** must be removed because the simulation starts from zero with no chlorine in the system, so it takes 3 days to stabilize.

Only the data for **global_warming** and **ageing_network** already have their three first days removed.