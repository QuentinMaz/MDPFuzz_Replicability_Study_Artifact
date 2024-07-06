# Reproduction Study

This repository contains the material of the reproduction study conducted in the paper *Policy Testing with MDPFuzz (Replicability Study)*.
As such, some of the code is borrowed from the [original implementation of MDPFuzz](https://github.com/Qi-Pang/MDPFuzz).

## Running the experiments

The repository is structured as the original one, where the case studies are implemented in dedicated folders.
Navigate to each of them and follow the instructions to run the methods.
If you are not using the Docker image, they detail how to manually install the Python virtual environments.
If not, make sure to have the correct environment activated as well as being in the correct folder before proceeding the experiments.
<!-- We had to make changes in the original code and updated the README files accordingly.
Nevertheless, we kept the original instructions (`ORIGINAL_README.md`). -->
By default, the executions log the data under `data/`.

## Postprocessing

Once experiments have been executed, the results can be extracted from the logs with the Python script `result_script.py`.
We recommend using either the virtual environment `RLWalk` -- installed for the *Bipedal Walker* case study (see the instructions in `Bipedal_Walker/`) -- or `acas` (see the instructions in `ACAS_Xu/`).
The script computes the results, stores them in `results/` and creates the figure presented in the paper (`fault_discovery_plot.png`) in the current directory.

If you are using the Docker image, don't forget to copy the file to your local system with:
```bash
(in the folder reproduction/)
conda activate acas
python result_script.py
cp fault_discovery_plot.png /output
```

<!-- ## Data Availability

The data used in the paper is available on [Zenodo](https://zenodo.org/records/10910437). -->

## Additional Notes

The experiments all share the same protocol, which consists of **2 hours** of initialization and **12 hours** of testing.
As such, unless the use cases are run in parallel, this entire study could take several days to complete.
The amount of data generated should be at most 2.0 GB.
As introduced previously, we strongly recommend avoiding the use case *CARLA* unless the user wants to *exactly* replicate the study.
We remind that *CARLA* cannot be run inside the container and, as such, the entire artifact has to be replicated on your system locally (or copy all the data stored in `data/` with `cp -r data/ /output`).
We provide convenient bash scripts to execute the experiments in parallel, called `launch_experiments.sh`, for each case studied.
To use them though, it might be required to make the script executable, which can be done with `chmod +x launch_experiments.sh`.

## Data Availability

If you can't (or don't want to) reproduce all the experiments, you can download [the data we uploaded on Zenodo in the original submission](https://zenodo.org/records/10910437).

### Instructions

Download (the data using the link below) and copy-paste it in `data/`.
Then, install the virtual environment `RLWalk`, whose instructions are detailed in `Bipedal_Walker/`.
Finally, activate the latter and run the Python script mentioned above:
```
conda activate RLWalk
python result_script.py
```