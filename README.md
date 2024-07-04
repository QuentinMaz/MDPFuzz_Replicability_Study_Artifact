# Policy Testing with MDPFuzz (Replicability Study): Artifact

This repository is the artifact of the paper *Policy Testing with MDPFuzz (Replicability Study)*.
Its content consists of a <ins>reproduction</ins> study and a <ins>replication</ins> study of the paper [MDPFuzz: testing models solving Markov decision processes](https://dl.acm.org/doi/abs/10.1145/3533767.3534388).
<!-- To that regard, the submission was already supported by two distinct code basis of the studies. -->
This present artifact's structure reflects this, as it is composed of two main folders, namely: `reproduction/` and `replication/`.
The very own nature of the work makes that it involves a tremendous amount of computation; yet we design the code such that the results should be similar if not identical to the ones in the paper.
We additionally include a demontration as well as step-by-step instructions to entirely reproduce the two studies.

## Getting Started

### Introduction

The studies consist in executing testing methods for Reinforcement Learning models (also known as *policies*) in different scenarios, later referred to as *case studies*.
Each method is executed several times (with differrent random seeds) in order to eventually compute statistical results.
The scenarios (also called *environments*) solved by the policies are quite diverse, but they mainly consist of games and system control problems such as [landing a spacecraft](https://gymnasium.farama.org/) or [driving a taxi](https://gymnasium.farama.org/environments/toy_text/taxi/).

### Requirements

The experiments solely use [Python](https://www.python.org/), and the dependencies of each case study are installed in virtual environments with the package manager [Conda](https://anaconda.org/).

We provide a Docker image which has already all the virtual environments installed.
As such, we invite the user to follow the software's instructions, which are detailed [here](https://docs.docker.com/engine/install/).

We detail how to manually install each virtual environment in the beginning of the corresponding `README` files in the case study subdirectories.
We have tested their functionality with a Ubuntu 20.04 system.
Even though a graphic card is not needed, we strongly recommend it for the case study *CARLA*, a driving simulator, which is by far the most time-consuming case to execute.
Similarly, we encourage the user to replicate the other use cases first, since we deem the *CARLA*'s experiments untractable (we needed around three months to setup and succesfully run all of them).

#### Running the experiments on your system
If you want to install the virtual environments on your local machine, the only requirement is to install Python as well as Conda, whose installation instructions can be found [here](https://www.python.org/downloads/) and [here](https://docs.anaconda.com/miniconda/#quick-command-line-install), respectively.
Your system should be ready for running the experiments once the command `conda` works on your system (you can quickly check that with `conda --version`).

#### Running the experiments inside the container
```bash
# Build the image
docker build -t artifact .
# Run the image iteractively
docker run -it artifact
```

If you are running a non-x86 system, e.g. Apple Silicon, you must add the target platform to the Docker commands.
This is necessary, because the reproduction requires an old Python version (3.7.4) which is not available for the aarch64 architecture.
```bash
# Build the image
docker build --platform linux/amd64 -t artifact .
# Run the image iteractively
docker run --platform linux/amd64 -it artifact
```

## Detailed Description

The testing methods evaluated, as well as the use cases, are implemented in dedicated sub folders, for the two studies (reproduction and replication).
They all include a `README` file that explains how to install the virtual environment (if you are not using the Docker image) and what commands to execute to replicate the experiments performed in the paper.
Overall, the artifact consists in:
 1) Running the methods (for a fixed amount of *time* in the reproduction study, and for a fixed amount of *iterations* in the replication study);
 2) Processing the log files to compute the performance of the methods (as the number of faults found over time/iteration).
 3) Plotting the results.

### Step-by-step Instructions

#### Reproduction Study

Navigate to the folder `reproduction/` and follow the `README` file.

#### Replication Study

Navigate to the folder `replication/` and follow the `README` file.

### Demonstration

As previously introduced, the artifact involves a lot of executions; the ones from the *CARLA* case being especially long.
Also, the reproduction study has intrisic time constraints, since its evaluation protocol consists in testing the *policies* for a fixed amount of time (2 hours of initialization + 12 hours of testing).
Therefore, we provide a short, small-scale experiment for demonstrating the functionality of the artifact.
This demonstration tests the two testing methods of the reproduction study (referred as *Fuzzer-O* and *MDPFuzz-O* in the paper) but only the use case *ACAS Xu* for a few minutes.

#### Instructions

If you are using the container, navigate to the *ACAS Xu* use case of the reproduction folder with `cd reproduction/ACAS_Xu` and activate the corresponding Python environment with `conda activate acas`.
If you prefer to install the latter on your system, follow the instructions there (make sure to activate the environment upon completion).
Then, execute each method for 10 minutes:
```python
# MDPFuzz-O
python my_simulate.py --seed 2020 --init_budget 2 --fuzz_budget 8
# Fuzzer-O
python my_simulate.py --seed 2020 --init_budget 2 --fuzz_budget 8 --no_coverage
```
Finally, process the logs and plot the number of faults found over time in a similar fashion as in the paper:
```
cd ..
python result_script.py
```
The plot is exported to the file `fault_discovery_plot.png`.
<!-- First, build the Docker image with `docker build -t demo .`. Then, run the image into a container with the command `docker run --rm -v .:/output demo`.
The command above ensures that container is automatically shut down once finished.
The results (both the image and the raw, log files) are exported in the current repository. -->

