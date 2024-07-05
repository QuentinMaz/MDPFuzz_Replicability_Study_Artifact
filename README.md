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

The experiments solely use [Python](https://www.python.org/), and the dependencies of each case study is installed in virtual environments with the package manager Conda.
We provide a Docker image which has already all the virtual environments installed (except the one for the use case *CARLA*; see below).
As such, we invite the user to follow the software's instructions, which are detailed [here](https://docs.docker.com/engine/install/).
We also detail how to install each virtual environment as the beginning of the corresponding `README` files in case the Docker image is not used.
To that regard, we tested them with a Ubuntu 20.04 system.

#### Note on *CARLA*
This use case is by far the most time-consuming to execute and the most difficult to setup and replicate.
Besides the very long execution times, this driving simulator uses a client-server communication, making the Docker image not compatible.
Therefore, to replicate the related experiments, you will have to install the environment on your local system.
Similarly, for that use case we strongly recommend a graphic card.

Therefore, we encourage the user to use the Docker image and replicate the other use cases, since we deem the *CARLA*'s experiments untractable (we needed around three months to setup and succesfully run all of them).

#### Running the experiments on your system

If you want to install the virtual environments on your local machine, the only requirement for is to install Python as well as Conda, whose installation instructions can be found [here](https://www.python.org/downloads/) and [here](https://docs.anaconda.com/miniconda/#quick-command-line-install), respectively.
Your system should be ready for running the experiments once the command `conda` works on your system (you can quickly check that with `conda --version`).

<!-- #### Running the experiments inside the container
If you want to use the provided Docker image, run the following commands:
```bash
# build the image
docker build -t artifact .
# run the image iteractively
docker run -it artifact
``` -->

If you are running a non-x86 system, e.g. Apple Silicon, you must add the target platform to the Docker commands.
This is necessary, because the reproduction requires an old Python version (3.7.4) which is not available for the aarch64 architecture.
You must further ensure that the Docker VM has sufficient memory, at least 8GB. See [this torch issue](https://github.com/pytorch/pytorch/issues/1022) and [this stackoverflow question](https://stackoverflow.com/questions/44533319/how-to-assign-more-memory-to-docker-container) for context.
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

Run a container named `exp_container` with a connection to your local file system with:
```
docker run --name exp_container -v .:/output -it artifact
```
You can stop the container with `exit` (once in the terminal).
To restart the container (since you won't be able to run all the experiments at once):
```
docker start exp_container
docker exec -it exp_container /bin/bash
```


#### Reproduction Study

Navigate to the folder `reproduction/` and follow the `README` file.

#### Replication Study

Navigate to the folder `replication/` and follow the `README` file.

The two studies are independent: you can start replicating them in any order.

## Demonstration

As previously introduced, the artifact involves a lot of executions; the ones from the *CARLA* case being especially long.
Also, the reproduction study has intrisic time constraints, since its evaluation protocol consists in testing the *policies* for a fixed amount of time (2 hours of initialization + 12 hours of testing).
Therefore, we provide a short, small-scale experiment for demonstrating the functionality of the artifact.
This demonstration tests the two testing methods of the reproduction study (referred as *Fuzzer-O* and *MDPFuzz-O* in the paper) but only the use case *ACAS Xu* for a few minutes.

#### Instructions

If you are using the container, start it such that will be automatically removed once the terminal is closed, with a connection to your local file system with:
```
docker run --rm -v .:/output -it artifact
```
Then, navigate to the *ACAS Xu* use case of the reproduction folder and activate the corresponding Python environment:
```bash
(in the docker container)
cd /src/reproduction/ACAS
conda activate acas
```
If you prefer to install the latter on your system locally, follow the instructions in the use case reproduction folder (make sure to activate the environment upon completion).

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
If you are using the container, export the file to your local system and exit the container:
```
cp fault_discovery_plot.png /output
exit
```
The file should appear in you current directory.
The plot function is designed to work with the final results: therefore, the time axis (horizontal) does not correspond to the time of the demonstration (10 minutes).
<!-- First, build the Docker image with `docker build -t demo .`. Then, run the image into a container with the command `docker run --rm -v .:/output demo`.
The command above ensures that container is automatically shut down once finished.
The results (both the image and the raw, log files) are exported in the current repository. -->

