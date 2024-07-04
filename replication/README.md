# Replication Study of MDPFuzz

This repository contains the material of the replication study conducted in the paper *Replicability Study: Policy Testing with MDPFuzz*.
As such, some of the code is borrowed from the [original implementation of MDPFuzz](https://github.com/Qi-Pang/MDPFuzz).

## Running the experiments

Each folder contains the implementation of a case study, except `rl/`, which includes the *Bipedal Walker*, *Lunar Lander* and *Taxi* use-cases.

There, you will find details about how to manually install the Python virtual environments (needed if you are not using the Docker image).
If not, make sure to have the correct environment activated.

The three policy testing methods studied (*Fuzzer-R*, *MDPFuzz-R*, *Random Testing*) in `methods/`.

By default, the results of the executions are exported under `data_rq2/` and `data_rq3`.

## Postprocessing

Once experiments have been executed, the results can be extracted from the logs with the Python script `main.py`.
We recommend using the virtual environment `rl`, whose installations are detailed in the `README` file of the folder `rl/`.
The script computes the results, stores them under `results_rq2/` and `results_rq3/` and creates all the figures presented in the paper.

## Data Availability

The data used in the paper is available on [Zenodo](https://zenodo.org/records/10958452).

## Replication Study

The study aims to (RQ2) check the fault discovery ability of the fuzzers (*Fuzzer-R* and *MDPFuzz-R*) and to (RQ3) investigate the parameter sensibility of the latter (only *MDPFuzz* is parametrized).
We consider the (fixed) original use cases (see the reproduction study; see `reproduction/README`) and new ones, and, compared to the reproduction study, we include in the evaluation a random testing baseline.

Besides, we increase the robustness of the results by repeating every experiment 5 times (compared to 3 conducted in the reproduction study, something we did to follow the original experimental protocol of [MDPFuzz](https://github.com/Qi-Pang/MDPFuzz)).

## Additional Notes

In this study, the testing methods are run for a given number of **iterations** (5000 in total, among which 1000 are dedicated to initialize *Fuzzer-R* and *MDPFuzz-R*).
Therefore, we can't indicate the expected running time.


The parameter analysis of *MDPFuzz* (RQ3) first shows that the parameter $\tau$ has little impact on its performance (we explore 3 values: 0.01, 0.1 and 1.0), before considering a total of 20 configurations of its remaining parameters, $K$ and $\gamma$.

Therefore, for **each case study**, the parameter analysis requires **110** executions.
That's the reason why we definitively deem the *CARLA* case study untractable.