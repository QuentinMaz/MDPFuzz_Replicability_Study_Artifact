import time
import numpy as np
import pandas as pd

from typing import List, Tuple, Dict


def compute_statistics(data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Helper that computes statistical results from a set of results.'''
    if not isinstance(data, np.ndarray):
        data: np.ndarray = np.array(data)
    y = np.median(data, axis=0)
    perc_25 = np.percentile(data, 25, axis=0)
    perc_75 = np.percentile(data, 75, axis=0)
    return y, perc_25, perc_75


########## Fault over time ##########


def count_distinct_faults_over_time(df_list: List[pd.DataFrame], time_step: float = 1.0):
    '''
    Retuns the accumulation of the number of faults per @time_step.
    '''
    results = []
    for df in df_list:
        oracles = df['oracle'].to_numpy()
        times = df['time'].to_numpy()

        time = time_step
        fault_accumulator = []
        num_faults = 0

        for o, t in zip(oracles, times):
            # if fault and time lower than current time
            if o and t <= time:
                # counts
                num_faults += 1
            # if time is exceeded
            elif t > time:
                # accumulate, updates time and counts possible fault
                fault_accumulator.append(num_faults)
                time += time_step
                num_faults += int(o)
        # considers the last time which is always above the last time step
        fault_accumulator.append(num_faults)
        results.append(np.array(fault_accumulator))
    return results


def count_distinct_faults_in_redundant_data_over_time(df_list: List[pd.DataFrame], time_step: float = 1.0):
    '''
    Counts the number of faults where the inputs can be redundant and executions stochastic.
    The latter case corresponds to the CARLA and Bipedal Walker use-cases.
    As such, redundant inputs are not discarded.
    Instead, the function stores the fault-revealing inputs and counts a fault if the current input has not been already recorded.
    Indeed, because of stochasticity, an input already evaluated can still find a fault.
    '''
    results = []
    for df in df_list:
        inputs = np.vstack(df['input'])
        oracles = df['oracle'].to_numpy()
        times = df['time'].to_numpy()

        fault_revealing_inputs = []
        fault_accumulator = []
        time = time_step
        num_faults = 0

        for i, o, t in zip(inputs, oracles, times):
            if o and t <= time:
                tmp = i.tolist()
                try:
                    index = fault_revealing_inputs.index(tmp)
                    print(f'Already counted! (index: {index})')
                except:
                    num_faults += 1
                    fault_revealing_inputs.append(tmp)
            elif t > time:
                fault_accumulator.append(num_faults)
                time += time_step
                if o and (not i.tolist() in fault_revealing_inputs):
                    num_faults += 1
                    fault_revealing_inputs.append(i.tolist())

        fault_accumulator.append(num_faults)
        print(f'Found {num_faults} faults in total and detects {len(fault_revealing_inputs)} fault-revealing inputs.')
        results.append(np.array(fault_accumulator))
    return results


# main one
def compute_fault_discovery_results(df_list: List[pd.DataFrame]) -> Tuple[np.ndarray]:
    '''
    Processes experimental data for a single use-case (as a list of pd.DataFrames) and computes the results.
    Precisely, the processing consists in counting non-redundant faults.
    To ease plotting, the results are extended to the longest execution.
    '''
    inputs = []
    for df in df_list:
        inputs.append(np.vstack(df['input']))
        # oracles.append(df['oracle'].to_numpy())
    t0 = time.time()
    if not np.all([len(arr) == len(np.unique(arr, axis=0)) for arr in inputs]):
        print('WARNING: redundant inputs found.')
        # tmp = count_faults_stochastic_executions(inputs, oracles)
        tmp = count_distinct_faults_in_redundant_data_over_time(df_list)
    else:
        # tmp = compute_faults_distinct_inputs(oracles)
        tmp = count_distinct_faults_over_time(df_list)
    max_length = np.max([len(l) for l in tmp])
    print(f'process time: {(time.time() - t0):.2f}s. Maximum duration found: {max_length}s.')
    for i in range(len(tmp)):
        arr = tmp[i]
        last_value = arr[-1]
        arr_length = len(arr)
        if arr_length < max_length:
            tmp[i] = np.array([v for v in arr] + [last_value for _ in range(max_length - len(arr))])
    return compute_statistics(tmp)


# main one for multiple results
def compute_statistical_fault_discovery_results(data: List[List[pd.DataFrame]]) -> List[Tuple[np.ndarray]]:
    '''Aggregates statistical results for fault discovery by calling compute_fault_discovery_results(.) for each DataFrame list.'''
    results = []
    for df_list in data:
        results.append(compute_fault_discovery_results(df_list))
    return results


########## Iterations over time ##########

##### Needed to know when a method #####
##### executes as many test cases  #####
##### as the other one.            #####


def aggregate_iterations(df_list: List[pd.DataFrame], time_step: float = 1.0):
    '''
    Retuns the accumulation of the number of iterations per @time_step.
    '''
    results = []
    for df in df_list:
        times = df['time'].to_numpy()

        time = time_step
        iteration_accumulator = []
        num_iterations = 0

        for t in times:
            if t <= time:
                num_iterations += 1
            elif t > time:
                iteration_accumulator.append(num_iterations)
                time += time_step
                num_iterations += 1

        iteration_accumulator.append(num_iterations)
        results.append(np.array(iteration_accumulator))
    return results


def count_median_iterations_over_time(data: List[List[pd.DataFrame]]):
    '''Return the average number iter / sec for each list of pd.DataFrames.'''
    results = []
    for df_list in data:
        # aggregates iter. over time
        tmp = aggregate_iterations(df_list)
        # extends the arrays to the longest one
        max_length = np.max([len(l) for l in tmp])
        for i in range(len(tmp)):
            arr = tmp[i]
            last_value = arr[-1]
            arr_length = len(arr)
            if arr_length < max_length:
                tmp[i] = np.array([v for v in arr] + [last_value for _ in range(max_length - len(arr))])
        median = np.median(np.array(tmp), axis=0)
        results.append(median)
    return results
