import os
import sys
import json
import pandas as pd
import numpy as np

from typing import List, Tuple, Dict, Union

from logger.logger import MyLogger
from common import MDPFUZZ_SUFFIX, FUZZER_SUFFIX

############# DATA LOADING #####################


def load_log_file(filename: str):
    '''Returns a log file as a pd.DataFrame.'''
    if not filename.endswith('.txt'):
        filename += '.txt'
    if not os.path.isfile(filename):
        raise FileNotFoundError("\"{}\" not found.".format(filename))

    logger = MyLogger(filename)
    df = logger.load()

    # accounts for data of carla for which inputs are None

    if np.isnan(np.vstack(df['input'])).any():
        df['input'] = np.arange(len(df))

    return df


def get_fuzzing_logs(path: str):
    '''Return the results of log files ending with "fuzzing_logs.txt" (found at @path) as a list of pd.DataFrames.'''
    df_list = [load_log_file(path + f) for f in os.listdir(path) if f.endswith('fuzzing_logs.txt')]
    return add_time_to_dataframes(df_list)


def add_time_to_dataframes(df_list: List[pd.DataFrame]) -> List[pd.DataFrame]:
    '''
    Processes a list of pd.DataFrames.
    The processing consists in adding relative timestamps (column 'time'), which thus removes the first iteration.
    '''
    results = []
    for df in df_list:
        t0 = df['run_time'][0]
        df['time'] = df['run_time'] - t0
        results.append(df.tail(len(df) - 1))
    return results



def store_dict(filepath: str, result_dict: Dict[str, Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]]):
    '''Stores a dictionary of results at the given @filename.'''
    dict_to_store = {}
    for k, v in result_dict.items():
        # accumulation of number of executions over time
        if isinstance(v, np.ndarray):
            new_v = v.tolist()
        # a tuple of statistical results (fault discovery)
        elif isinstance(v, Tuple):
            assert np.all([isinstance(t, np.ndarray) for t in v])
            new_v = [t.tolist() for t in v]
        else:
            new_v = v
        dict_to_store[k] = new_v

    filename = filepath.split('.json')[0]
    with open(f'{filename}.json', 'w') as f:
        f.write(json.dumps(dict_to_store))


def load_dict(filepath: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    '''
    Loads stored results.
    It return an empty dictionnary if the dumped dictionnary is malformed.
    '''
    filepath = filepath.split('.')[0] + '.json'
    # assert os.path.exists(filepath), filepath
    if not os.path.isfile(filepath):
        print("\"{}\" not found.".format(filepath))
        print("Empty dictionary returned.")
        return {}
    try:
        with open(filepath, 'r') as f:
            d = json.load(f)
    except:
        d = {}
    finally:
        for k in d.keys():
            v = d[k]
            assert isinstance(v, List), 'Results should be list(s).'
            if isinstance(v[0], List):
                assert np.all(isinstance(l, List) for l in v[1:]), 'Malformed data: expect a list of lists.'
                new_v = [np.array(l) for l in v]
            else:
                new_v = np.array(v)
            d[k] = new_v
        return d


def store_results(
        keys: List[str],
        results: List,
        output_path: str
    ):
    '''
    Stores the results of MDPFuzz or Fuzzer.
    The results are assumed to be sorted w.r.t @keys!
    '''
    store_dict(output_path, {k: t for k, t in zip(keys, results)})


def load_results_of_fuzzers(
        path: str,
        use_case_keys: List[str] = None) -> Tuple[List[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]], List[List[str]]]:
    '''
    Convenient function that returns the results of MDPFuzz and Fuzzer as lists of statistical results.
    The lists are indexed by the keys (default to the ones found in the dictionary stored).
    MDPFuzz first.
    '''
    dict_list = load_dict(path + MDPFUZZ_SUFFIX), load_dict(path + FUZZER_SUFFIX)
    results_list = []
    keys_list = []

    for d in dict_list:
        keys = list(d.keys())

        if use_case_keys is None:
            keys.sort()
        else:
            assert np.all([k in keys for k in use_case_keys])
            keys = use_case_keys

        results_list.append([d[k] for k in keys])
        keys_list.append(keys)

    return results_list, keys_list
