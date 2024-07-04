import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import torch

import sys
sys.path.append('../methods/src/')
from acas_executor import AcasExecutor
from mdpfuzz import Fuzzer

EXPERIMENT_SEEDS = [2021, 42, 2023, 20, 0]

if __name__ == '__main__':
    torch.set_num_threads(1)
    test_budget = 5000
    init_budget = 1000
    k = 10
    tau = 0.01
    gamma = 0.01

    executor = AcasExecutor(100, env_seed=0)
    policy = executor.load_policy()

    args = sys.argv[1:]
    assert len(args) == 2, 'Use: path, method name ("fuzzer", "mdpfuzz" or "rt")'

    path = args[0]
    method = args[1]
    method_names = ['fuzzer', 'mdpfuzz', 'rt']
    if method not in method_names:
        index = int(method)
        method = method_names[index]

    if not path.endswith('/'):
        path += '/'

    result_path = path + method

    for seed in EXPERIMENT_SEEDS:
        path = '{}_{}_{}_{}_{}'.format(result_path, k, tau, gamma, seed)
        print(path)
        fuzzer = Fuzzer(random_seed=seed, k=k, tau=tau, gamma=gamma, executor=executor)
        if method == 'rt':
            fuzzer.random_testing(
                n=test_budget,
                policy=policy,
                path=path,
                exp_name='ACAS Xu')
        elif method == 'fuzzer':
            fuzzer.fuzzing_no_coverage(
                n=init_budget,
                policy=policy,
                test_budget=test_budget,
                saving_path=path,
                local_sensitivity=True,
                save_logs_only=True,
                exp_name='ACAS Xu')
        else:
            fuzzer.fuzzing(
                n=init_budget,
                policy=policy,
                test_budget=test_budget,
                saving_path=path,
                local_sensitivity=True,
                save_logs_only=True,
                exp_name='ACAS Xu')