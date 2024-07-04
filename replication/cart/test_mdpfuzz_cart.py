import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import torch

import sys
sys.path.append('../methods/src/')
from cp_executor import CartPoleExecutor
from mdpfuzz import Fuzzer


if __name__ == '__main__':
    torch.set_num_threads(1)
    test_budget = 5000
    init_budget = 1000

    tmp = sys.argv[1:]

    # reads parameters and path in the parameters.txt if not given
    if len(tmp) == 5:
        args = tmp
    else:
        assert len(tmp) == 2
        index = int(tmp[0])
        with open('../parameters.txt', 'r') as f:
            lines = f.readlines()
        parameters = lines[index].strip().split(' ')
        assert len(parameters) == 4
        args = []
        for p in parameters:
            args.append(p)
        args.append(tmp[-1])

    k = int(args[0])
    tau = float(args[1])
    gamma = float(args[2])
    seed = int(args[3])


    print('k', 'tau', 'gamma', 'seed')
    print(k, tau, gamma, seed)

    path = '{}_{}_{}_{}_{}'.format(args[4], k, tau, gamma, seed)
    print(path)

    executor = CartPoleExecutor(400, env_seed=0)
    policy = executor.load_policy()

    fuzzer = Fuzzer(random_seed=seed, k=k, tau=tau, gamma=gamma, executor=executor)
    fuzzer.fuzzing(
        n=init_budget,
        policy=policy,
        test_budget=test_budget,
        saving_path=path,
        local_sensitivity=True,
        save_logs_only=True,
        exp_name='Cart Pole')