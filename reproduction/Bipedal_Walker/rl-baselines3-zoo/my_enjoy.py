import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse, importlib, sys, time, copy, tqdm, pickle, yaml
import numpy as np
import torch

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper

from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict
from fuzz.fuzz import fuzzing

# implementation of our logs
sys.path.append('../../logger/')
from logger import MyLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    # num_timesteps (i.e. M) set to default to 300
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=300, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
        "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environemnt package modules to import (e.g. gym_minigrid)",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument("--em", action="store_true", default=False)
    # adds path for logging
    parser.add_argument("--path", type=str, default="../../data/bw/no_change/")
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    folder = args.folder

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
        print(f"Loading latest experiment, id={args.exp_id}")

    # Sanity checks
    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_id}_{args.exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    found = False
    for ext in ["zip"]:
        model_path = os.path.join(log_path, f"{env_id}.{ext}")
        found = os.path.isfile(model_path)
        if found:
            break

    if args.load_best:
        model_path = os.path.join(log_path, "best_model.zip")
        found = os.path.isfile(model_path)

    if args.load_checkpoint is not None:
        model_path = os.path.join(log_path, f"rl_model_{args.load_checkpoint}_steps.zip")
        found = os.path.isfile(model_path)

    if not found:
        raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")

    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)
    # sets PyTorch num of threads to 1 for monothreaded executions
    torch.set_num_threads(1)

    is_atari = ExperimentManager.is_atari(env_id)

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    env_kwargs = {}
    args_path = os.path.join(log_path, env_id, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    env = create_test_env(
        env_id,
        n_envs=args.n_envs,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        kwargs.update(dict(buffer_size=1))

    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)


    # Deterministic by default except for atari games
    stochastic = args.stochastic or is_atari and not args.deterministic
    deterministic = not stochastic

    # sets the seed used for sampling and fuzzing
    seed = args.seed
    np.random.seed(seed)

    init_budget_in_minutes = 120
    fuzzing_budget_in_minutes = 720

    # no_coverage indicates whether to skip the coverage computation
    no_coverage = (args.em == False)
    path = args.path

    folder = '../../data/bw/'
    if not os.path.isdir(folder):
        os.mkdir(folder)
    subfolder = 'no_change/'
    if not os.path.isdir(folder + subfolder):
        os.mkdir(folder + subfolder)

    folder += subfolder

    if no_coverage:
        subfolder = 'fuzzer/'
    else:
        subfolder = 'mdpfuzz/'

    if not os.path.isdir(folder + subfolder):
        os.mkdir(folder + subfolder)

    path += subfolder
    path += str(seed)

    logger = MyLogger(path + '_sampling_logs.txt')
    logger.write_columns()

    f = open(path + '.txt', 'w', buffering=1)
    sys.stdout = f

    fuzzer = fuzzing()

    budget = (60 * init_budget_in_minutes)
    pbar = tqdm.tqdm(total=init_budget_in_minutes)
    start_time = time.time()
    current_time = time.time()
    last_minute_time = time.time()
    num_executions = 0
    # sampling starts
    while (current_time - start_time) < budget:
        # executes a random input
        states = np.random.randint(low=1, high=4, size=15)
        state = None
        episode_reward = 0.0
        obs = env.reset(states)
        sequences = [obs[0]]
        t0 = time.time()
        for _ in range(args.n_timesteps):
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, infos = env.step(action)
            sequences.append(obs[0])
            episode_reward += reward[0]
            if done:
                break
        run_time = time.time()
        exec_time = run_time - t0
        num_executions += 1
        crash_flag = (done[0] or episode_reward < 10) # only for logging

        if not done:
            state = None
            episode_reward_mutate = 0.0
            # input mutation
            delta_states = np.random.choice(2, 15, p=[0.9, 0.1])
            if np.sum(delta_states) == 0:
                delta_states[0] = 1
            mutate_states = states + delta_states
            mutate_states = np.remainder(mutate_states, 4)
            mutate_states = np.clip(mutate_states, 1, 3)
            # mutated input execution
            obs = env.reset(mutate_states)
            t0 = time.time()
            for _ in range(args.n_timesteps):
                action, state = model.predict(obs, state=state, deterministic=deterministic)
                obs, reward, done, infos = env.step(action)
                episode_reward_mutate += reward[0]
                if done:
                    break
            mutate_run_time = time.time()
            mutate_exec_time = mutate_run_time - t0
            num_executions += 1
            mutate_crash_flag = (done[0] or episode_reward_mutate < 10)
            # computes sensitivity
            sensitivity = np.abs(episode_reward_mutate - episode_reward) / np.sum(delta_states)
            if no_coverage:
                coverage = coverage_time = None
            else:
                t0 = time.time()
                coverage = fuzzer.state_coverage(sequences)
                coverage_time = time.time() - t0
            fuzzer.further_mutation(states, episode_reward, sensitivity, coverage, states)
            # logs the execution of the input
            logger.log(
                input=states,
                oracle=crash_flag,
                reward=episode_reward,
                coverage=coverage,
                sensitivity=sensitivity,
                test_exec_time=exec_time,
                coverage_time=coverage_time,
                run_time=run_time
            )
            # logs the execution of the mutant
            logger.log(
                input=mutate_states,
                oracle=mutate_crash_flag,
                reward=episode_reward_mutate,
                test_exec_time=mutate_exec_time,
                run_time=mutate_run_time
            )

        # update the pbar
        now = time.time()
        if now - last_minute_time > 60:
            pbar.update(1)
            last_minute_time = now
            pbar.set_postfix({'Iterations': num_executions, 'Pool': len(fuzzer.corpus)})
        current_time = now

    pbar.close()
    print('Corpus of size {} initialized after {} executions'.format(len(fuzzer.corpus), num_executions))

    fuzzer.count = [5] * len(fuzzer.corpus)

    tau = 0.01

    logger = MyLogger(path + '_fuzzing_logs.txt')
    logger.write_columns()

    budget = (60 * fuzzing_budget_in_minutes)
    pbar = tqdm.tqdm(total=fuzzing_budget_in_minutes)
    start_time = time.time()
    last_minute_time = time.time()
    num_executions = 0
    # fuzzing starts
    while len(fuzzer.corpus) > 0:
        states = fuzzer.get_pose()
        mutate_states = fuzzer.mutation(states)
        state = None
        episode_reward = 0.0
        obs = env.reset(mutate_states)
        sequences = [obs[0]]
        t0 = time.time()
        for _ in range(args.n_timesteps):
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, infos = env.step(action)
            sequences.append(obs[0])
            episode_reward += reward[0]
            if done:
                break
        run_time = time.time()
        exec_time = run_time - t0
        num_executions += 1
        crash_flag = (done[0] or episode_reward < 10)

        # computes sensitivity
        local_sensitivity = np.abs(episode_reward - fuzzer.current_reward)

        if no_coverage:
            coverage = coverage_time = None
        else:
            t0 = time.time()
            coverage = fuzzer.state_coverage(sequences)
            coverage_time = time.time() - t0

        logger.log(
                input=mutate_states,
                oracle=crash_flag,
                reward=episode_reward,
                coverage=coverage,
                sensitivity=local_sensitivity,
                test_exec_time=exec_time,
                coverage_time=coverage_time,
                run_time=time.time()
            )

        if crash_flag:
            fuzzer.add_crash(mutate_states)
        elif (episode_reward < fuzzer.current_reward) or ((not no_coverage) and (coverage < tau)):
            current_pose = copy.deepcopy(mutate_states)
            orig_pose = fuzzer.current_original
            fuzzer.further_mutation(current_pose, episode_reward, local_sensitivity, coverage, orig_pose)

        # update the pbar
        now = time.time()
        if (now - last_minute_time) > 60:
            pbar.update(1)
            last_minute_time = now
            pbar.set_postfix({'Iterations': num_executions, 'Pool': len(fuzzer.corpus)})
        if (now - start_time) > budget:
            break

    pbar.close()
    f.close()

    if no_coverage:
        file_name = path + 'crash_noEM.pkl'
    else:
        file_name = path + 'crash_EM.pkl'
    with open(file_name, 'wb') as handle:
        pickle.dump(fuzzer.result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if not args.no_render:
        if args.n_envs == 1 and "Bullet" not in env_id and not is_atari and isinstance(env, VecEnv):
            while isinstance(env, VecEnvWrapper):
                env = env.venv
            if isinstance(env, DummyVecEnv):
                env.envs[0].env.close()
            else:
                env.close()
        else:
            env.close()

# python my_enjoy.py --alg tqc --env BipedalWalkerHardcore-v3 --folder rl-trained-agents/ --no-render --seed 2020 --em
if __name__ == "__main__":
    main()