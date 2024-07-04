import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import time
import pickle
import copy
import tqdm
import sys

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTester
import tensorflow.contrib.layers as layers
from fuzz.fuzz import fuzzing

from typing import List

sys.path.append('../../../logger/')
from logger import MyLogger

'''
During our code review, we noticed 2 major bugs, likely to threaten the validity of the results:
    (1) The initial states are changed during test executions (use of references).
    (2) During the sampling, the mutated states are taken into account.
    In other words, the sensitivities are computed between two random (and uncorrelated) inputs.


--> This version has no fix.
'''

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    # default timestep number to 100
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=300000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='spread', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="../checkpoints/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    # adds seed, coverage guidance and path (for logging)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--no_coverage", action="store_true", default=False)
    parser.add_argument("--path", type=str, default="../../../data/coop/no_fix/")
    parser.add_argument('--init_budget', type=int, default=120, help="Time for initializing fuzzing (in minutes)")
    parser.add_argument('--fuzz_budget', type=int, default=720, help="Time for fuzzing (in minutes)")
    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False, fuzz=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if fuzz:
        env = MultiAgentEnv(world, scenario.reset_world_fuzz, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.done_flag, verify_func=scenario.verify)
    else:
        env = MultiAgentEnv(world, scenario.reset_world_before_fuzz, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.done_flag)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTester
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def get_observe(env):
    '''Observations are the positions, velocities and communication states of the agents + the positions of the landmarks.'''
    state = []
    for agent in env.world.agents:
        state.append(agent.state.p_pos)
        state.append(agent.state.p_vel)
        state.append(agent.state.c)
    for landmark in env.world.landmarks:
        state.append(landmark.state.p_pos)
    return list(np.array(state).flatten())


def get_init_state(env):
    '''Init states are the positions and the agents and the landmarks.'''
    state = []
    for agent in env.world.agents:
        state.append(agent.state.p_pos)
    for landmark in env.world.landmarks:
        state.append(landmark.state.p_pos)
    return state
    # return np.array(state).flatten()


def get_collision_num(env):
    '''Returns the number of collisions in the environment.'''
    collisions = 0
    for i, agent in enumerate(env.world.agents):
        for j, agent_other in enumerate(env.world.agents):
            if i == j:
                continue
            delta_pos = agent.state.p_pos - agent_other.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = (agent.size + agent_other.size)
            if dist < dist_min:
                collisions += 1
    # the number of collisions found is divided by 2 because we count them twice.
    return collisions / 2


def sample_seeds(args_list, log_path: str, budget_in_minutes: int = 120, seed: int = 2021):
    fuzzer = fuzzing()
    no_coverage = args_list.no_coverage
    # this is how the original work manages randomness
    np.random.seed(seed)

    logger = MyLogger(log_path + '_sampling_logs.txt')
    logger.write_columns()

    budget = (60 * budget_in_minutes)
    pbar = tqdm.tqdm(total=budget_in_minutes)
    start_time = time.time()

    num_executions = 0

    with U.single_threaded_session():
        # two environments: one which random initial start and another that sets the situation w.r.t input
        env = make_env(args_list.scenario, args_list, args_list.benchmark, fuzz=False)
        env_fuzzer = make_env(args_list.scenario, args_list, args_list.benchmark, fuzz=True)

        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, args_list.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, args_list)
        U.initialize()
        if args_list.load_dir == "":
            args_list.load_dir = args_list.save_dir
        U.load_state(args_list.load_dir)

        current_time = time.time()
        last_minute_time = time.time()
        while (current_time - start_time) < budget:
            # executes a random input by resetting the environment
            episode_rewards = 0
            obs_n = env.reset()
            episode_step = 0
            state_seqs = []
            state_seqs.append(get_observe(env))
            collisions = 0
            init_state = get_init_state(env)
            t0 = time.time()
            while True:
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                episode_step += 1
                state_seqs.append(get_observe(env))
                done = all(done_n)
                terminal = (episode_step >= args_list.max_episode_len)
                obs_n = new_obs_n
                collisions += get_collision_num(env)
                for rew in rew_n:
                    episode_rewards += rew
                if done or terminal:
                    print('original ends after:', episode_step)
                    break
            run_time = time.time()
            exec_time = run_time - t0
            num_executions += 1
            collide_flag = (collisions > 5)

            # coverage computation
            if no_coverage:
                coverage = None
                coverage_time = None
            else:
                t0 = time.time()
                coverage = fuzzer.state_coverage(state_seqs)
                coverage_time = time.time() - t0

            # input mutation and computation
            # mutated_init_state = fuzzer.mutate(copy.deepcopy(init_state))
            mutated_init_state = copy.deepcopy(init_state)
            mutated_obs_n = env_fuzzer.reset(mutated_init_state[0:3], mutated_init_state[3:])
            mutated_episode_step = 0
            mutated_reward = 0
            mutated_done = False
            mutated_terminal = False
            mutated_collisions = 0
            t0 = time.time()
            while True:
                action_n = [agent.action(obs) for agent, obs in zip(trainers, mutated_obs_n)]
                new_obs_n, rew_n, done_n, info_n = env_fuzzer.step(action_n)
                mutated_episode_step += 1
                mutated_done = all(done_n)
                mutated_terminal = (mutated_episode_step >= args_list.max_episode_len)
                mutated_obs_n = new_obs_n
                mutated_collisions += get_collision_num(env_fuzzer)
                for rew in rew_n:
                    mutated_reward += rew
                if mutated_done or mutated_terminal:
                    print('mutated ends after:', mutated_episode_step)
                    break
            mutated_run_time = time.time()
            mutated_exec_time = mutated_run_time - t0
            num_executions += 1
            mutated_collide_flag = (collisions > 5)
            # computes the sensitivity of the original input and feeds the fuzzer
            sensitivity = np.abs(episode_rewards - mutated_reward) / 100
            fuzzer.further_mutation(copy.deepcopy(init_state), episode_rewards, sensitivity, coverage, copy.deepcopy(init_state))

            # logs the execution of the input
            logger.log(
                input=np.array(init_state).flatten(),
                oracle=collide_flag,
                reward=episode_rewards,
                coverage=coverage,
                sensitivity=sensitivity,
                test_exec_time=exec_time,
                coverage_time=coverage_time,
                run_time=run_time
            )
            # logs the execution of the mutant
            logger.log(
                input=np.array(mutated_init_state).flatten(),
                oracle=mutated_collide_flag,
                reward=mutated_reward,
                test_exec_time=mutated_exec_time,
                run_time=mutated_run_time
            )
            # update the pbar
            now = time.time()
            if now - last_minute_time > 60:
                pbar.update(1)
                last_minute_time = now
                pbar.set_postfix({'Iterations': num_executions, 'Pool': len(fuzzer.corpus)})
            current_time = now
    pbar.close()
    return fuzzer, trainers, logger, num_executions


def fuzz(args_list, fuzzer: fuzzing, trainers, budget_in_minutes: int, log_path: str):
    no_coverage = args_list.no_coverage
    tau = 0.01

    logger = MyLogger(log_path + '_fuzzing_logs.txt')
    logger.write_columns()

    budget = 60 * budget_in_minutes
    pbar = tqdm.tqdm(total=budget_in_minutes)
    num_executions = 0

    with U.single_threaded_session():
        env = make_env(args_list.scenario, args_list, args_list.benchmark, fuzz=True)
        U.initialize()
        if args_list.load_dir == "":
            args_list.load_dir = args_list.save_dir
        U.load_state(args_list.load_dir)

        # time management
        last_minute_time = time.time()
        start_time = time.time()
        while len(fuzzer.corpus) > 0:
            # input selection and mutation
            current_pos = fuzzer.get_pose()
            new_pos = fuzzer.mutate(current_pos)
            obs_n = env.reset(new_pos[0:3], new_pos[3:])
            agent_flag, landmark_flag = env.verify_func(env.world)
            mutation_count = 0
            while agent_flag or landmark_flag:
                mutation_count += 1
                new_pos = fuzzer.mutate(current_pos)
                obs_n = env.reset(new_pos[0:3], new_pos[3:])
                agent_flag, landmark_flag = env.verify_func(env.world)
                if mutation_count > 10 and (agent_flag or landmark_flag):
                    fuzzer.drop_current()
                    mutation_count = 0
                    current_pos = fuzzer.get_pose()
            # input execution
            state_seqs = []
            init_state = get_init_state(env)
            episode_rewards = 0
            episode_step = 0
            collisions = 0
            t0 = time.time()
            while True:
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                episode_step += 1
                state_seqs.append(get_observe(env))
                done = all(done_n)
                terminal = (episode_step >= args_list.max_episode_len)
                collisions += get_collision_num(env)
                obs_n = new_obs_n
                for rew in rew_n:
                    episode_rewards += rew
                if terminal or done:
                    break
            run_time = time.time()
            exec_time = run_time - t0
            num_executions += 1
            collide_flag = (collisions > 5)

            if no_coverage:
                coverage_time = None
                coverage = None
            else:
                t0 = time.time()
                coverage = fuzzer.state_coverage(state_seqs)
                coverage_time = time.time() - t0

            local_sensitivity = np.abs(episode_rewards - fuzzer.current_reward) / 100.0

            # logs the new execution
            logger.log(
                input=np.array(init_state).flatten(),
                oracle=collide_flag,
                reward=episode_rewards,
                coverage=coverage,
                sensitivity=local_sensitivity,
                test_exec_time=exec_time,
                coverage_time=coverage_time,
                run_time=time.time()
            )

            if collide_flag:
                fuzzer.add_crash(copy.deepcopy(init_state))
            elif (episode_rewards < fuzzer.current_reward) or ((not no_coverage) and (coverage < tau)):
                fuzzer.further_mutation(copy.deepcopy(init_state), episode_rewards, local_sensitivity, coverage, fuzzer.current_original)

            now = time.time()
            if now - last_minute_time > 60:
                pbar.update(1)
                last_minute_time = now
                pbar.set_postfix({'Found': len(fuzzer.result), 'Pool': len(fuzzer.corpus)})
            if now - start_time > budget:
                break

    pbar.close()


if __name__ == '__main__':
    args_list = parse_args()
    # args_list = argparse.Namespace(
    #     adv_policy='maddpg',
    #     batch_size=1024,
    #     benchmark=False,
    #     display=False,
    #     exp_name='exp_name',
    #     gamma=0.95,
    #     good_policy='maddpg',
    #     load_dir='',
    #     max_episode_len=100,
    #     num_adversaries=0,
    #     num_units=64,
    #     restore=False,
    #     save_dir='../checkpoints/',
    #     scenario='simple_spread'
    # )
    # print(type(args_list), args_list._get_kwargs())

    init_budget = args_list.init_budget  # type: int
    fuzz_budget = args_list.fuzz_budget  # type: int

    no_coverage = args_list.no_coverage
    seed = args_list.seed
    path = args_list.path

    folder = '../../../data/coop/'
    if not os.path.isdir(folder):
        os.mkdir(folder)
    subfolder = 'no_fix/'
    if not os.path.isdir(folder + subfolder):
        try:
            os.mkdir(folder + subfolder)
        except:
            pass

    folder += subfolder

    if no_coverage:
        subfolder = 'fuzzer/'
    else:
        subfolder = 'mdpfuzz/'

    if not os.path.isdir(folder + subfolder):
        try:
            os.mkdir(folder + subfolder)
        except:
            pass

    path += subfolder
    path += str(seed)

    f = open(path + '.txt', 'w', buffering=1)
    sys.stdout = f

    fuzzer, trainers, _logger, num_iterations = sample_seeds(args_list, path, budget_in_minutes=init_budget, seed=seed)

    print('Corpus of size {} initialized after {} executions'.format(len(fuzzer.corpus), num_iterations))

    fuzz(args_list, fuzzer, trainers, fuzz_budget, path)

    f.close()


