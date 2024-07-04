import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import time

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTester
import tensorflow.contrib.layers as layers
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv

from typing import Any, Tuple, List

import sys
sys.path.append('../../../methods/src/')
from executor import Executor


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    #TODO: changed the default timestep number to 100
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
    return parser.parse_args()


def make_env(scenario_name: str):
    scenario = scenarios.load(scenario_name + '.py').Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world_fuzz, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.done_flag, verify_func=scenario.verify)
    return env


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


class CoopExecutor(Executor):

    def __init__(self, sim_steps, env_seed) -> None:
        super().__init__(sim_steps, env_seed)
        self.scenario_name = 'simple_spread'
        self._tf_session = U.single_threaded_default_session()
        # self._arglist = argparse.Namespace(
        #     adv_policy='maddpg',
        #     batch_size=1024,
        #     benchmark=False,
        #     display=False,
        #     good_policy='maddpg',
        #     load_dir='',
        #     max_episode_len=self.sim_steps,
        #     num_adversaries=0,
        #     num_units=64,
        #     restore=False,
        #     save_dir='../checkpoints/',
        #     scenario=self.scenario_name)
        self._arglist = argparse.Namespace(
            adv_policy='maddpg',
            batch_size=1024,
            benchmark=False,
            display=False,
            exp_name='caca',
            gamma=0.95,
            good_policy='maddpg',
            load_dir='',
            max_episode_len=100,
            num_adversaries=0,
            num_units=64,
            restore=False,
            save_dir='../checkpoints/',
            scenario='simple_spread')

        self._env = make_env(self.scenario_name)
        self._num_agents = len(self._env.world.agents)
        self._num_landmarks = len(self._env.world.landmarks)
        self._total_num = self._num_agents + self._num_landmarks
        assert np.all([self._num_agents > 0, self._num_landmarks > 0])
        self._dim = self._env.world.dim_p * self._total_num
        assert all(self._env.agents[i].size == self._env.agents[0].size for i in range(self._num_agents))
        self._agent_size = self._env.agents[0].size
        self._min_valid_dist = self._agent_size * 1.1
        self._collision_dist = 2 * self._agent_size


    def generate_input(self, rng: np.random.Generator) -> np.ndarray:
        input = rng.uniform(-1, 1, self._dim)
        agent_initially_collided, landmarks_too_close = self._verify_input(input)

        while (agent_initially_collided and landmarks_too_close):
            input = rng.uniform(-1, 1, self._dim)
            agent_initially_collided, landmarks_too_close = self._verify_input(input)

        return input


    def generate_inputs(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if n == 1:
            return self.generate_input(rng)
        else:
            inputs = []
            for _ in range(n):
                inputs.append(self.generate_input(rng))
            return np.vstack(inputs)


    def mutate(self, input: np.ndarray, rng: np.random.Generator, **kwargs) -> np.ndarray:
        mutant = self._mutate(input, rng)
        agent_initially_collided, landmarks_too_close = self._verify_input(mutant)

        while (agent_initially_collided and landmarks_too_close):
            mutant = self._mutate(input, rng)
            agent_initially_collided, landmarks_too_close = self._verify_input(mutant)

        return mutant


    def execute_policy(self, input: np.ndarray, policy: Any) -> Tuple[float, bool, np.ndarray, float]:
        agent_positions, landmark_positions = self._get_positions(input)
        obs_list = self._env.reset(agent_positions, landmark_positions)
        state_sequence = []
        state_sequence.append(self._get_state())
        collisions = 0
        reward = 0
        t0 = time.time()
        for _ in range(self.sim_steps):
            action_list = [agent.action(obs) for agent, obs in zip(policy, obs_list)]
            new_obs_list, reward_list, done_list, _ = self._env.step(action_list)

            state_sequence.append(self._get_state())
            reward += sum(reward_list)
            done = np.all(done_list)
            collisions += self._count_collisions()

            if done:
                break

            obs_list = new_obs_list
        # follows the original implementation
        return reward, (collisions > 5), np.array(state_sequence), (time.time() - t0)


    def load_policy(self):
        trainers = self._get_trainers()
        U.initialize()
        U.load_state(self._arglist.save_dir)
        return trainers


    def _get_trainers(self):
        obs_shape_list = [self._env.observation_space[i].shape for i in range(self._env.n)]
        trainers = []
        model = mlp_model
        trainer = MADDPGAgentTester
        for i in range(self._env.n):
            trainers.append(trainer(
                'agent_%d' % i, model, obs_shape_list, self._env.action_space, i, self._arglist,
                local_q_func=(self._arglist.good_policy=='ddpg')))
        return trainers


    def _mutate(self, input: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mutant = []
        for i in range(len(input)):
            mutant.append(np.clip(input[i] + rng.uniform(-0.05, 0.05), -1, 1))
        return np.array(mutant)


    def _verify_input(self, input: np.ndarray):
        agent_collid = False
        landmark_collid = False
        a_positions, l_positions = self._get_positions(input)
        for i, agent_pos in enumerate(a_positions):
            for j, agent_other_pos in enumerate(a_positions):
                if i == j:
                    continue
                delta_pos = agent_pos - agent_other_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                if dist < self._min_valid_dist:
                    agent_collid = True
                    break
            if agent_collid:
                break
        for i, landmark_pos in enumerate(l_positions):
            for j, landmark_other_pos in enumerate(l_positions):
                if i == j:
                    continue
                delta_pos = landmark_pos - landmark_other_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                if dist < 0.15:
                    landmark_collid = True
                    break
            if landmark_collid:
                break
        return agent_collid, landmark_collid


    def _get_positions(self, input: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        # using array_split directly references the input
        splits = [arr.copy() for arr in np.array_split(input, self._total_num)]
        return splits[0:self._num_agents], splits[self._num_agents:self._total_num]


    def _get_state(self) -> np.ndarray:
        state = []
        for agent in self._env.world.agents:
            state.append(agent.state.p_pos)
            state.append(agent.state.p_vel)
            # state.append(agent.state.c) # always 0
        # worth keeping? I don't think so IMHO
        for landmark in self._env.world.landmarks:
            state.append(landmark.state.p_pos)
        return np.array(np.array(state).flatten())


    def _get_current_positions(self):
        '''Returns the current positions of the agents and the landmarks of the environment.'''
        state = []
        for agent in self._env.world.agents:
            state.append(agent.state.p_pos)
        for landmark in self._env.world.landmarks:
            state.append(landmark.state.p_pos)
        return np.array(state).flatten()


    def _count_collisions(self):
        '''Returns the number of collisions in the environment.'''
        counter = 0
        for i, agent in enumerate(self._env.world.agents):
            for j, agent_other in enumerate(self._env.world.agents):
                if i == j:
                    continue
                delta_pos = agent.state.p_pos - agent_other.state.p_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                if dist < self._collision_dist:
                    counter += 1
        # the number of collisions found is divided by 2 because we count them twice.
        return counter / 2


    def render_policy(self, input: np.ndarray, policy: Any) -> Tuple[float, bool, List[np.ndarray], float]:
        agent_positions, landmark_positions = self._get_positions(input)
        obs_list = self._env.reset(agent_positions, landmark_positions)
        collisions = 0
        reward = 0
        t0 = time.time()
        frames = [self._env.render(mode='rgb_array')[0]]
        for _ in range(self.sim_steps):
            action_list = [agent.action(obs) for agent, obs in zip(policy, obs_list)]
            new_obs_list, reward_list, done_list, _ = self._env.step(action_list)
            reward += sum(reward_list)
            done = np.all(done_list)
            collisions += self._count_collisions()
            frames.append(self._env.render(mode='rgb_array')[0])
            if done:
                break

            obs_list = new_obs_list
        return reward, (collisions > 5), frames, (time.time() - t0)

# xvfb-run -a python coop_executor.py
def render_frames(frames: List[np.ndarray], output_path='output.gif'):
    import imageio
    imageio.mimsave(output_path, frames, fps=10, loop=1)


if __name__ == '__main__':
    main_seed = 2021
    rng = np.random.default_rng(main_seed)
    executor = CoopExecutor(100, 0)
    input = executor.generate_input(rng)
    input2 = executor.mutate(input, rng)
    print(input)
    print(input2)
    policy = executor.load_policy()
    # reward, oracle, sequence, exec_time = executor.execute_policy(input, policy)
    print(executor.execute_policy(input, policy)[0])
    print(executor.execute_policy(input2, policy)[0])
    print(executor.execute_policy(input2, policy)[0])
    print(executor.execute_policy(input, policy)[0])