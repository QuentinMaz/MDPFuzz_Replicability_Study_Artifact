import time

import numpy as np
import gymnasium as gym
from typing import Any, Tuple, List
from stable_baselines3 import DQN

import sys
sys.path.append('../methods/src/')
from executor import Executor

from gimitest.env_decorator import EnvDecorator
from gimitest.gtest import GTest


class InitialStateTester(GTest):

    def __init__(self, env, env_seed: int = 0, agent=None):
        super().__init__(env, agent)
        assert env_seed >= 0
        self.env_seed = env_seed
        self.initial_state: np.ndarray = np.zeros(4, dtype=np.float32)


    # def pre_reset_configuration(self):
    #     return {'seed': self.env_seed}


    def post_reset_configuration(self, next_state):
        # deterministic executions by setting the unwrapped environment
        self.set_attribute(self.env.unwrapped, 'state', np.array(self.initial_state, dtype=np.float32))
        return np.array(self.initial_state, dtype=np.float32)


    def set_initial_state(self, state: np.ndarray):
        assert len(state) == 4
        assert np.all(state >= DEFAULT_MIN), state
        assert np.all(state <= DEFAULT_MAX), state
        self.initial_state = np.array(state, dtype=np.float32)


DEFAULT_MIN = -0.05
DEFAULT_MAX = 0.05
MUTATION_INTENSITY = 0.01


class CartPoleExecutor(Executor):

    def __init__(self, sim_steps: int, env_seed: int = 0) -> None:
        super().__init__(sim_steps, env_seed)
        self._env = gym.make('CartPole-v1')
        self.gtester = InitialStateTester(self._env)
        EnvDecorator.decorate(self._env, self.gtester)


    def generate_input(self, rng: np.random.Generator):
        '''Generates a single input between the given bounds (parameters).'''
        return rng.uniform(low=DEFAULT_MIN, high=DEFAULT_MAX, size=4)


    def generate_inputs(self, rng: np.random.Generator, n: int = 1):
        '''Generates @n inputs with the lower and upper bounds parameters.'''
        if n == 1:
            return self.generate_input(rng)
        else:
            return rng.uniform(low=DEFAULT_MIN, high=DEFAULT_MAX, size=(n, 4))


    def mutate(self, input: np.ndarray, rng: np.random.Generator, **kwargs) -> np.ndarray:
        intensity = kwargs.get('intensity', MUTATION_INTENSITY) # type: float
        return np.clip(rng.normal(input, intensity), DEFAULT_MIN, DEFAULT_MAX)


    def load_policy(self, model_path: str = 'models/10140000.zip'):
        return DQN.load(model_path, device='cpu')


    def execute_policy(self, input: np.ndarray, policy: Any) -> Tuple[float, bool, np.ndarray, float]:
        '''Executes the model and returns the trajectory data. Useful for MDPFuzz.'''
        t0 = time.time()
        self.gtester.set_initial_state(input)
        obs, _info = self._env.reset()
        state = None
        acc_reward = 0.0

        obs_seq = [obs]

        for _ in range(self.sim_steps):
            action, state = policy.predict(obs, state=state, deterministic=True)
            obs, reward, terminated, truncated, _info = self._env.step(action)
            acc_reward += reward

            obs_seq.append(obs)

            if terminated or truncated:
                break
        return acc_reward, (acc_reward != self.sim_steps), np.array(obs_seq), time.time() - t0


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    executor: Executor = CartPoleExecutor(400, 0)
    policy = executor.load_policy()
    # input = executor.generate_input(rng)
    input = np.array([-0.00389543,-0.00785703, 0.05      , 0.03909101])
    reward, oracle, sequence, exec_time = executor.execute_policy(input, policy)
    print(input, reward, oracle, exec_time)
    print(sequence.shape)


