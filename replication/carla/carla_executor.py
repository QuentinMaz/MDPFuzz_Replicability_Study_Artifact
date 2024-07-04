import copy
import time
import sys

sys.path.append('./carla_RL_IAs/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg')
sys.path.append('../methods/src/')

import pandas as pd
import numpy as np

from typing import Any, Tuple, List, Dict
from benchmark import PointGoalSuite
from bird_view.models import agent_IAs_RL

from executor import Executor


class CarlaExecutor(Executor):

    def __init__(self, sim_steps: int, env: PointGoalSuite) -> None:
        super().__init__(sim_steps, 0)
        self.carla_env = env
        self.env_seed = self.carla_env.seed
        self.num_vehicles = self.carla_env.n_vehicles + 1 # adds one to consider the player
        print('Executor found Env of {} vehicles and {} seed'.format(self.num_vehicles, self.env_seed))
        self.start_positions = self._init_start_positions()
        self.num_start_positions = len(self.start_positions)
        self.initial_tasks = self._init_tasks()


    def generate_input(self, rng: np.random.Generator) -> np.ndarray:
        '''
        Samples randomly in the input space.
        The latter consists of a weather index, the start and destination positions and the positions of all the vehicles.
        '''
        w, s, t = rng.choice(self.initial_tasks)

        start = self.start_positions[s].copy()

        indices = []
        while len(indices) < self.num_vehicles - 1:
            i = rng.choice(self.num_start_positions)
            if (i != s) and (i not in indices):
                indices.append(i)
        return np.hstack([np.array([w, t]), start] + [self.start_positions[i].copy() for i in indices])


    def generate_inputs(self, rng: np.random.Generator, n: int) -> np.ndarray:
        '''Returns @n inputs as a 3d numpy array.'''
        inputs = []
        for _ in range(n):
            inputs.append(self.generate_input(rng))
        return np.array(inputs)


    def mutate(self, input: np.ndarray, rng: np.random.Generator, **kwargs) -> np.ndarray:
        mutant = self._mutate(input, rng)
        is_valid = self._validate_input(mutant)
        while not is_valid:
            mutant = self._mutate(input, rng)
            is_valid = self._validate_input(mutant)
        return mutant


    def execute_policy(self, input: np.ndarray, policy: Any) -> Tuple[float, bool, np.ndarray, float]:
        t0 = time.time()
        self.carla_env._reset(input)
        execution_result, sequence = self._execute_env(policy)
        return execution_result['reward'], execution_result['collided'], np.array(sequence), time.time() - t0


    def load_policy(self, args):
        agent_class = agent_IAs_RL.AgentIAsRL
        agent_maker = lambda: agent_class(args)
        agent = agent_maker()
        return agent


    def _execute_env(self, agent: agent_IAs_RL.AgentIAsRL) -> Tuple[Dict, List[np.ndarray]]:
        agent_entropy = 0.0
        total_reward = 0.0
        previous_speed = 0.0
        previous_distance = self.carla_env._local_planner.distance_to_goal
        current_invaded = self.carla_env.invaded
        previous_invaded_frame_number = self.carla_env._invaded_frame_number
        current_collided = self.carla_env.collided
        previous_collided_frame_number = self.carla_env._collided_frame_number
        sequence = []
        while self.carla_env.tick():
            # observations and reward
            observations = self.carla_env.get_observations()
            reward = self._calculate_reward(previous_distance, self.carla_env._local_planner.distance_to_goal, current_collided, current_invaded, observations['velocity'], previous_speed)
            # updates the previous and current values
            previous_speed = observations['velocity']
            previous_distance = self.carla_env._local_planner.distance_to_goal
            current_invaded = (previous_invaded_frame_number != self.carla_env._invaded_frame_number)
            current_collided = (previous_collided_frame_number != self.carla_env._collided_frame_number)
            total_reward += reward

            control, entropy, _ = agent.run_step(observations)
            agent_entropy += entropy
            diagnostic = self.carla_env.apply_control(control)

            # sequence calculation (copy)
            temp = copy.deepcopy(observations['node'])
            temp = np.hstack((temp, copy.deepcopy(observations['orientation']), copy.deepcopy(observations['velocity']), copy.deepcopy(observations['acceleration']), copy.deepcopy(observations['position']), copy.deepcopy(np.array([observations['command']]))))
            vehicle_index = np.nonzero(observations['vehicle'])
            vehicle_obs = np.zeros(3)
            vehicle_obs[0] = vehicle_index[0].mean()
            vehicle_obs[1] = vehicle_index[1].mean()
            vehicle_obs[2] = np.sum(observations['vehicle']) / 1e5
            temp = np.hstack((temp, vehicle_obs))
            sequence.append(temp)

            terminate = self.carla_env.collided or self.carla_env._tick > 100 or self.carla_env.is_failure() or self.carla_env.is_success()
            if terminate:
                result = {}
                result['reward'] = total_reward
                result['success'] = self.carla_env.is_success()
                result['total_lights_ran'] = self.carla_env.traffic_tracker.total_lights_ran
                result['total_lights'] = self.carla_env.traffic_tracker.total_lights
                result['collided'] = self.carla_env.collided
                result['t'] = self.carla_env._tick
                return result, sequence


    def _calculate_reward(self, prev_distance, cur_distance, cur_collid, cur_invade, cur_speed, prev_speed) -> float:
        '''
        Copy of the initial work.
        It rewards agent based on the distance and speed differences.
        Invading lanes creates a small penalty.
        Collisions create significant penalties.
        '''
        reward = 0.0
        reward += np.clip(prev_distance - cur_distance, -10.0, 10.0)
        cur_speed_norm = np.linalg.norm(cur_speed)
        prev_speed_norm = np.linalg.norm(prev_speed)
        reward += 0.2 * (cur_speed_norm - prev_speed_norm)
        if cur_collid:
            reward -= 100 * cur_speed_norm
        if cur_invade:
            reward -= cur_speed_norm
        return reward


    def _init_start_positions(self) -> np.ndarray:
        '''Retrieves from the map of the CARLA environment the starting points as a numpy array.'''
        positions_list = []
        transform_list = self.carla_env._map.get_spawn_points()
        for t in transform_list:
            # pitch and roll equal 0.0
            positions_list.append(np.array([t.location.x, t.location.y, t.location.z, t.rotation.yaw]))
        return np.vstack(positions_list)


    def _init_tasks(self) -> np.ndarray:
        '''Retrieves from the map of the CARLA environment the initial tasks.'''
        initial_tasks = [[w, s, t] for w, (s, t), _ in self.carla_env.all_tasks]
        return np.vstack(initial_tasks)


    def _mutate(self, input: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        '''
        Mutates all the vehicles by perturbing their x and y positions.
        It follows the original mutation operation implementation.
        '''
        mutant = input.copy()
        # does not change the weather and destination indices
        # perturbs more significantly the position of the player
        mutant[2] += rng.uniform(-0.15, 0.15)
        mutant[3] += rng.uniform(-0.15, 0.15)
        mutant[5] += rng.uniform(-5, 5)
        for i in range(6, len(mutant), 4):
            # x and y
            mutant[i] += rng.uniform(-0.1, 0.1)
            mutant[i + 1] += rng.uniform(-0.1, 0.1)
        return mutant


    def _validate_input(self, input: np.ndarray):
        '''Checks all the positions are far enough from each other.'''
        # skips the weather index and the destination
        vehicle_positions = input[2:]
        splits = np.array_split(vehicle_positions, self.num_vehicles)
        for i in range(len(splits) - 1):
            xi, yi = splits[i][0], splits[i][1]
            for j in range(i + 1, len(splits)):
                xj, yj = splits[j][0], splits[j][1]
                dist = np.linalg.norm([xi - xj, yi - yj])
                if dist <= 1.90:
                    return False
        return True