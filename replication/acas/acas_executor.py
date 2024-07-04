import torch
import time
import copy
import numpy as np

import sys
sys.path.append('../methods/src/')
from executor import Executor
from typing import Any, Tuple
from models.load_model import read_onnx

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class ACASagent:
    def __init__(self, acas_speed):
        self.x = 0
        self.y = 0
        self.theta = np.pi / 2
        self.speed = acas_speed
        self.interval = 0.1
        self.model_1 = read_onnx(1, 2)
        self.model_2 = read_onnx(2, 2)
        self.model_3 = read_onnx(3, 2)
        self.model_4 = read_onnx(4, 2)
        self.model_5 = read_onnx(5, 2)
        self.prev_action = 0
        self.current_active = None

    def step(self, action):
        if action == 1:
            self.theta = self.theta + 1.5 / 180 * np.pi * self.interval
            while self.theta > np.pi:
                self.theta -= np.pi * 2
            while self.theta < -np.pi:
                self.theta += np.pi * 2
        elif action == 2:
            self.theta = self.theta - 1.5 / 180 * np.pi * self.interval
            while self.theta > np.pi:
                self.theta -= np.pi * 2
            while self.theta < -np.pi:
                self.theta += np.pi * 2
        elif action == 3:
            self.theta = self.theta + 3 / 180 * np.pi * self.interval
            while self.theta > np.pi:
                self.theta -= np.pi * 2
            while self.theta < -np.pi:
                self.theta += np.pi * 2
        elif action == 4:
            self.theta = self.theta - 3 / 180 * np.pi * self.interval
            while self.theta > np.pi:
                self.theta -= np.pi * 2
            while self.theta < -np.pi:
                self.theta += np.pi * 2
        self.x = self.x + self.speed * np.cos(self.theta) * self.interval
        self.y = self.y + self.speed * np.sin(self.theta) * self.interval

    def act(self, inputs):
        inputs = torch.Tensor(inputs)
        # action = np.random.randint(5)
        if self.prev_action == 0:
            model = self.model_1
        elif self.prev_action == 1:
            model = self.model_2
        elif self.prev_action == 2:
            model = self.model_3
        elif self.prev_action == 3:
            model = self.model_4
        elif self.prev_action == 4:
            model = self.model_5
        action, active = model(inputs)
        # action = model(inputs)
        self.current_active = [action.clone().detach().numpy(), active.clone().detach().numpy()]
        action = action.argmin()
        self.prev_action = action
        return action

    def act_proof(self, direction):
        return direction


class Autoagent:
    def __init__(self, x, y, auto_theta, speed=None):
        self.x = x
        self.y = y
        self.theta = auto_theta
        self.speed = speed
        self.interval = 0.1

    def step(self, action):
        if action == 1:
            self.theta = self.theta + 1.5 / 180 * np.pi * self.interval
            while self.theta > np.pi:
                self.theta -= np.pi * 2
            while self.theta < -np.pi:
                self.theta += np.pi * 2
        elif action == 2:
            self.theta = self.theta - 1.5 / 180 * np.pi * self.interval
            while self.theta > np.pi:
                self.theta -= np.pi * 2
            while self.theta < -np.pi:
                self.theta += np.pi * 2
        elif action == 3:
            self.theta = self.theta + 3 / 180 * np.pi * self.interval
            while self.theta > np.pi:
                self.theta -= np.pi * 2
            while self.theta < -np.pi:
                self.theta += np.pi * 2
        elif action == 4:
            self.theta = self.theta - 3 / 180 * np.pi * self.interval
            while self.theta > np.pi:
                self.theta -= np.pi * 2
            while self.theta < -np.pi:
                self.theta += np.pi * 2
        self.x = self.x + self.speed * np.cos(self.theta) * self.interval
        self.y = self.y + self.speed * np.sin(self.theta) * self.interval

    def act(self):
        # action = np.random.randint(5)
        action = 0
        return action


class env:
    def __init__(self, acas_speed, x2, y2, auto_theta):
        self.ownship = ACASagent(acas_speed)
        self.inturder = Autoagent(x2, y2, auto_theta)
        self.row = np.linalg.norm([self.ownship.x - self.inturder.x, self.ownship.y - self.inturder.y])
        if self.inturder.x - self.ownship.x > 0:
            self.alpha = np.arcsin((self.inturder.y - self.ownship.y) / self.row) - self.ownship.theta
        else:
            self.alpha = np.pi - np.arcsin((self.inturder.y - self.ownship.y) / self.row) - self.ownship.theta
        # if self.inturder.x - self.ownship.x < 0:
        #     self.alpha = np.pi - self.alpha
        while self.alpha > np.pi:
            self.alpha -= np.pi * 2
        while self.alpha < -np.pi:
            self.alpha += np.pi * 2
        self.phi = self.inturder.theta - self.ownship.theta
        while self.phi > np.pi:
            self.phi -= np.pi * 2
        while self.phi < -np.pi:
            self.phi += np.pi * 2

        if x2 == 0:
            if y2 > 0:
                self.inturder.speed = self.ownship.speed / 2
            else:
                self.inturder.speed = np.min([self.ownship.speed * 2, 1600])
        elif self.ownship.theta == self.inturder.theta:
            self.inturder.speed = self.ownship.speed
        else:
            self.inturder.speed = self.ownship.speed * np.sin(self.alpha) / np.sin(self.alpha + self.ownship.theta - self.inturder.theta)

        if self.inturder.speed < 0:
            self.inturder.theta = self.inturder.theta + np.pi
            while self.inturder.theta > np.pi:
                self.inturder.theta -= 2 * np.pi
            self.inturder.speed = -self.inturder.speed
        self.Vown = self.ownship.speed
        self.Vint = self.inturder.speed

    def update_params(self):
        self.row = np.linalg.norm([self.ownship.x - self.inturder.x, self.ownship.y - self.inturder.y])
        self.Vown = self.ownship.speed
        self.Vint = self.inturder.speed
        self.alpha = np.arcsin((self.inturder.y - self.ownship.y) / self.row)
        if self.inturder.x - self.ownship.x > 0:
            self.alpha = np.arcsin((self.inturder.y - self.ownship.y) / self.row) - self.ownship.theta
        else:
            self.alpha = np.pi - np.arcsin((self.inturder.y - self.ownship.y) / self.row) - self.ownship.theta

        while self.alpha > np.pi:
            self.alpha -= np.pi * 2
        while self.alpha < -np.pi:
            self.alpha += np.pi * 2
        self.phi = self.inturder.theta - self.ownship.theta
        while self.phi > np.pi:
            self.phi -= np.pi * 2
        while self.phi < -np.pi:
            self.phi += np.pi * 2

    def step(self):
        acas_act = self.ownship.act([self.row, self.alpha, self.phi, self.Vown, self.Vint])
        auto_act = self.inturder.act()
        self.ownship.step(acas_act)
        self.inturder.step(auto_act)
        self.update_params()
        # time.sleep(0.1)

    def step_proof(self, direction):
        acas_act = self.ownship.act_proof(direction)
        auto_act = self.inturder.act()
        self.ownship.step(acas_act)
        self.inturder.step(auto_act)
        self.update_params()


class AcasExecutor(Executor):

    def __init__(self, sim_steps, env_seed) -> None:
        super().__init__(sim_steps, env_seed)
        self.dist_threshold = 200


    def generate_input(self, rng: np.random.Generator) -> np.ndarray:
        is_input_valid = False
        while not is_input_valid:
            acas_speed = rng.uniform(low=10, high=1100)
            row = rng.uniform(low=1000, high=60261)
            theta = rng.uniform(low=-np.pi, high=np.pi)
            x2 = row * np.cos(theta)
            y2 = row * np.sin(theta)
            bound1, bound2 = self._calculate_init_bounds(x2, y2)
            auto_theta = rng.uniform(low=bound1, high=bound2)
            is_input_valid = self._verify(acas_speed, x2, y2, auto_theta)
        return np.array([acas_speed, x2, y2, auto_theta])


    def generate_inputs(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if n == 1:
            return self.generate_input(rng)
        else:
            inputs = []
            for _ in range(n):
                inputs.append(self.generate_input(rng))
            return np.vstack(inputs)


    def mutate(self, input: np.ndarray, rng: np.random.Generator, **kwargs) -> np.ndarray:
        orig_acas_speed, orig_x2, orig_y2, orig_auto_theta = input
        new_acas_speed = orig_acas_speed + rng.uniform(-5, 5)
        new_x2 = orig_x2 + rng.uniform(-5, 5)
        new_y2 = orig_y2 + rng.uniform(-5, 5)
        new_auto_theta = orig_auto_theta + rng.uniform(-0.2, 0.2)
        return np.array([new_acas_speed, new_x2, new_y2, new_auto_theta])


    def execute_policy(self, input: np.ndarray, policy: Any) -> Tuple[float, bool, np.ndarray, float]:
        acas_speed, x2, y2, auto_theta = input
        air1 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
        air1.update_params()
        gamma = 0.99
        reward = 0
        collide_flag = False
        states_seq = []

        t0 = time.time()
        for _ in range(self.sim_steps):
            air1.step()
            reward = reward * gamma + air1.row / 60261.0
            states_seq.append(self._normalize_state([air1.row, air1.alpha, air1.phi, air1.Vown, air1.Vint]))
            if air1.row < self.dist_threshold:
                collide_flag = True
                reward -= 100

        return reward, collide_flag, np.array(states_seq), (time.time() - t0)


    def load_policy(self):
        return None


    def _calculate_init_bounds(self, x2, y2):
        bound1 = np.arcsin(-y2/np.linalg.norm([x2, y2]))
        if x2 > 0:
            bound1 = np.pi - bound1
        if bound1 > np.pi / 2:
            return np.pi / 2, bound1
        else:
            return bound1, np.pi / 2


    def _verify(self, acas_speed, x2, y2, auto_theta):
        dis_threshold = 200
        air1 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
        air1.update_params()

        air2 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
        air2.update_params()

        min_dis1 = air1.row
        min_dis2 = air2.row
        for _ in range(100):
            air1.step_proof(3)
            if min_dis1 > air1.row:
                min_dis1 = air1.row
        for _ in range(100):
            air2.step_proof(4)
            if min_dis2 > air2.row:
                min_dis2 = air2.row
        # print(min_dis1, min_dis2)
        if (air1.Vint <= 1200 and air1.Vint >= 0) and (min_dis1 >= dis_threshold or min_dis2 >= dis_threshold):
            return True
        else:
            return False


    def _normalize_state(self, x):
        y = copy.deepcopy(x)
        y = np.array(y)
        y = y - np.array([1.9791091e+04, 0.0, 0.0, 650.0, 600.0])
        y = y / np.array([60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0])
        return y.tolist()


    def _reward_func(self, acas_speed, x2, y2, auto_theta):
        air1 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
        air1.update_params()
        gamma = 0.99
        reward = 0
        collide_flag = False
        states_seq = []

        for j in range(100):
            air1.step()
            reward = reward * gamma + air1.row / 60261.0
            states_seq.append(self._normalize_state([air1.row, air1.alpha, air1.phi, air1.Vown, air1.Vint]))
            if air1.row < self.dist_threshold:
                collide_flag = True
                reward -= 100

        return reward, collide_flag, states_seq



def replay_test_case(input: np.ndarray, filepath: str, sim_steps: int = 100, dist_threshold: int = 200):
    x_of_model = []
    y_of_model = []
    x_of_auto = []
    y_of_auto = []
    acas_speed, x2, y2, auto_theta = input
    air = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
    air.update_params()
    min_row = air.row
    for _ in range(sim_steps):
        x_of_model.append(air.ownship.x)
        y_of_model.append(air.ownship.y)
        x_of_auto.append(air.inturder.x)
        y_of_auto.append(air.inturder.y)
        air.step()
        if min_row > air.row:
            min_row = air.row
    print('min distance:', min_row, 'crash:', min_row < dist_threshold)
    visualize(filepath, x_of_model, y_of_model, x_of_auto, y_of_auto)


def visualize(filepath: str, x_acas, y_acas, x_auto, y_auto):
    length = len(x_acas)
    assert np.all([len(arr) == length for arr in [y_acas, x_auto, y_auto]])

    min_x = np.min([np.min(x_acas), np.min(x_auto)])
    min_x -= 100
    max_x = np.max([np.max(x_acas), np.max(x_auto)])
    max_x += 100
    min_y = np.min([np.min(y_acas), np.min(y_auto)])
    min_y -= 100
    max_y = np.max([np.max(y_acas), np.max(y_auto)])
    max_y += 100
    fig = plt.figure()
    ax = plt.axes(xlim=(min_x, max_x), ylim=(min_y, max_y))
    line1, = ax.plot([], [], lw=2, c='red', label='model-controlled agent')
    line2, = ax.plot([], [], lw=2, c='blue', label='intruder')
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2
    def animate(k):
        line1.set_data(x_acas[:k], y_acas[:k])
        line2.set_data(x_auto[:k], y_auto[:k])
        return line1, line2
    ax.legend(loc=1)
    fig.tight_layout()
    ani = FuncAnimation(fig, animate, init_func=init, frames=length, interval=1, blit=True)
    gif_name = filepath.split('.gif')[0] + '.gif'
    # ani = animation.ArtistAnimation(fig, tmp, interval=1, repeat_delay=0, blit=True)
    ani.save(gif_name, writer='pillow')


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    executor: Executor = AcasExecutor(100, 0)
    input = executor.generate_input(rng)
    policy = executor.load_policy()
    reward, oracle, sequence, exec_time = executor.execute_policy(input, policy)
    print(input, reward, oracle, exec_time)
    print(sequence.shape)
    # replay_test_case(input, 'acastest', 100, 200)
