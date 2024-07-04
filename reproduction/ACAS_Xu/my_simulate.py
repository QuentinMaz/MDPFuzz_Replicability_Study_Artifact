import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from models.load_model import read_onnx
import torch, time, pickle, copy, argparse, sys, tqdm
from fuzz.fuzz import fuzzing

sys.path.append('../logger/')
from logger import MyLogger


class ACASagent:
    def __init__(self, acas_speed):
        self.x = 0
        self.y = 0
        self.theta = np.pi / 2
        self.speed = acas_speed
        self.interval = 0.1
        self.model_1 = read_onnx(1, 2)
        # self.model_1 = load_repair(model_index=1)
        self.model_2 = read_onnx(2, 2)
        self.model_3 = read_onnx(3, 2)
        self.model_4 = read_onnx(4, 2)
        # self.model_4 = load_repair(model_index=4)
        self.model_5 = read_onnx(5, 2)
        # self.model_5 = load_repair(model_index=5)
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


def calculate_init_bounds(x2, y2):
    bound1 = np.arcsin(-y2/np.linalg.norm([x2, y2]))
    if x2 > 0:
        bound1 = np.pi - bound1
    if bound1 > np.pi / 2:
        return np.pi / 2, bound1
    else:
        return bound1, np.pi / 2


def sample_seeds(folder: str, num_seeds=1000):
    '''
    Function that initializes a fuzzer with @num_seeds seeds (i.e., pool of size @num_seeds).
    The sampling procedure follows the original implementation.
    It returns the fuzzer, the number of iterations needed and two wrappers.
    The first one is the .txt file of the inputs sampled, and the second one is an additional log file.
    '''
    np.random.seed(2020)
    dis_threshold = 200
    fuzzer = fuzzing()
    pbar = tqdm.tqdm(total=num_seeds)

    if not folder.endswith('/'):
        folder += '/'
    logs_buffer = open(folder + 'logs.txt', 'w', buffering=1)
    inputs_buffer = open(folder + 'inputs.txt', 'w', buffering=1)
    num_iterations = 0
    start_time = time.time()

    while len(fuzzer.corpus) < num_seeds:
        acas_speed = np.random.uniform(low=10, high=1100)
        row = np.random.uniform(low=1000, high=60261)
        theta = np.random.uniform(low=-np.pi, high=np.pi)
        x2 = row * np.cos(theta)
        y2 = row * np.sin(theta)
        bound1, bound2 = calculate_init_bounds(x2, y2)
        auto_theta = np.random.uniform(low=bound1, high=bound2)

        air1 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
        air1.update_params()

        air2 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
        air2.update_params()

        min_dis1 = air1.row
        min_dis2 = air2.row
        for j in range(100):
            air1.step_proof(3)
            if min_dis1 < air1.row:
                min_dis1 = air1.row
        for j in range(100):
            air2.step_proof(4)
            if min_dis2 < air2.row:
                min_dis2 = air2.row

        if (air1.Vint <= 1200 and air1.Vint >= 0) and (min_dis1 >= dis_threshold or min_dis2 >= dis_threshold):
            reward, collide_flag, states_seq = reward_func(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
            cvg = fuzzer.state_coverage(states_seq)
            current_pose = [acas_speed, x2, y2, auto_theta]
            fuzzer.further_mutation(current_pose, reward, 0, cvg, current_pose)

            input = np.array([acas_speed, x2, y2, auto_theta])
            print(f'iteration: {num_iterations}, oracle: {float(collide_flag)}, coverage: {cvg}, elapsed_time: {time.time() - start_time}', file=logs_buffer)
            np.savetxt(inputs_buffer, input.reshape(1, -1), delimiter=',')
            num_iterations += 1
            pbar.update(1)
    pbar.close()
    return fuzzer, num_iterations, inputs_buffer, logs_buffer


def sample_seeds_without_coverage(log_path: str, budget_in_minutes: int = 120, seed: int = 2020):
    '''
    Function that initializes a fuzzing object w.r.t the initialization procedure defined the paper.
    The implementation follows the original function sample_seeds(.).
    The main differences are:
    - The budget is specified in minutes
    - Logging of the exections are implemented by our own class
    This smapling version is intented to allow fuzzing without coverage guidance.
    '''
    # this is how the original work manages randomness
    np.random.seed(seed)
    dis_threshold = 200 # used to validate inputs
    fuzzer = fuzzing()

    logger = MyLogger(log_path + '_sampling_logs.txt')
    logger.write_columns()

    budget = (60 * budget_in_minutes)
    pbar = tqdm.tqdm(total=budget_in_minutes)
    start_time = time.time()
    current_time = time.time()
    last_minute_time = time.time()

    num_executions = 0
    while (current_time - start_time) < budget:
        # input generation
        acas_speed = np.random.uniform(low=10, high=1100)
        row = np.random.uniform(low=1000, high=60261)
        theta = np.random.uniform(low=-np.pi, high=np.pi)
        x2 = row * np.cos(theta)
        y2 = row * np.sin(theta)
        bound1, bound2 = calculate_init_bounds(x2, y2)
        auto_theta = np.random.uniform(low=bound1, high=bound2)

        air1 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
        air1.update_params()

        air2 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
        air2.update_params()

        min_dis1 = air1.row
        min_dis2 = air2.row
        for j in range(100):
            air1.step_proof(3)
            if min_dis1 < air1.row:
                min_dis1 = air1.row
        for j in range(100):
            air2.step_proof(4)
            if min_dis2 < air2.row:
                min_dis2 = air2.row

        # input validation
        if (air1.Vint <= 1200 and air1.Vint >= 0) and (min_dis1 >= dis_threshold or min_dis2 >= dis_threshold):
            t0 = time.time()
            reward, collide_flag, states_seq = reward_func(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
            exec_time = time.time() - t0

            num_executions += 1
            run_time = time.time() # for logging the execution of the input

            # mutates to compute the sensitivity
            break_flag = 0
            mutate_acas_speed, mutate_x2, mutate_y2, mutate_auto_theta = fuzzer.mutate(acas_speed, x2, y2, auto_theta)
            while verify(mutate_acas_speed, mutate_x2, mutate_y2, mutate_auto_theta) == False:
                break_flag += 1
                mutate_acas_speed, mutate_x2, mutate_y2, mutate_auto_theta = fuzzer.mutate(acas_speed, x2, y2, auto_theta)
                if break_flag > 50:
                    break

            t0 = time.time()
            mutate_reward, mutate_collide_flag, _ = reward_func(mutate_acas_speed, mutate_x2, mutate_y2, mutate_auto_theta)
            mutate_exec_time = time.time() - t0

            num_executions += 1
            sensitivity = np.abs(reward - mutate_reward)

            # logs the execution of the input
            logger.log(
                input=np.array([acas_speed, x2, y2, auto_theta]),
                oracle=collide_flag,
                reward=reward,
                sensitivity=sensitivity,
                test_exec_time=exec_time,
                run_time=run_time
            )
            # logs the executions of the mutant (without sensitivity)
            logger.log(
                input=np.array([mutate_acas_speed, mutate_x2, mutate_y2, mutate_auto_theta]),
                oracle=mutate_collide_flag,
                reward=mutate_reward,
                test_exec_time=mutate_exec_time,
                run_time=time.time()
            )

            current_pose = [acas_speed, x2, y2, auto_theta]
            fuzzer.further_mutation(current_pose, reward, sensitivity, 0, current_pose)

        now = time.time()
        if now - last_minute_time > 60:
            pbar.update(1)
            last_minute_time = now
            pbar.set_postfix({'Iterations': num_executions, 'Pool': len(fuzzer.corpus)})
        current_time = now
    pbar.close()
    return fuzzer, logger, num_executions


def sample_seeds_with_coverage(log_path: str, budget_in_minutes: int = 120, seed: int = 2020):
    '''
    Function that initializes a fuzzing object w.r.t the initialization procedure defined the paper.
    The implementation follows the original function sample_seeds(.).
    The main differences are:
    - The budget is specified in minutes
    - Logging of the exections are implemented by our own class
    This smapling version is intented to allow fuzzing without coverage guidance.
    '''
    # this is how the original work manages randomness
    np.random.seed(seed)
    dis_threshold = 200 # used to validate inputs
    fuzzer = fuzzing()

    logger = MyLogger(log_path + '_sampling_logs.txt')
    logger.write_columns()

    budget = (60 * budget_in_minutes)
    pbar = tqdm.tqdm(total=budget_in_minutes)
    start_time = time.time()
    current_time = time.time()
    last_minute_time = time.time()

    num_executions = 0
    while (current_time - start_time) < budget:
        # input generation
        acas_speed = np.random.uniform(low=10, high=1100)
        row = np.random.uniform(low=1000, high=60261)
        theta = np.random.uniform(low=-np.pi, high=np.pi)
        x2 = row * np.cos(theta)
        y2 = row * np.sin(theta)
        bound1, bound2 = calculate_init_bounds(x2, y2)
        auto_theta = np.random.uniform(low=bound1, high=bound2)

        air1 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
        air1.update_params()

        air2 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
        air2.update_params()

        min_dis1 = air1.row
        min_dis2 = air2.row
        for j in range(100):
            air1.step_proof(3)
            if min_dis1 < air1.row:
                min_dis1 = air1.row
        for j in range(100):
            air2.step_proof(4)
            if min_dis2 < air2.row:
                min_dis2 = air2.row

        # input validation
        if (air1.Vint <= 1200 and air1.Vint >= 0) and (min_dis1 >= dis_threshold or min_dis2 >= dis_threshold):
            t0 = time.time()
            reward, collide_flag, states_seq = reward_func(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
            exec_time = time.time() - t0

            t0 = time.time()
            cvg = fuzzer.state_coverage(states_seq)
            coverage_time = time.time() - t0

            num_executions += 1
            run_time = time.time() # for logging the execution of the input

            # mutates to compute the sensitivity
            break_flag = 0
            mutate_acas_speed, mutate_x2, mutate_y2, mutate_auto_theta = fuzzer.mutate(acas_speed, x2, y2, auto_theta)
            while verify(mutate_acas_speed, mutate_x2, mutate_y2, mutate_auto_theta) == False:
                break_flag += 1
                mutate_acas_speed, mutate_x2, mutate_y2, mutate_auto_theta = fuzzer.mutate(acas_speed, x2, y2, auto_theta)
                if break_flag > 50:
                    break

            t0 = time.time()
            mutate_reward, mutate_collide_flag, _ = reward_func(mutate_acas_speed, mutate_x2, mutate_y2, mutate_auto_theta)
            mutate_exec_time = time.time() - t0

            num_executions += 1
            sensitivity = np.abs(reward - mutate_reward)

            # logs the execution of the input
            logger.log(
                input=np.array([acas_speed, x2, y2, auto_theta]),
                oracle=collide_flag,
                reward=reward,
                sensitivity=sensitivity,
                coverage=cvg,
                test_exec_time=exec_time,
                coverage_time=coverage_time,
                run_time=run_time
            )
            # logs the executions of the mutant (without coverage nor sensitivity)
            logger.log(
                input=np.array([mutate_acas_speed, mutate_x2, mutate_y2, mutate_auto_theta]),
                oracle=mutate_collide_flag,
                reward=mutate_reward,
                test_exec_time=mutate_exec_time,
                run_time=time.time()
            )

            current_pose = [acas_speed, x2, y2, auto_theta]
            fuzzer.further_mutation(current_pose, reward, sensitivity, cvg, current_pose)

        now = time.time()
        if now - last_minute_time > 60:
            pbar.update(1)
            last_minute_time = now
            pbar.set_postfix({'Iterations': num_executions, 'Pool': len(fuzzer.corpus)})
        current_time = now
    pbar.close()
    return fuzzer, logger, num_executions


def random_sampling(log_path: str, budget_in_minutes: int = 720, seed: int = 2020):
    '''
    Slightly adapted version of what *we suspect to be* the implementation of the Fuzzer baseline.
    The changes only consist of logging feature.
    In other words, we don't consider this function as a correct implementation of the Fuzzer, and adapted it for comparison.
    '''
    tau = 0.01
    np.random.seed(seed)
    fuzzer = fuzzing()

    original_log_file = open(log_path + '.txt', 'w', buffering=1)
    logger = MyLogger(log_path + '_fuzzing_logs.txt')
    logger.write_columns()

    budget = (60 * budget_in_minutes)
    num_executions, num_original_faults = 0, 0
    pbar = tqdm.tqdm(total=budget_in_minutes)

    start_time = time.time()
    current_time = time.time()
    last_minute_time = time.time()

    while (current_time - start_time) < budget:
        # random input generation
        acas_speed = np.random.uniform(low=10, high=1100)
        row = np.random.uniform(low=1000, high=60261)
        theta = np.random.uniform(low=-np.pi, high=np.pi)
        x2 = row * np.cos(theta)
        y2 = row * np.sin(theta)
        bound1, bound2 = calculate_init_bounds(x2, y2)
        auto_theta = np.random.uniform(low=bound1, high=bound2)
        # input validation and execution
        verify_flag = verify(acas_speed, x2, y2, auto_theta)
        if verify_flag == True:
            t0 = time.time()
            reward, collide_flag, states_seq = reward_func(acas_speed, x2, y2, auto_theta)
            run_time = time.time()
            coverage = fuzzer.state_coverage(states_seq)
            coverage_time = time.time() - run_time
            exec_time = run_time - t0
            if collide_flag and (coverage < tau):
                num_original_faults += 1
            # logs the execution
            print(num_original_faults, file=original_log_file)
            num_executions += 1
            logger.log(
                input=np.array([acas_speed, x2, y2, auto_theta]),
                oracle=collide_flag,
                reward=reward,
                coverage=coverage,
                test_exec_time=exec_time,
                coverage_time=coverage_time,
                run_time=run_time
            )

        current_time = time.time()
        if current_time - last_minute_time > 60:
            pbar.update(1)
            last_minute_time = current_time
            pbar.set_postfix({'Iteration': num_executions, 'Num. original faults': num_original_faults})

    pbar.close()
    original_log_file.close()


def verify(acas_speed, x2, y2, auto_theta):
    dis_threshold = 200
    air1 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
    air1.update_params()

    air2 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
    air2.update_params()

    min_dis1 = air1.row
    min_dis2 = air2.row
    for j in range(100):
        air1.step_proof(3)
        if min_dis1 > air1.row:
            min_dis1 = air1.row
    for j in range(100):
        air2.step_proof(4)
        if min_dis2 > air2.row:
            min_dis2 = air2.row
    # print(min_dis1, min_dis2)
    if (air1.Vint <= 1200 and air1.Vint >= 0) and (min_dis1 >= dis_threshold or min_dis2 >= dis_threshold):
        return True
    else:
        return False


def normalize_state(x):
    y = copy.deepcopy(x)
    y = np.array(y)
    y = y - np.array([1.9791091e+04, 0.0, 0.0, 650.0, 600.0])
    y = y / np.array([60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0])
    return y.tolist()


def reward_func(acas_speed, x2, y2, auto_theta):
    dis_threshold = 200
    air1 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
    air1.update_params()
    gamma = 0.99
    min_dis1 = np.inf
    reward = 0
    collide_flag = False
    states_seq = []

    for j in range(100):
        air1.step()
        reward = reward * gamma + air1.row / 60261.0
        states_seq.append(normalize_state([air1.row, air1.alpha, air1.phi, air1.Vown, air1.Vint]))
        if air1.row < dis_threshold:
            collide_flag = True
            reward -= 100

    return reward, collide_flag, states_seq


def fuzz_with_coverage(fuzzer: fuzzing, budget_in_minutes: int, log_path: str):
    tau = 0.01

    logger = MyLogger(log_path + '_fuzzing_logs.txt')
    logger.write_columns()

    budget = 60 * budget_in_minutes
    pbar = tqdm.tqdm(total=budget_in_minutes)
    start_time = time.time()
    last_minute_time = time.time()
    while len(fuzzer.corpus) > 0:
        # sensitivity-biased input selection
        orig_pose = fuzzer.get_pose()
        [orig_acas_speed, orig_x2, orig_y2, orig_auto_theta] = orig_pose
        # input mutation and validation
        acas_speed, x2, y2, auto_theta = fuzzer.mutate(orig_acas_speed, orig_x2, orig_y2, orig_auto_theta)
        break_flag = 0
        while verify(acas_speed, x2, y2, auto_theta) == False:
            break_flag += 1
            acas_speed, x2, y2, auto_theta = fuzzer.mutate(orig_acas_speed, orig_x2, orig_y2, orig_auto_theta)
            if break_flag > 50:
                break
        if break_flag > 50:
            fuzzer.drop_current()
            continue

        t0 = time.time()
        reward, collide_flag, states_seq = reward_func(acas_speed, x2, y2, auto_theta)
        exec_time = time.time() - t0
        t0 = time.time()
        cvg = fuzzer.state_coverage(states_seq)
        coverage_time = time.time() - t0

        local_sensitivity = np.abs(reward - fuzzer.current_reward)
        # logs the new execution
        logger.log(
                input=np.array([acas_speed, x2, y2, auto_theta]),
                oracle=collide_flag,
                reward=reward,
                sensitivity=local_sensitivity,
                coverage=cvg,
                test_exec_time=exec_time,
                coverage_time=coverage_time,
                run_time=time.time()
            )

        if collide_flag:
            fuzzer.add_crash([acas_speed, x2, y2, auto_theta])

        elif (reward < fuzzer.current_reward) or (cvg < tau):
            current_pose = [acas_speed, x2, y2, auto_theta]
            orig_pose = fuzzer.current_original
            fuzzer.further_mutation(current_pose, reward, local_sensitivity, cvg, orig_pose)

        now = time.time()
        if now - last_minute_time > 60:
            pbar.update(1)
            last_minute_time = now
            pbar.set_postfix({'Found': len(fuzzer.result), 'Pool': len(fuzzer.corpus)})
        if now - start_time > budget:
            break

    pbar.close()

    # with open(path + '_crash.pkl', 'wb') as handle:
    #     pickle.dump(fuzzer.result, handle, protocol=pickle.HIGHEST_PROTOCOL)


def fuzz_without_coverage(fuzzer: fuzzing, budget_in_minutes: int, log_path: str):
    logger = MyLogger(log_path + '_fuzzing_logs.txt')
    logger.write_columns()
    budget = 60 * budget_in_minutes
    pbar = tqdm.tqdm(total=budget_in_minutes)
    start_time = time.time()
    last_minute_time = time.time()
    while len(fuzzer.corpus) > 0:
        # sensitivity-biased input selection
        orig_pose = fuzzer.get_pose()
        [orig_acas_speed, orig_x2, orig_y2, orig_auto_theta] = orig_pose
        # input mutation and validation
        acas_speed, x2, y2, auto_theta = fuzzer.mutate(orig_acas_speed, orig_x2, orig_y2, orig_auto_theta)
        break_flag = 0
        while verify(acas_speed, x2, y2, auto_theta) == False:
            break_flag += 1
            acas_speed, x2, y2, auto_theta = fuzzer.mutate(orig_acas_speed, orig_x2, orig_y2, orig_auto_theta)
            if break_flag > 50:
                break
        if break_flag > 50:
            fuzzer.drop_current()
            continue

        t0 = time.time()
        reward, collide_flag, states_seq = reward_func(acas_speed, x2, y2, auto_theta)
        exec_time = time.time() - t0

        local_sensitivity = np.abs(reward - fuzzer.current_reward)
        # logs the new execution
        logger.log(
                input=np.array([acas_speed, x2, y2, auto_theta]),
                oracle=collide_flag,
                reward=reward,
                sensitivity=local_sensitivity,
                test_exec_time=exec_time,
                run_time=time.time()
            )

        if collide_flag:
            fuzzer.add_crash([acas_speed, x2, y2, auto_theta])

        elif (reward < fuzzer.current_reward):
            current_pose = [acas_speed, x2, y2, auto_theta]
            orig_pose = fuzzer.current_original
            fuzzer.further_mutation(current_pose, reward, local_sensitivity, 0, orig_pose)

        now = time.time()
        if now - last_minute_time > 60:
            pbar.update(1)
            last_minute_time = now
            pbar.set_postfix({'Found': len(fuzzer.result), 'Pool': len(fuzzer.corpus)})
        if now - start_time > budget:
            break

    pbar.close()

    # with open(path + '_crash.pkl', 'wb') as handle:
    #     pickle.dump(fuzzer.result, handle, protocol=pickle.HIGHEST_PROTOCOL)


def replay(pickle_path):
    with open(pickle_path, 'rb') as handle:
        result = pickle.load(handle)

    for i in range(len(result)):
        for m in range(3):
            if m == 0:
                X1 = []
                Y1 = []
                X2 = []
                Y2 = []
                [acas_speed, x2, y2, auto_theta] = result[i]
                air = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
                air.update_params()
                min_row = air.row
                print('Vint: ', air.Vint)
                for j in range(100):
                    X1.append(air.ownship.x)
                    Y1.append(air.ownship.y)
                    X2.append(air.inturder.x)
                    Y2.append(air.inturder.y)
                    air.step()
                    if min_row > air.row:
                        min_row = air.row
                print('Min distance: ', min_row)
                min_x = np.min([np.min(X1), np.min(X2)])
                min_x -= 100
                max_x = np.max([np.max(X1), np.max(X2)])
                max_x += 100
                min_y = np.min([np.min(Y1), np.min(Y2)])
                min_y -= 100
                max_y = np.max([np.max(Y1), np.max(Y2)])
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
                    line1.set_data(X1[:k], Y1[:k])
                    line2.set_data(X2[:k], Y2[:k])
                    return line1, line2
                ani = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=len(X1), interval=1, blit=True)
                gif_name = './results/artifact/crash/' + str(i) + '_' + str(min_row) + '.gif'
                ani.save(gif_name)
            else:
                pass
                X1_3 = []
                Y1_3 = []
                X2_3 = []
                Y2_3 = []
                X1_4 = []
                Y1_4 = []
                X2_4 = []
                Y2_4 = []

                [acas_speed, x2, y2, auto_theta] = result[i]
                air3 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
                air4 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
                air3.update_params()
                air4.update_params()

                min_row_3 = air3.row
                min_row_4 = air4.row
                for j in range(100):
                    X1_3.append(air3.ownship.x)
                    Y1_3.append(air3.ownship.y)
                    X2_3.append(air3.inturder.x)
                    Y2_3.append(air3.inturder.y)
                    X1_4.append(air4.ownship.x)
                    Y1_4.append(air4.ownship.y)
                    X2_4.append(air4.inturder.x)
                    Y2_4.append(air4.inturder.y)
                    air3.step_proof(3)
                    air4.step_proof(4)
                    if min_row_3 > air3.row:
                        min_row_3 = air3.row
                    if min_row_4 > air4.row:
                        min_row_4 = air4.row
                if min_row_3 > min_row_4:
                    X1 = X1_3
                    Y1 = Y1_3
                    X2 = X2_3
                    Y2 = Y2_3
                    min_row = min_row_3
                else:
                    X1 = X1_4
                    Y1 = Y1_4
                    X2 = X2_4
                    Y2 = Y2_4
                    min_row = min_row_4

                min_x = np.min([np.min(X1), np.min(X2)])
                min_x -= 100
                max_x = np.max([np.max(X1), np.max(X2)])
                max_x += 100
                min_y = np.min([np.min(Y1), np.min(Y2)])
                min_y -= 100
                max_y = np.max([np.max(Y1), np.max(Y2)])
                max_y += 100
                fig = plt.figure()
                ax = plt.axes(xlim=(min_x, max_x), ylim=(min_y, max_y))
                line1, = ax.plot([], [], lw=2, c='red')
                line2, = ax.plot([], [], lw=2, c='blue')

                def init():
                    line1.set_data([], [])
                    line2.set_data([], [])
                    return line1, line2
                def animate(k):
                    line1.set_data(X1[:k], Y1[:k])
                    line2.set_data(X2[:k], Y2[:k])
                    return line1, line2
                ani = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=len(X1), interval=1, blit=True)
                gif_name = './results/artifact/noncrash/' + str(i) + '_' + str(min_row) + '.gif'
                ani.save(gif_name)


def replay_repair(pickle_path, render=False):
    with open(pickle_path, 'rb') as handle:
        result = pickle.load(handle)
    count_nocrash = 0
    count_crash = 0
    dis_threshold = 200
    for i in range(len(result)):
        X1 = []
        Y1 = []
        X2 = []
        Y2 = []
        [acas_speed, x2, y2, auto_theta] = result[i]
        air = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
        air.update_params()
        min_row = air.row
        for j in range(100):
            X1.append(air.ownship.x)
            Y1.append(air.ownship.y)
            X2.append(air.inturder.x)
            Y2.append(air.inturder.y)
            air.step()
            if min_row > air.row:
                min_row = air.row
        if min_row < dis_threshold:
            count_crash += 1
        else:
            count_nocrash += 1
        print('Min distance: ', min_row)
        # print(temp)
        if render:
            min_x = np.min([np.min(X1), np.min(X2)])
            min_x -= 100
            max_x = np.max([np.max(X1), np.max(X2)])
            max_x += 100
            min_y = np.min([np.min(Y1), np.min(Y2)])
            min_y -= 100
            max_y = np.max([np.max(Y1), np.max(Y2)])
            max_y += 100
            fig = plt.figure()
            ax = plt.axes(xlim=(min_x, max_x), ylim=(min_y, max_y))
            line1, = ax.plot([], [], lw=2, c='red')
            line2, = ax.plot([], [], lw=2, c='blue')

            def init():
                line1.set_data([], [])
                line2.set_data([], [])
                return line1, line2
            def animate(k):
                line1.set_data(X1[:k], Y1[:k])
                line2.set_data(X2[:k], Y2[:k])
                return line1, line2
            ani = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=len(X1), interval=1, blit=True)
            gif_name = './results/artifact/crash/' + str(i) + '_' + str(min_row) + '.gif'
            # ani = animation.ArtistAnimation(fig, tmp, interval=1, repeat_delay=0, blit=True)
            ani.save(gif_name)
    print('crash: ', count_crash, ', no crash: ', count_nocrash)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--repair', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--no_coverage', action='store_true', default=False)
    parser.add_argument('--terminate', type=float, default=1.0)
    parser.add_argument('--seed_size', type=float, default=50)
    parser.add_argument('--picklepath', type=str, default='')
    parser.add_argument('--seed', type=int, default=2020)

    parser.add_argument('--init_budget', type=int, default=120, help="Time for initializing fuzzing (in minutes)")
    parser.add_argument('--fuzz_budget', type=int, default=720, help="Time for fuzzing (in minutes)")

    args = parser.parse_args()

    if args.replay and args.repair:
        # replay(args.picklepath)
        replay_repair(args.picklepath, args.render)
    elif args.replay:
        replay(args.picklepath)
    else:
        seed = args.seed
        # as minutes
        init_budget = args.init_budget  # type: int
        fuzz_budget = args.fuzz_budget  # type: int
        data_path = '../data/'
        folder = 'acas/'
        if not os.path.isdir(data_path + folder):
            os.mkdir(data_path + folder)
        path = data_path + folder
        if args.random:
            subfolder = 'original_fuzzer/'
        elif args.no_coverage:
            subfolder = 'fuzzer/'
        else:
            subfolder = 'mdpfuzz/'

        if not os.path.isdir(path + subfolder):
            os.mkdir(path + subfolder)

        path += subfolder
        path += str(seed)

        if args.random:
            random_sampling(path, budget_in_minutes=fuzz_budget, seed=seed)
        else:
            if args.no_coverage:
                fuzzer, _logger, num_iterations = sample_seeds_without_coverage(path, budget_in_minutes=init_budget, seed=seed)
            else:
                fuzzer, _logger, num_iterations = sample_seeds_with_coverage(path, budget_in_minutes=init_budget, seed=seed)

            print('Corpus of size {} initialized after {} executions'.format(len(fuzzer.corpus), num_iterations))

            if args.random:
                fuzz_without_coverage(fuzzer, fuzz_budget, path)
            else:
                fuzz_with_coverage(fuzzer, fuzz_budget, path)