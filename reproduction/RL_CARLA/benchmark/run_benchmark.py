import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from typing import List, Tuple
import pandas as pd
import tqdm
import bird_view.utils.carla_utils as cu
from fuzz.fuzz import fuzzing
from fuzz.replayer import replayer
import numpy as np
import copy
import time
import pickle

import carla
from carla import ColorConverter
from carla import WeatherParameters

from collections import OrderedDict

# implementation of our logs
import sys
sys.path.append('../logger/')
from logger import MyLogger


def calculate_reward(prev_distance, cur_distance, cur_collid, cur_invade, cur_speed, prev_speed):
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

def collect_corpus():
    return None

def get_index(sequence):
    result = np.zeros(17)
    mins = [-10, 90, -1, -1, -10, -10, -1, -20, -20, -5, -10, 90, 0, 1, 50, 50, 0]
    maxs = [200, 350, 1, 1, 10, 10, 1, 20, 20, 5, 200, 350, 0.5, 4, 300, 300, 15]
    for i in range(sequence.shape[0]):
        if sequence[i] < mins[i]:
            result[i] = 0
        elif sequence[i] > maxs[i]:
            result[i] = 10000
        else:
            result_i = (sequence[i] - mins[i]) / (maxs[i] - mins[i]) * 10000 + 1
            if np.isnan(result_i):
                result[i] = 10000
            else:
                result[i] = int(result_i)
    return result

def update_dict(storage, sequences):
    for i in range(len(sequences)):
        index = get_index(sequences[i])
        for j in range(index.shape[0]):
            storage[int(j * 10000 + index[j])] = 1
    return storage

def run_single(env, weather, start, target, agent_maker, seed, path: str, em_guide: bool, replay=False):
    # print(start, target, weather)
    # # env.replayer = replayer()
    env.init(start=start, target=target, weather=cu.PRESET_WEATHERS[weather])
    agent = agent_maker()
    # HACK: deterministic vehicle spawns.)
    env.seed = seed
    env.init(start=start, target=target, weather=cu.PRESET_WEATHERS[weather])
    print('run_single(.) running with input values:')
    print('replay:', replay, 'em guide:', em_guide, 'start:', start, 'target:', target, 'weather:', weather)
    ########### replay is False ###########
    if replay == True:
        prints = env._blueprints.filter("vehicle.*")
        replayer_load = load_pickle('./results/EM_crash.pkl', prints)
        print('Load Done!')
        env.replay = True
        env.replayer = replayer_load
        print(env.replayer.rewards)
        env.replayer.replay_list = [14, 24, 26, 28, 29, 30, 32, 34, 41, 42, 44, 45, 46, 54, 64, 66, 70, 71, 73, 75, 76, 78, 80, 81, 83, 95]
        # env.replayer.replay_list = [73]
        agent = agent_maker()
        tsne_data = []

        while len(env.replayer.replay_list) > 0:
            print('current id: ', env.replayer.replay_list[-1])
            env.init(start=start, target=target, weather=cu.PRESET_WEATHERS[weather])
            diagnostics = list()
            result = {
                "weather": weather,
                "start": start,
                "target": target,
                "success": None,
                "t": None,
                "total_lights_ran": None,
                "total_lights": None,
                "collided": None,
            }
            seq_entropy = 0

            first_reward_flag = True
            total_reward = 0
            sequence = []
            while env.tick():
                observations = env.get_observations()
                if first_reward_flag == False:
                    cur_invade = (prev_invaded_frame_number != env._invaded_frame_number)
                    cur_collid = (prev_collided_frame_number != env._collided_frame_number)
                if first_reward_flag:
                    first_reward_flag = False
                    prev_distance = env._local_planner.distance_to_goal
                    prev_speed = observations['velocity']
                    prev_invaded_frame_number = env._invaded_frame_number
                    prev_collided_frame_number = env._collided_frame_number
                    cur_invade = False
                    cur_collid = False
                    if env.invaded:
                        cur_invade = True
                    if env.collided:
                        cur_collid = True

                reward = calculate_reward(prev_distance, env._local_planner.distance_to_goal, cur_collid, cur_invade, observations['velocity'], prev_speed)
                total_reward += reward
                prev_distance = env._local_planner.distance_to_goal
                prev_speed = observations['velocity']
                prev_invaded_frame_number = env._invaded_frame_number
                prev_collided_frame_number = env._collided_frame_number

                control, current_entropy, _ = agent.run_step(observations)
                seq_entropy += current_entropy
                diagnostic = env.apply_control(control)
                diagnostic.pop("viz_img")
                diagnostics.append(diagnostic)
                # HACK: T-SNE
                # current_tsne = np.array(current_tsne).flatten()
                # tsne_data.append(current_tsne)
                if env.is_failure() or env.is_success() or env._tick > 100:
                    result["success"] = env.is_success()
                    result["total_lights_ran"] = env.traffic_tracker.total_lights_ran
                    result["total_lights"] = env.traffic_tracker.total_lights
                    result["collided"] = env.collided
                    result["t"] = env._tick
                    break
            print(total_reward)

        # HACK: T-SNE
        # tsne_data = np.array(tsne_data)
        # print(tsne_data.shape)
        # with open('../../results/tsne_crash.pkl', 'wb') as tsne_file:
        #     np.save(tsne_file, tsne_data)

    else:
        env.replayer = replayer()
        fuzzer = fuzzing()
        env.fuzzer = fuzzer
        agent = agent_maker()
        # tsne_data = []
        # min_seq = []
        # max_seq = []
        # k_storage_state_cvg = {}
        # count_of_GMM = 0

        # used for debugging
        # result = {
        #     "weather": weather,
        #     "start": start,
        #     "target": target,
        #     "success": False,
        #     "t": 0,
        #     "total_lights_ran": 0,
        #     "total_lights": 0,
        #     "collided": False,
        # }

        # doubled original times (our hardware did not have GPU)
        init_budget_in_minutes = 120 * 2
        fuzzing_budget_in_minutes = 720 * 2

        # no_coverage indicates whether to skip the coverage computation
        no_coverage = not em_guide


        print('receive path:', path)
        if not os.path.isdir(path):
            path = '../../' + path
            print('new path', path)

        if no_coverage:
            path += 'fuzzer/'
        else:
            path += 'mdpfuzz/'

        if not os.path.isdir(path):
            os.mkdir(path)

        path += str(seed)

        logger = MyLogger(path + '_sampling_logs.txt')
        logger.write_columns()

        f = open(path + '.txt', 'w', buffering=1)
        sys.stdout = f


        budget = (60 * init_budget_in_minutes)
        pbar = tqdm.tqdm(total=init_budget_in_minutes)
        start_time = time.time()
        last_minute_time = time.time()
        num_executions = 0

        # diagnostics = list()
        # results = []
        # sampling starts
        for weather, (start, target), run_name in tqdm.tqdm(env.all_tasks, total=len(list(env.all_tasks))):
            try:
                t0 = time.time()
                # executes a random input
                env.init(start=start, target=target, weather=cu.PRESET_WEATHERS[weather])
                start_pose = env._start_pose

                # diagnostics = list()
                # result = {
                #     "start": start,
                #     "target": target,
                #     "weather": weather,
                #     "reward": 0,
                #     "t": env._tick,
                #     "success": env.is_success(),
                #     "collided": env.collided,
                #     "total_lights": env.traffic_tracker.total_lights,
                #     "total_lights_ran": env.traffic_tracker.total_lights_ran
                # }
                seq_entropy = 0
                first_reward_flag = True
                total_reward = 0
                sequence = []
                # executes the input
                while env.tick():
                    observations = env.get_observations()
                    if first_reward_flag == False:
                        cur_invade = (prev_invaded_frame_number != env._invaded_frame_number)
                        cur_collid = (prev_collided_frame_number != env._collided_frame_number)
                    if first_reward_flag:
                        first_reward_flag = False
                        prev_distance = env._local_planner.distance_to_goal
                        prev_speed = observations['velocity']
                        prev_invaded_frame_number = env._invaded_frame_number
                        prev_collided_frame_number = env._collided_frame_number
                        cur_invade = False
                        cur_collid = False
                        if env.invaded:
                            cur_invade = True
                        if env.collided:
                            cur_collid = True
                    reward = calculate_reward(prev_distance, env._local_planner.distance_to_goal, cur_collid, cur_invade, observations['velocity'], prev_speed)
                    total_reward += reward
                    prev_distance = env._local_planner.distance_to_goal
                    prev_speed = observations['velocity']
                    prev_invaded_frame_number = env._invaded_frame_number
                    prev_collided_frame_number = env._collided_frame_number
                    control, current_entropy, _ = agent.run_step(observations)
                    temp = copy.deepcopy(observations['node'])
                    temp = np.hstack((temp, copy.deepcopy(observations['orientation']), copy.deepcopy(observations['velocity']), copy.deepcopy(observations['acceleration']), copy.deepcopy(observations['position']), copy.deepcopy(np.array([observations['command']]))))
                    vehicle_index = np.nonzero(observations['vehicle'])
                    vehicle_obs = np.zeros(3)
                    vehicle_obs[0] = vehicle_index[0].mean()
                    vehicle_obs[1] = vehicle_index[1].mean()
                    vehicle_obs[2] = np.sum(observations['vehicle']) / 1e5
                    temp = np.hstack((temp, vehicle_obs))
                    seq_entropy += current_entropy
                    env.apply_control(control)
                    # diagnostic = env.apply_control(control)
                    # diagnostic.pop("viz_img")
                    # diagnostics.append(diagnostic)
                    sequence.append(temp)
                    if env.is_failure() or env.is_success() or env._tick > 100:
                        # result = {
                        #     "start": start,
                        #     "target": target,
                        #     "weather": weather,
                        #     "reward": total_reward,
                        #     "t": env._tick,
                        #     "success": env.is_success(),
                        #     "collided": env.collided,
                        #     "total_lights": env.traffic_tracker.total_lights,
                        #     "total_lights_ran": env.traffic_tracker.total_lights_ran
                        # }
                        print("Finalized after {} steps; crash: ".format(env._tick), env.collided)
                        break
                # print(result)
                run_time = time.time()
                exec_time = run_time - t0
                num_executions += 1

                if no_coverage:
                    coverage = coverage_time = None
                else:
                    t0 = time.time()
                    coverage = env.fuzzer.state_coverage(sequence)
                    coverage_time = time.time() - t0

                env.fuzzer.further_mutation((start_pose, env.init_vehicles), rewards=total_reward, entropy=seq_entropy, cvg=coverage, original=(start_pose, env.init_vehicles), further_envsetting=[start, target, cu.PRESET_WEATHERS[weather], weather])
                # logs the execution
                crash_flag = env.is_failure() or env.collided
                logger.log(
                    oracle=crash_flag,
                    reward=total_reward,
                    coverage=coverage,
                    sensitivity=seq_entropy,
                    test_exec_time=exec_time,
                    coverage_time=coverage_time,
                    run_time=run_time
                )
            except:
                print('something went wrong during initialization (step {})'.format(num_executions))
                continue
            # update the pbar
            now = time.time()
            if now - last_minute_time > 60:
                pbar.update(1)
                last_minute_time = now
                pbar.set_postfix({'Iterations': num_executions, 'Pool': len(fuzzer.corpus)})
            if (now - start_time) > budget:
                break

        pbar.close()
        print('Corpus of size {} initialized after {} executions'.format(len(fuzzer.corpus), num_executions))

        logger = MyLogger(path + '_fuzzing_logs.txt')
        logger.write_columns()

        budget = (60 * fuzzing_budget_in_minutes)
        pbar2 = tqdm.tqdm(total=fuzzing_budget_in_minutes)
        start_time = time.time()
        last_minute_time = time.time()
        num_executions = 0

        env.first_run = False
        while len(env.fuzzer.corpus) > 0:
            t0 = time.time()
            # since first_run is False, init(.) performs (internally) input selection and mutation with the fuzzer
            # the code is intricated
            try:
                initial_check = env.init(start=start, target=target, weather=cu.PRESET_WEATHERS[weather])
            except:
                initial_check = False
            if initial_check == False:
                print('Trigger initial collision!')
                env.fuzzer.drop_current()
                continue

            seq_entropy = 0
            first_reward_flag = True
            total_reward = 0
            sequence = []

            while env.tick():
                observations = env.get_observations()

                if first_reward_flag == False:
                    cur_invade = (prev_invaded_frame_number != env._invaded_frame_number)
                    cur_collid = (prev_collided_frame_number != env._collided_frame_number)
                if first_reward_flag:
                    first_reward_flag = False
                    prev_distance = env._local_planner.distance_to_goal
                    prev_speed = observations['velocity']
                    prev_invaded_frame_number = env._invaded_frame_number
                    prev_collided_frame_number = env._collided_frame_number
                    cur_invade = False
                    cur_collid = False
                    if env.invaded:
                        cur_invade = True
                    if env.collided:
                        cur_collid = True
                reward = calculate_reward(prev_distance, env._local_planner.distance_to_goal, cur_collid, cur_invade, observations['velocity'], prev_speed)
                total_reward += reward
                prev_distance = env._local_planner.distance_to_goal
                prev_speed = observations['velocity']
                prev_invaded_frame_number = env._invaded_frame_number
                prev_collided_frame_number = env._collided_frame_number

                control, current_entropy, _ = agent.run_step(observations)
                temp = copy.deepcopy(observations['node'])
                temp = np.hstack((temp, copy.deepcopy(observations['orientation']), copy.deepcopy(observations['velocity']), copy.deepcopy(observations['acceleration']), copy.deepcopy(observations['position']), copy.deepcopy(np.array([observations['command']]))))
                vehicle_index = np.nonzero(observations['vehicle'])
                vehicle_obs = np.zeros(3)
                vehicle_obs[0] = vehicle_index[0].mean()
                vehicle_obs[1] = vehicle_index[1].mean()
                vehicle_obs[2] = np.sum(observations['vehicle']) / 1e5
                temp = np.hstack((temp, vehicle_obs))
                seq_entropy += current_entropy
                # diagnostic = env.apply_control(control)
                env.apply_control(control)
                # diagnostic.pop("viz_img")
                # diagnostics.append(diagnostic)

                sequence.append(copy.deepcopy(temp))

                if env.is_failure() or env.is_success() or env._tick > 100:
                    # result["success"] = env.is_success()
                    # result["total_lights_ran"] = env.traffic_tracker.total_lights_ran
                    # result["total_lights"] = env.traffic_tracker.total_lights
                    # result["collided"] = env.collided
                    # result["t"] = env._tick
                    break
            run_time = time.time()
            exec_time = run_time - t0
            num_executions += 1

            if no_coverage:
                coverage = coverage_time = None
            else:
                t0 = time.time()
                coverage = env.fuzzer.state_coverage(sequence)
                coverage_time = time.time() - t0

            crash_flag = env.is_failure() or env.collided
            logger.log(
                oracle=crash_flag,
                reward=total_reward,
                coverage=coverage,
                sensitivity=seq_entropy,
                test_exec_time=exec_time,
                coverage_time=coverage_time,
                run_time=run_time
            )

            if crash_flag:
                env.replayer.store((env.fuzzer.current_pose, env.fuzzer.current_vehicle_info), rewards=total_reward, entropy=seq_entropy, cvg=coverage, original=env.fuzzer.current_original, further_envsetting=env.fuzzer.current_envsetting)
                env.fuzzer.add_crash(env.fuzzer.current_pose)
                print('fuzzing iteration:', num_executions, ', found:', len(env.fuzzer.result))
            elif (total_reward < fuzzer.current_reward) or (em_guide and (coverage < env.fuzzer.GMMthreshold)):
                env.fuzzer.further_mutation((env.fuzzer.current_pose, env.fuzzer.current_vehicle_info), rewards=total_reward, entropy=seq_entropy, cvg=coverage, original=env.fuzzer.current_original, further_envsetting=env.fuzzer.current_envsetting)

            # update the pbar
            now = time.time()
            if (now - last_minute_time) > 60:
                pbar2.update(1)
                last_minute_time = now
                pbar2.set_postfix({'Iterations': num_executions, 'Pool': len(fuzzer.corpus)})
            if (now - start_time) > budget:
                break

        pbar2.close()
        f.close()
        sys.stdout = sys.__stdout__


def save_pickle(replayer):
    corpus = []
    total_crash = len(replayer.corpus)
    for i in range(total_crash):
        single_crash = []
        temp_trans = replayer.corpus[i][0]
        print(temp_trans)
        single_crash.append([temp_trans.location.x, temp_trans.location.y, temp_trans.location.z, temp_trans.rotation.pitch, temp_trans.rotation.yaw, temp_trans.rotation.roll])
        temp_vehicleinfo = replayer.corpus[i][1]
        total_vehicle = len(temp_vehicleinfo)
        vehicle_info_crash = []
        for j in range(total_vehicle):
            temp_blue_print = temp_vehicleinfo[j][0]
            temp_trans = temp_vehicleinfo[j][1]
            temp_color = temp_vehicleinfo[j][2]
            temp_vehicle_id = temp_vehicleinfo[j][3]
            vehicle_info_crash.append([temp_blue_print.id, temp_blue_print.tags, temp_trans.location.x, temp_trans.location.y, temp_trans.location.z, temp_trans.rotation.pitch, temp_trans.rotation.yaw, temp_trans.rotation.roll, temp_color, temp_vehicle_id])
        corpus.append([single_crash, vehicle_info_crash])
        replayer.envsetting[i][2] = replayer.envsetting[i][3]

    replayer.corpus = corpus
    replayer.original = []
    with open('../../results/noEM_crash.pkl', 'wb') as handle:
        pickle.dump(replayer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Pickle Saved!!!')

def load_pickle(pickle_path, prints):
    with open(pickle_path, 'rb') as handle:
        replayer = pickle.load(handle)
    corpus = []
    envsetting = []
    total_crash = len(replayer.corpus)

    for i in range(total_crash):
        temp_trans = replayer.corpus[i][0][0]
        single_trans = carla.Transform(carla.Location(x=temp_trans[0], y=temp_trans[1], z=temp_trans[2]), carla.Rotation(pitch=temp_trans[3], yaw=temp_trans[4], roll=temp_trans[5]))
        vehicle_info_crash = replayer.corpus[i][1]
        total_vehicle = len(vehicle_info_crash)
        singel_vehicle = []
        for j in range(total_vehicle):
            blue_print = prints.filter(vehicle_info_crash[j][0])[0]
            assert blue_print.tags == vehicle_info_crash[j][1]
            blue_print.set_attribute("role_name", "autopilot")

            color = vehicle_info_crash[j][8]
            vehicle_id = vehicle_info_crash[j][9]
            if color != None:
                blue_print.set_attribute("color", color)
            if vehicle_id != None:
                blue_print.set_attribute("driver_id", vehicle_id)

            trans = carla.Transform(carla.Location(x=vehicle_info_crash[j][2], y=vehicle_info_crash[j][3], z=vehicle_info_crash[j][4]), carla.Rotation(pitch=vehicle_info_crash[j][5], yaw=vehicle_info_crash[j][6], roll=vehicle_info_crash[j][7]))

            singel_vehicle.append((blue_print, trans))

        corpus.append((single_trans, singel_vehicle))

        envsetting.append([replayer.envsetting[i][0], replayer.envsetting[i][1], cu.PRESET_WEATHERS[replayer.envsetting[i][2]], replayer.envsetting[i][3]])
        # replayer.envsetting[i][2] = cu.PRESET_WEATHERS[replayer.envsetting[i][2]]
    replayer.corpus = corpus
    replayer.envsetting = envsetting

    return replayer

# resume is False, max_run=100 (number of steps), replay is False, em_guide enables coverage guidance
def run_benchmark(agent_maker, env, seed, path, em_guide, replay=False):
    """
    benchmark_dir must be an instance of pathlib.Path
    """
    # logs execution?
    # summary_csv = benchmark_dir / "summary.csv"
    # creates empty csv files...
    # diagnostics_dir = benchmark_dir / "diagnostics"
    # diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # summary = list()
    # total = len(list(env.all_tasks))

    # if summary_csv.exists() and resume:
    #     summary = pd.read_csv(summary_csv)
    #     print('Loaded a summary of {} logs'.format(len(summary)))
    # else:
    #     summary = pd.DataFrame()

    # num_run = 0
    for weather, (start, target), run_name in env.all_tasks:
        # if (
        #     resume
        #     and len(summary) > 0
        #     and (
        #         (summary["start"] == start)
        #         & (summary["target"] == target)
        #         & (summary["weather"] == weather)
        #     ).any()
        # ):
        #     print('The suite task {} {} {} has been found in summary. Skipped.'.format(weather, start, target))
        #     # print(weather, start, target)
        #     continue

        # this is where empty .csv files are created
        # diagnostics_csv = str(diagnostics_dir / ("%s.csv" % run_name))
        print('run_benchmark(.) with:', weather, start, target)
        try:
            # how test is actually performed
            run_single(env, weather, start, target, agent_maker, seed, path, em_guide, replay)
            break
        except Exception as e:
            print("run_single(.) FAILED", weather, start, target)
            print("Exception", e)
            raise
            # print(e.stackTrace)