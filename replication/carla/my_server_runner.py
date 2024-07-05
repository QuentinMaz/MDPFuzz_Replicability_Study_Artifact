import os
import signal
import subprocess
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import argparse
import time
import torch
import numpy as np

import bird_view.utils.my_carla_utils as cu
from benchmark import make_suite, get_suites, ALL_SUITES

from carla_executor import CarlaExecutor
sys.path.append('methods/src/')
from mdpfuzz import Fuzzer

'''
Script which shares the code of "my_runner.py" while taking care of launching its own server.
The latter is launched in a second process, which is killed after the execution.
In case of unexpected error, the server process writes its PID in a file "carla_server_pid_{PORT}.txt" when it starts.
'''


def test(args):
    port = args.port
    suite = args.suite
    env_seed = args.seed
    method_seed = args.method_seed
    init_budget = args.method_init
    test_budget = args.method_testing
    results_path = args.path
    index = args.method_index

    k = args.k
    tau = args.tau
    gamma = args.gamma

    suite_name = get_suites(suite)[0]

    print('--------- CONFIG --------')
    print('env and method seeds:', env_seed, method_seed)
    print('budgets:', init_budget, test_budget)
    print('K, tau and gamma:', k, tau, gamma)
    print('method\'s index:', index)
    print('--------------------------')

    with make_suite(suite_name, port=port, crop_sky=args.crop_sky) as env:
        t0 = time.time()

        weather, (start, target), run_name = list(env.all_tasks)[0]

        env.seed = env_seed
        env.init(start=start, target=target, weather=cu.PRESET_WEATHERS[weather])

        executor = CarlaExecutor(100, env)
        executor.carla_env = env

        print('Loading policy...')
        agent = executor.load_policy(args)
        fuzzer = Fuzzer(method_seed, k, tau, gamma, executor)
        print('...Policy loaded. Switching log file and start testing.')

        # assert os.path.isdir(path), path
        if os.path.isdir(results_path) and (not results_path.endswith('/')):
            results_path += '/'

        if index == 1:
            method_name = 'fuzzer'
        elif index == 2:
            method_name = 'mdpfuzz'
        else:
            method_name = 'rt'

        if not 'data_rq3' in results_path:
            results_path += '{}_{}_{}_{}_{}'.format(method_name, k, tau, gamma, method_seed)
        else:
            results_path += '{}_{}_{}_{}'.format(k, tau, gamma, method_seed)

        f = open(results_path + '.txt', 'w', buffering=1)
        sys.stdout = f

        print('path:', results_path)
        print('method seed:', method_seed)
        print('method budget', init_budget, test_budget)

        print('CUDA INFO:', args.device)
        if args.device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        else:
            print('cpu')

        if index == 0:
            fuzzer.random_testing(
                n=test_budget,
                policy=agent,
                path=results_path,
                exp_name='CARLA'
            )
        elif index == 1:
            fuzzer.fuzzing_no_coverage(
                n=init_budget,
                policy=agent,
                test_budget=test_budget,
                saving_path=results_path,
                save_logs_only=True, # won't save the pool
                light_pool=True,
                local_sensitivity=True,
                exp_name='CARLA'
            )
        elif index == 2:
            fuzzer.fuzzing(
                n=init_budget,
                policy=agent,
                test_budget=test_budget,
                saving_path=results_path,
                save_logs_only=True, # won't save the pool
                light_pool=True,
                local_sensitivity=True,
                exp_name='CARLA'
            )

        total_time = time.time() - t0

    print('Testing done. Total time of {} min.'.format(total_time / 60))
    f.close()
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    torch.set_num_threads(1)
    import warnings
    warnings.filterwarnings('ignore')

    # suite_name = 'town2'
    # sim_steps = 100
    # nodel_path = 'model_RL_IAs_only_town01_train_weather/'
    # crop_sky = True
    # port=1453
    # disable_cuda, disable_cudnn = True, True

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path-folder-model',
        required=True,
        type=str,
        help='Folder containing all models, ie the supervised Resnet18 and the RL models',
    )
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--suite', choices=ALL_SUITES, default='town1')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--max-run', type=int, default=100)
    parser.add_argument('--emguide', action='store_true')
    parser.add_argument('--plotting', action='store_true')
    parser.add_argument('--path_videos', type=str, default='./')

    parser.add_argument(
        '--nb_action_steering',
        type=int,
        default=27,
        help='How much different steering values in the action (should be odd)',
    )
    parser.add_argument(
        '--max_steering', type=float, default=0.6, help='Max steering value possible in action'
    )
    parser.add_argument(
        '--nb_action_throttle',
        type=int,
        default=3,
        help='How much different throttle values in the action',
    )
    parser.add_argument(
        '--max_throttle', type=float, default=1, help='Max throttle value possible in action'
    )

    parser.add_argument('--front-camera-width', type=int, default=288)
    parser.add_argument('--front-camera-height', type=int, default=288)
    parser.add_argument('--front-camera-fov', type=int, default=100)
    parser.add_argument(
        '--crop-sky',
        action='store_true',
        default=False,
        help='if using CARLA challenge model, let sky, we cropped '
        'it for the models trained only on Town01/train weather',
    )

    parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
    parser.add_argument('--disable-cuda', action='store_true', help='disable cuda')
    parser.add_argument('--disable-cudnn', action='store_true', help='disable cuDNN')

    parser.add_argument('--kappa', default=1.0, type=float, help='kappa for Huber Loss in IQN')
    parser.add_argument(
        '--num-tau-samples', default=8, type=int, help='N in equation 3 in IQN paper'
    )
    parser.add_argument(
        '--num-tau-prime-samples', default=8, type=int, help='N in equation 3 in IQN paper'
    )
    parser.add_argument(
        '--num-quantile-samples', default=32, type=int, help='K in equation 3 in IQN paper'
    )
    parser.add_argument(
        '--quantile-embedding-dim', default=64, type=int, help='n in equation 4 in IQN paper'
    )

    parser.add_argument('--path', default='../../../data_rq2/carla/', type=str)
    parser.add_argument('--method-seed', default=2021, type=int)
    parser.add_argument('--method-index', default=0, type=int)
    parser.add_argument('--method-init', default=1000, type=int)
    parser.add_argument('--method-testing', default=5000, type=int)

    parser.add_argument('--k', default=10, type=int)
    parser.add_argument('--tau', default=0.01, type=float)
    parser.add_argument('--gamma', default=0.01, type=float)


    args = parser.parse_args()
    args.steps_image = [
        -10,
        -2,
        -1,
        0,
    ]

    stdout_file = 'client_pid_{}.txt'.format(os.getpid())
    print('stdout redirected to', stdout_file)
    f = open(stdout_file, 'w', buffering=1)
    sys.stdout = f

    if torch.cuda.is_available() and not args.disable_cuda:
        print('Find cuda!')
        args.device = torch.device('cuda:0')
        torch.cuda.manual_seed(2021)
        torch.backends.cudnn.enabled = not args.disable_cudnn
        # print('device selected is', os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    else:
        print('Will use CPU...')
        args.device = torch.device('cpu')
        rng = torch.manual_seed(0)


    args.path_folder_model = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), args.path_folder_model
    )

    index = args.method_index


    if index not in [0, 1, 2]:
        print('Wrong method index.')
    else:
        # launches a server
        try:
            pid_file = 'carla_server_pid_{}.txt'.format(args.port)
            process = subprocess.Popen(
                ['cd carla_RL_IAs && DISPLAY= ./CarlaUE4_pid.sh ../{} -fps=10 -benchmark -carla-port={} -opengl'.format(pid_file, args.port)],
                stdout=subprocess.PIPE, universal_newlines=True, shell=True
            )
            server_pid = process.pid
            print('Server launched on port {}.'.format(args.port))
            sleep_time = 120
            print('Sleeping {}s to let it time to initialize...'.format(sleep_time))
            time.sleep(sleep_time)
            print('Device passed to test(.):', args.device)
            test(args)
        except subprocess.CalledProcessError as e:
            print('Launching server on port {} failed.'.format(args.port))
            print('Command: "{}"; exit status: {}'.format(e.cmd, e.returncode))
            print('Output:', e.output.decode())
        except Exception as e:
            print('An error occured (unrelated to launching the server on port {}). Error:'.format(args.port))
            print(e)
        finally:
            if server_pid:
                print('Trying to kill the python subprocess (PID: {})...'.format(server_pid))
                try:
                    os.kill(server_pid, signal.SIGTERM)
                except ProcessLookupError:
                    print('Subprocess not found. Already finished?')
                print('Server process successfully killed.')
                print('Reading the real process to kill in {} ...'.format(pid_file))
                try:
                    with open('../../{}'.format(pid_file), 'r') as f:
                        pid = int(f.read().strip())
                    print('Got PID {}.'.format(pid))
                    try:
                        os.kill(pid, signal.SIGTERM)
                        print('Server successfully killed!')
                    except ProcessLookupError:
                        print('Something went wrong when (finally...) trying to kill sub-sub-process.')
                except FileNotFoundError as e:
                    print('An error occured when trying to read file "{}"'.format(pid_file))
                    print(e)
                    print('current path:', os.getcwd())
                except ValueError as e:
                    print('An error occured when casting the PID.')
                    print('Error:')
                    print(e)


    f.close()
    sys.stdout = sys.__stdout__