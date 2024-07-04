import os

import numpy as np
from typing import Dict, List

from common import METHOD_LABELS, USE_CASES
from data_management import get_data_from_experiment, get_logs,\
    store_results, load_results, store_dict
from data_processing import compute_statistical_fault_discovery_results, compute_time_results
from plotting import create_bar_plots, get_colors, get_gamma_colors, get_method_colors, get_tau_colors,\
    plot_results, plot_k_g_analysis, adds_results_to_axs

'''
This file retrieves the data from the replication study,
computes the results and plots the latter.
'''

# computes and stores results or loads them
LOAD = False
# plots fault discovery of the three methods
RQ2 = True
# exports execution time analysis of the three methods as a Latex table
TIME_METRIC = True
# plots fault discovery of MDPFuzz for K=10, \gamma=0.01 and \tau in [0.01, 0.1, 1.0]
RQ3_TAU = True
# plots fault discovery of MDPFuzz for \tau=0.01, K in [6, 8, 10, 12, 14] and \gamma in [0.05, 0.1, 0.15, 0.2]
RQ3 = True

if __name__ == '__main__':
    colors = get_method_colors()
    # variables to find data
    method_names = ['fuzzer', 'mdpfuzz', 'rt']
    use_case_keys = ['acas', 'bw', 'carla', 'cart', 'coop', 'll', 'tt']

    #################### 1st metric: Fault Discovery ###################
    rq2_data_folder = 'data_rq2'
    rq2_results_folder = 'results_rq2'
    if RQ2:
        if not os.path.isdir(rq2_results_folder):
            os.mkdir(rq2_results_folder)

        results_of_each_method = []
        for name in method_names:
            results_path = f'{rq2_results_folder}/{name}'
            if not LOAD:
                d = {
                    'k': 10,
                    'tau': 0.01,
                    'gamma': 0.01
                }
                logs = []
                for use_case_folder in use_case_keys:
                    method_logs = get_logs(f'{rq2_data_folder}/{use_case_folder}/', name)
                    if use_case_folder == 'carla':
                        assert len(method_logs) == 3
                    else:
                        assert len(method_logs) == 5, f'{use_case_folder}: {name}'
                    logs.append(method_logs)
                fault_results = compute_statistical_fault_discovery_results(logs)
                d['name'] = name
                results = store_results(use_case_keys, fault_results, results_path, d)
            else:
                results = load_results(results_path)
            results_of_each_method.append(results)
        # plotting
        d = results_of_each_method[0]
        results_to_plot = [d[u] for u in use_case_keys]
        fig, axs = plot_results(USE_CASES, results_to_plot, colors[0], METHOD_LABELS[0], vertical=False)
        for i in [1, 2]:
            d = results_of_each_method[i]
            results_to_plot = [d[u] for u in use_case_keys]
            adds_results_to_axs(axs, results_to_plot, colors[i], METHOD_LABELS[i])

        # for ax in axs:
        ax = axs[3]
        legend = ax.legend(prop={'size': 18}, labelspacing=1.0, handletextpad=1.0, borderpad=0.55, borderaxespad=0.7)
        legend_frame = legend.get_frame()
        legend_frame.set_facecolor('0.9')
        legend_frame.set_edgecolor('0.9')
        for line in legend.get_lines():
            line.set_linewidth(4.0)
        filename = 'rq2.png'
        fig.tight_layout()
        fig.savefig(filename)
    else:
        print('RQ2 Skipped.')

    ########## 2nd metric: Time ##########
    if TIME_METRIC:
        method_results_list: List[Dict] = []
        for name in method_names:
            results_path = f'{rq2_results_folder}/time_{name}'
            if not LOAD:
                dict_list = [
                    get_data_from_experiment(
                        f'{rq2_data_folder}/{case}/',
                        config_pattern=f'{name}.*_config.json')
                        for case in use_case_keys
                    ]
                results = compute_time_results(dict_list)
                store_dict(results_path, results)
            else:
                results = load_results(results_path)
            method_results_list.append(results)

        plot_data = {}
        print('checking time results\' internal structure...')
        # re-organizes the results per use-case
        # checks results' internal structure
        keys_list = [list(d.keys()) for d in method_results_list]
        [l.sort() for l in keys_list]
        assert np.all([l == USE_CASES for l in keys_list]), [l for l in keys_list]

        time_keys = ['run', 'test', 'cov']
        for d in method_results_list:
            # time results of all the use-cases as dictionaries
            time_results: List[Dict] = list(d.values())
            # checks that all those dicitonaries have the correct keys
            assert np.all([list(tmp.keys()) == time_keys for tmp in time_results])
        print('checking done.')
        for r in USE_CASES:
            data = []
            numerical_data = []
            for i in range(len(method_results_list)):
                tmp = []
                numerical_tmp = []
                times = method_results_list[i][r]
                for (m, e) in times.values():
                    m = m / 60
                    e = e / (2 * 60)
                    numerical_tmp.append((m, 2 * e))
                    if m != 0:
                        tmp.append((f'{m:.2f}', f'{e:.1f}'))
                data.append(tuple(tmp))
                numerical_data.append(tuple(numerical_tmp))
            plot_data[r] = numerical_data
        time_colors = { k: v for k, v in zip(['Total', '$\\pi$-Env.', 'Cov.'],  get_colors(10, 'tab10')[:3]) }
        fig, axs = create_bar_plots(plot_data, METHOD_LABELS, time_colors)
        fig.tight_layout()
        fig.savefig('rq1_time.png')
    else:
        print('Time analysis skipped.')


    # RQ3 "pre-study" that shows that tau is not important
    rq3_data_folder = 'data_tau'
    rq3_results_folder = 'results_rq3'
    if RQ3_TAU:
        tau_list = [0.01, 0.1, 1.0]
        tau_colors = get_tau_colors()
        k = 10
        gamma = 0.01
        tau_results = []
        for t in tau_list:
            results_path = f'{rq3_results_folder}/mdpfuzz_{k}_{t}_{gamma}'
            if not LOAD:
                d = {
                    'k': k,
                    'tau': t,
                    'gamma': gamma
                }
                if t == 0.01:
                    logs = [get_logs(f'{rq3_data_folder}/{case}/', 'mdpfuzz') for case in use_case_keys]
                else:
                    logs = [get_logs(f'{rq3_data_folder}/', f'{case}_{k}_{t}_{gamma}_') for case in use_case_keys]

                # checking data loading
                assert len(logs) == 7
                assert np.all([(len(l) == 5) or (len(l) == 3) for l in logs]), [len(l) for l in logs]

                fault_results = compute_statistical_fault_discovery_results(logs)
                d['name'] = 'mdpfuzz'
                results = store_results(use_case_keys, fault_results, results_path, d)
            else:
                results = load_results(results_path)
            tau_results.append(results)
        # plotting
        d = tau_results[0]
        results_to_plot = [d[u] for u in use_case_keys]
        fig, axs = plot_results(USE_CASES, results_to_plot, tau_colors[0], f'$\\tau={tau_list[0]}$', vertical=False)
        for i in [1, 2]:
            d = tau_results[i]
            results_to_plot = [d[u] for u in use_case_keys]
            adds_results_to_axs(axs, results_to_plot, tau_colors[i], f'$\\tau={tau_list[i]}$')

        ax = axs[3]
        legend = ax.legend(prop={'size': 18}, labelspacing=1.0, handletextpad=1.0, borderpad=0.55, borderaxespad=0.7)
        legend_frame = legend.get_frame()
        legend_frame.set_facecolor('0.9')
        legend_frame.set_edgecolor('0.9')
        for line in legend.get_lines():
            line.set_linewidth(4.0)

        filename = 'rq3_tau.png'
        fig.tight_layout()
        fig.savefig(filename)
    else:
        print('Tau analysis skipped.')

    # RQ3: parameter analysis: how do K and gamma impact MDPFuzz's performance?
    K = [6, 8, 10, 12, 14]
    G = [0.05, 0.1, 0.15, 0.2]
    TAU = 0.01
    if RQ2:
        if not os.path.isdir(rq3_results_folder):
            os.mkdir(rq3_results_folder)

        rq3_results = []
        for k in K:
            for g in G:
                results_path = f'{rq3_results_folder}/mdpfuzz_{k}_{TAU}_{g}'
                d = {
                    'k': k,
                    'tau': TAU,
                    'gamma': g,
                    'name': 'mdpfuzz'
                }
                if not LOAD:
                    # list of logs per use-case
                    logs = []
                    for key in use_case_keys:
                        prefix = f'{key}_{k}_{TAU}_{g}_'
                        config_logs = get_logs(rq3_data_folder, prefix)
                        if key == 'carla':
                            assert len(config_logs) == 3, len(config_logs)
                        else:
                            assert len(config_logs) == 5, f'{prefix}: {len(config_logs)}'
                        logs.append(config_logs)

                    assert len(logs) == 7, f'{k}_{TAU}_{g}_'
                    fault_results = compute_statistical_fault_discovery_results(logs)
                    results = store_results(use_case_keys, fault_results, results_path, d)
                else:
                    results = load_results(results_path)
                rq3_results.append(results)
        gamma_dict = {g: c for g, c in zip(G, get_gamma_colors())}
        fig, axs = plot_k_g_analysis(
            use_case_keys,
            rq3_results,
            gamma_dict,
            y_axis='k',
            use_case_labels=USE_CASES,
            filename='rq3.png'
        )
    else:
        print('RQ3 Skipped.')
