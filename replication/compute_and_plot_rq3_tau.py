import os

import numpy as np
from typing import Dict, List

from common import METHOD_LABELS, USE_CASE_LABELS_DICT, USE_CASES
from data_management import get_data_from_experiment, get_logs,\
    store_results, load_results, store_dict
from data_processing import compute_statistical_fault_discovery_results, compute_time_results
from plotting import create_bar_plots, get_colors, get_gamma_colors, get_method_colors, get_tau_colors,\
    plot_results, plot_k_g_analysis, adds_results_to_axs

'''
This file retrieves the data from the tau analysis and plots Figure 5.
'''

# computes and stores results or loads them
LOAD = False

if __name__ == '__main__':
    colors = get_method_colors()
    # variables to find data
    method_names = ['fuzzer', 'mdpfuzz', 'rt']
    use_case_keys = ['acas', 'bw', 'carla', 'cart', 'coop', 'll', 'tt']

    rq2_data_folder = 'data_rq2'
    rq3_data_folder = 'data_rq3'
    rq3_results_folder = 'results_rq3'
    tau_list = [0.01, 0.1, 1.0]
    tau_colors = get_tau_colors()
    k = 10
    gamma = 0.01
    tau_results = []
    labels = []
    for t in tau_list:
        results_path = f'{rq3_results_folder}/mdpfuzz_{k}_{t}_{gamma}'
        if not LOAD:
            d = {
                'k': k,
                'tau': t,
                'gamma': gamma
            }
            if t == 0.01:
                log_found = [get_logs(f'{rq2_data_folder}/{case}/', 'mdpfuzz') for case in use_case_keys]
            else:
                log_found = [get_logs(f'{rq3_data_folder}/', f'{case}_{k}_{t}_{gamma}_') for case in use_case_keys]

            logs = []
            use_case_found = []
            for l, c in zip(log_found, use_case_keys):
                if len(l) == 0:
                    print("No data found for {} in {}".format(t, c))
                else:
                    logs.append(l)
                    use_case_found.append(c)
                    if c not in labels:
                        labels.append(c)
            # checking data loading
            # assert len(logs) == 7
            # assert np.all([(len(l) == 5) or (len(l) == 3) for l in logs]), [len(l) for l in logs]
            fault_results = compute_statistical_fault_discovery_results(logs)
            d['name'] = 'mdpfuzz'
            results = store_results(use_case_found, fault_results, results_path, d)
        else:
            results = load_results(results_path)
            use_cases_found = list(results.keys())
            for u in use_cases_found:
                if (u in use_case_keys) and (u not in labels):
                    labels.append(u)
        tau_results.append(results)
    # plotting
    d = tau_results[0]
    results_to_plot = [d[u] for u in labels]
    fig, axs = plot_results(
        [USE_CASE_LABELS_DICT[k] for k in labels],
        results_to_plot,
        tau_colors[0],
        f'$\\tau={tau_list[0]}$',
        vertical=False
    )
    for i in [1, 2]:
        d = tau_results[i]
        results_to_plot = [d[u] for u in labels]
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
    print("Tau analysis done (see \"rq3_tau.png\").")
