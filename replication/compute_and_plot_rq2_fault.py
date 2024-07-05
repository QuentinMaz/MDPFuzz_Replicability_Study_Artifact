import os

import numpy as np
from typing import Dict, List

from common import METHOD_LABELS, USE_CASES, USE_CASE_LABELS_DICT
from data_management import get_data_from_experiment, get_logs,\
    store_results, load_results, store_dict
from data_processing import compute_statistical_fault_discovery_results, compute_time_results
from plotting import create_bar_plots, get_colors, get_gamma_colors, get_method_colors, get_tau_colors,\
    plot_results, plot_k_g_analysis, adds_results_to_axs

'''
This file processes the data from data_rq2/, counts the number of faults found and plots the latter.
It should return a .png similar to Figure 2 of the paper.
'''

# computes and stores results or loads them
LOAD = False

if __name__ == '__main__':
    colors = get_method_colors()
    # variables to find data
    method_names = ['fuzzer', 'mdpfuzz', 'rt']
    use_case_keys = ['acas', 'bw', 'carla', 'cart', 'coop', 'll', 'tt']

    #################### 1st metric: Fault Discovery ###################
    rq2_data_folder = 'data_rq2'
    rq2_results_folder = 'results_rq2'

    if not os.path.isdir(rq2_results_folder):
        os.mkdir(rq2_results_folder)

    results_of_each_method = []
    labels = []
    for name in method_names:
        results_path = f'{rq2_results_folder}/{name}'
        if not LOAD:
            d = {
                'k': 10,
                'tau': 0.01,
                'gamma': 0.01
            }
            logs = []
            use_case_found = []
            for use_case_folder in ['acas', 'bw', 'cart', 'coop', 'll', 'tt']:
                try:
                    method_logs = get_logs(f'{rq2_data_folder}/{use_case_folder}/', name)
                    print("Found {} for method {} in {}".format(len(method_logs), name, use_case_folder))
                    logs.append(method_logs)
                    use_case_found.append(use_case_folder)
                    if use_case_folder not in labels:
                        labels.append(use_case_folder)
                except:
                    print("no data for {} {}".format(name, use_case_folder))
            fault_results = compute_statistical_fault_discovery_results(logs)
            d['name'] = name
            results = store_results(use_case_found, fault_results, results_path, d)
            print("Stored {} results for {} (cases: {}).".format(len(fault_results), name, use_case_found))
        else:
            results = load_results(results_path)
            use_cases_found = list(results.keys())
            for u in use_cases_found:
                if (u in use_case_keys) and (u not in labels):
                    labels.append(u)
        results_of_each_method.append(results)
    # plotting
    d = results_of_each_method[0]
    results_to_plot = [d[u] for u in labels]
    # labels = [USE_CASE_LABELS_DICT[k] for k in labels]
    print(labels)
    fig, axs = plot_results(
        [USE_CASE_LABELS_DICT[k] for k in labels],
        results_to_plot,
        colors[0],
        METHOD_LABELS[0],
        vertical=False
    )

    for i in [1, 2]:
        d = results_of_each_method[i]
        results_to_plot = [d[u] for u in labels]
        adds_results_to_axs(axs, results_to_plot, colors[i], METHOD_LABELS[i])

    # for ax in axs:
    if len(axs) > 4:
        ax = axs[3]
    else:
        ax = axs[0]
    legend = ax.legend(prop={'size': 18}, labelspacing=1.0, handletextpad=1.0, borderpad=0.55, borderaxespad=0.7)
    legend_frame = legend.get_frame()
    legend_frame.set_facecolor('0.9')
    legend_frame.set_edgecolor('0.9')
    for line in legend.get_lines():
        line.set_linewidth(4.0)
    filename = 'rq2_fault.png'
    fig.tight_layout()
    fig.savefig(filename)
    print("Fault discovery analysis done (see \"rq2_fault.png\").")

