import matplotlib.ticker as ticker

from common import FUZZER_COLOR, MDPFUZZ_COLOR, MDPFUZZ_SUFFIX, FUZZER_SUFFIX, ITERATION_SUFFIX, MDPFUZZ_LABEL, FUZZER_LABEL
from data_manager import get_fuzzing_logs, store_dict, load_dict,\
    store_results, load_results_of_fuzzers
from process_data import count_median_iterations_over_time, compute_statistical_fault_discovery_results
from plot import plot_reproduction_results, adds_results_to_axs,\
    find_time_by_which_a_method_preforms_as_many_tests_as_the_other_one, mark_timestep,\
    add_results_to_axis, as_thousands_notation


'''
This file aims at retrieving the data from the reproduction study,
computes the results and plotting the latter.
'''

# whether to compute the results
LOAD = False


if __name__ == '__main__':
    # variables for plotting
    use_cases = ['ACAS Xu', 'Bipedal Walker', 'CARLA', 'Coop Navi']
    # variables to find data
    data_folder = 'data'
    use_case_folders = ['acas', 'bw/no_change', 'carla', 'coop/no_fix']
    use_case_keys = ['acas', 'bw', 'carla', 'coop']
    results_folder = 'results'

    #################### 1st computation: original reproduction ###################
    results_path = f'{results_folder}/results'
    if not LOAD:

        # MDPFuzz
        mdpfuzz_logs = []
        mdpfuzz_keys = []
        for use_case_folder in use_case_folders:
            try:
                logs = get_fuzzing_logs(f'{data_folder}/{use_case_folder}/mdpfuzz/')
                if len(logs) != 0:
                    mdpfuzz_keys.append(use_case_folder)
                    mdpfuzz_logs.append(logs)
            except Exception as e:
                print("No MDPFuzz logs found for {}.".format(use_case_folder))

            # mdpfuzz_logs.append(get_fuzzing_logs(f'{data_folder}/{use_case_folder}/mdpfuzz/'))
        mdpfuzz_results = compute_statistical_fault_discovery_results(mdpfuzz_logs)
        mdpfuzz_iterations = count_median_iterations_over_time(mdpfuzz_logs)
        store_dict(
            results_path + MDPFUZZ_SUFFIX + ITERATION_SUFFIX,
            # {k: t for k, t in zip(use_case_keys, mdpfuzz_iterations)}
            {k: t for k, t in zip(mdpfuzz_keys, mdpfuzz_iterations)}
        )
        # stores the results
        # store_results(use_case_keys, mdpfuzz_results, results_path + MDPFUZZ_SUFFIX)
        store_results(mdpfuzz_keys, mdpfuzz_results, results_path + MDPFUZZ_SUFFIX)

        # Fuzzer
        fuzzer_logs = []
        fuzzer_keys = []
        for use_case_folder in use_case_folders:
            try:
                logs = get_fuzzing_logs(f'{data_folder}/{use_case_folder}/fuzzer/')
                if len(logs) != 0:
                    fuzzer_keys.append(use_case_folder)
                    fuzzer_logs.append(logs)
            except:
                print("No Fuzzer logs found for {}.".format(use_case_folder))
            # fuzzer_logs.append(get_fuzzing_logs(f'{data_folder}/{use_case_folder}/fuzzer/'))

        fuzzer_results = compute_statistical_fault_discovery_results(fuzzer_logs)
        fuzzer_iterations = count_median_iterations_over_time(fuzzer_logs)
        store_dict(
            results_path + FUZZER_SUFFIX + ITERATION_SUFFIX,
            # {k: t for k, t in zip(use_case_keys, fuzzer_iterations)}
            {k: t for k, t in zip(fuzzer_keys, fuzzer_iterations)}
        )
        # store_results(use_case_keys, fuzzer_results, results_path + FUZZER_SUFFIX)
        store_results(fuzzer_keys, fuzzer_results, results_path + FUZZER_SUFFIX)

    else:
        (mdpfuzz_results, fuzzer_results), (mdpfuzz_keys, fuzzer_keys) = load_results_of_fuzzers(results_path)
        print("MDPFuzz's results keys:", mdpfuzz_keys)
        print("Fuzzer's results keys:", fuzzer_keys)

        tmp = load_dict(results_path + MDPFUZZ_SUFFIX + ITERATION_SUFFIX)
        mdpfuzz_iterations = [tmp[k] for k in mdpfuzz_keys]
        # mdpfuzz_iterations = [tmp[k] for k in use_case_keys]

        tmp = load_dict(results_path + FUZZER_SUFFIX + ITERATION_SUFFIX)
        fuzzer_iterations = [tmp[k] for k in fuzzer_keys]
        # fuzzer_iterations = [tmp[k] for k in use_case_keys]

    if mdpfuzz_keys != fuzzer_keys:
        print("Inconsistent number of results for the fuzzers. Please run the 2 methods on the same use case(s).")
        exit(0)

    # plotting
    plot_cases = mdpfuzz_keys
    print("Plotting for use case key(s):", plot_cases)
    budgets_in_hours = [12 for _ in range(len(plot_cases))]
    # 24 hours of testing for CARLA
    if use_case_keys[2] in plot_cases:
        budgets_in_hours[plot_cases.index(use_case_keys[2])] = 24


    fig, axs = plot_reproduction_results([use_cases[use_case_keys.index(u.split("/")[0])] for u in plot_cases], mdpfuzz_results, MDPFUZZ_COLOR, MDPFUZZ_LABEL, budgets_in_hours)
    # adds the results of the fuzzer
    adds_results_to_axs(axs, fuzzer_results, FUZZER_COLOR, FUZZER_LABEL, budgets_in_hours)

    text_properties = {
        'x_offset': 0.5,
        'y_offset': -10.0,
        'fontsize': 12
    }
    marker_prop = {
        'markersize': 20,
        'markeredgewidth': 2,
        'color': FUZZER_COLOR
    }
    # indicates time after which Fuzzer performs MDPFuzz's number of iterations
    # done only if all computations have been performed
    if plot_cases != use_case_keys:
        # customized offsets to plot the time steps
        customized_y_offsets = [-50.0, -15.0, -2.0, 300]
        customized_x_offsets = [0.5, 0.4, -0.55, -0.5]
        for i, x, y in zip(range(len(fuzzer_results)), customized_x_offsets, customized_y_offsets):
            text_properties['x_offset'] = x
            text_properties['y_offset'] = y
            x, y = find_time_by_which_a_method_preforms_as_many_tests_as_the_other_one(
                fuzzer_results[i], fuzzer_iterations[i], mdpfuzz_iterations[i], budgets_in_hours[i]
            )
            if (x is not None) and (y is not None):
                mark_timestep(axs[i], x, y, marker_prop, text_properties)
        text_properties['y_offset'] = -100.0
        text_properties['x_offset'] = 0.25

    #################### 2st computation: threats to validity study ###################

    #### ACAS Xu ####
    # label = 'Suspected ' + FUZZER_LABEL
    # results_path = f'{results_folder}/suspected_fuzzer_acas'
    # if not LOAD:
    #     path = f'{data_folder}/acas/original_fuzzer/'
    #     suspected_fuzzer_acas = compute_statistical_fault_discovery_results([get_fuzzing_logs(path)])[0]
    #     store_dict(results_path, {'acas': suspected_fuzzer_acas})
    # else:
    #     suspected_fuzzer_acas = load_dict(results_path)['acas']
    # add_results_to_axis(axs[0], [suspected_fuzzer_acas], [label], [FUZZER_COLOR], 'dashed')

    for (key, subfolder, suffix, l) in [('bw', 'mutation_fixed', 'Mutation', 'dashed'), ('coop', 'sampling_fixed', 'Sampling', 'dashed'), ('coop', 'fixed', 'Fixed', 'dotted')]:
        try:
            fault_list, iter_list, labels = [], [], []
            for name in ['mdpfuzz', 'fuzzer']:
                # label = rf'$\textit{{{suffix}}}$'
                label = rf'${suffix}$'
                dict_path = f'{results_folder}/{key}_{subfolder}_{name}'
                if not LOAD:
                    path = f'{data_folder}/{key}/{subfolder}/'
                    logs = [get_fuzzing_logs(path + name + '/')]
                    tmp = compute_statistical_fault_discovery_results(logs)[0]
                    store_dict(dict_path, {key: tmp})

                    tmp2 = count_median_iterations_over_time(logs)[0]
                    store_dict(dict_path + ITERATION_SUFFIX, {key: tmp2})

                else:
                    tmp = load_dict(dict_path)[key]
                    tmp2 = load_dict(dict_path + ITERATION_SUFFIX)[key]
                fault_list.append(tmp)
                iter_list.append(tmp2)
                labels.append(label)
            # ax = axs[use_case_keys.index(key)]
            axis_index = None
            for j in range(len(plot_cases)):
                if plot_cases[j].startswith(key):
                    axis_index = j
                    break

            if axis_index is not None:
                ax = axs[j]
                add_results_to_axis(ax, fault_list, labels, [MDPFUZZ_COLOR, FUZZER_COLOR], l)

            if subfolder == 'fixed':
                x, y = find_time_by_which_a_method_preforms_as_many_tests_as_the_other_one(
                    fault_list[1], iter_list[1], iter_list[0]
                )
                mark_timestep(axs[-1], x, y, marker_prop, text_properties)
        except:
            print("No results found for {} in \"{}\"".format(name, dict_path))


    for ax in axs:
        legend = ax.legend(prop={'size': 11}, labelspacing=1.0, handletextpad=1.0, borderpad=0.55, borderaxespad=0.7)
        legend_frame = legend.get_frame()
        legend_frame.set_facecolor('0.9')
        legend_frame.set_edgecolor('0.9')

    for ax in axs:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(as_thousands_notation))


    filename = 'fault_discovery_plot.png'
    print("=============================")
    print("Results of the reproduction study done! (see \"{}\")".format(filename))
    print("=============================")
    fig.tight_layout()
    fig.savefig(filename)
