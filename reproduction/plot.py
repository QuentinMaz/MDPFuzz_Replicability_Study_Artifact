import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Colormap

from typing import List, Tuple, Dict, Union

from common import Y_LABEL, X_LABEL, TITLE_LABEL_FONTSIZE, AXIS_LABEL_FONTSIZE


# plt.rc('text', usetex=True)


def get_colors(n: int = 2, cmap: Union[str, Colormap] = plt.cm.jet):
    '''Return a list of @n tuples (RGBA colors) by sampling evenly in @cmap (default to "jet").'''
    if isinstance(cmap, str):
        cmap: Colormap = plt.cm.get_cmap(cmap)
    return [cmap(i) for i in np.linspace(0, 1, n)]


##### helpers for x labeling #####

def scientific_notation(x, pos):
    if x == 0:
        return '0'
    elif x < 1e3:
        return '{:.0f}'.format(x)
    else:
        exp = int(np.log10(x))
        coeff = x / 10**exp
        return r'${:.0f} \times 10^{{{}}}$'.format(coeff, exp)


def as_thousands_notation(x, pos):
    if x < 1000:
        return '{:.0f}'.format(x)
    else:
        return '{:.0f}K'.format(x / 1000)


##### plotting utilities #####


def plot_reproduction_results(
        use_cases: List[str],
        results: List[Tuple],
        color: Tuple,
        label: str,
        budgets_in_hours: Union[int, List[int]] = 12
    ):
    '''
    Plots the results of a method for each use-case.
    A chart per use-case; presented horizontally.
    '''
    n = len(use_cases)
    fig, axs = plt.subplots(ncols=n, figsize=(5 * n, 6))

    if n == 1:
        axs = np.array([axs])

    if isinstance(budgets_in_hours, int):
        budget = budgets_in_hours
        budgets_in_hours = [budget for _ in range(n)]
    else:
        assert len(budgets_in_hours) == n

    [ax.grid(axis='y', color='0.9', linestyle='-', linewidth=1) for ax in axs.flat]

    # y labeling
    axs[0].set_ylabel(Y_LABEL, fontsize=AXIS_LABEL_FONTSIZE - 1)

    # iterates over the use-cases
    for u in range(n):
        data = results[u]
        budget_in_hours = budgets_in_hours[u]
        # labeling
        axs[u].set_title(use_cases[u], fontsize=TITLE_LABEL_FONTSIZE)#, loc='left')
        axs[u].set_xlabel(X_LABEL, fontsize=AXIS_LABEL_FONTSIZE)
        y, perc_25, perc_75 = data
        # over time
        x = np.linspace(0, budget_in_hours, num=len(y), endpoint=True)
        # ticks = np.arange(1, budget_in_hours + 1)

        axs[u].plot(x, y, color=color, label=label)
        axs[u].fill_between(x, perc_25, perc_75, alpha=0.25, linewidth=0, color=color)
        # axs[u].set_xticks(ticks)
        # axs[u].set_xticklabels(ticks)

    return (fig, axs)


def adds_results_to_axs(
        axs: np.ndarray,
        results: List[Tuple],
        color: Tuple,
        label: str,
        budgets_in_hours: Union[int, List[int]] = 12
    ):
    '''
    Adds results to axes.
    It assumes the axes of shape (1, len(results)).
    '''
    ncols = len(axs)
    assert len(results) == ncols

    if isinstance(budgets_in_hours, int):
        budget = budgets_in_hours
        budgets_in_hours = [budget for _ in range(ncols)]
    else:
        assert len(budgets_in_hours) == ncols

    for r in range(ncols):
        data = results[r]
        budget_in_hours = budgets_in_hours[r]
        y, perc_25, perc_75 = data
        x = np.linspace(0, budget_in_hours, num=len(y), endpoint=True)
        axs[r].plot(x, y, color=color, label=label)
        axs[r].fill_between(x, perc_25, perc_75, alpha=0.25, linewidth=0, color=color)
    return axs


def add_results_to_axis(
        ax,
        results: List[Tuple],
        labels: List[str],
        colors: List[Tuple],
        linestyle: str,
        budget_in_hours: int = 12
    ):
    '''Plots all the results to the axis.'''
    for i in range(len(results)):
        data = results[i]
        color = colors[i]
        label = labels[i]
        y, perc_25, perc_75 = data
        x = np.linspace(0, budget_in_hours, num=len(y), endpoint=True)
        ax.plot(x, y, color=color, label=label, linestyle=linestyle)
        ax.fill_between(x, perc_25, perc_75, alpha=0.25, linewidth=0, color=color)
    return ax


def find_time_by_which_a_method_preforms_as_many_tests_as_the_other_one(
        fuzzer_result: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        fuzzer_iteration: np.ndarray,
        mdpfuzz_iteration: np.ndarray,
        budget_in_hours: int = 12

    ):
    # print(f'len of fuzzer iteration {len(fuzzer_iteration)}; of the results {len(fuzzer_result[0])}')
    # print(f'len of mdpfuzz iteration: {len(mdpfuzz_iteration)}')
    # print(f'iterations performed by fuzzer: {fuzzer_iteration[-1]}; by mdpfuzz: {mdpfuzz_iteration[-1]}.')

    x, y = None, None

    time_ticks = np.linspace(0, budget_in_hours, num=len(fuzzer_result[0]), endpoint=True)
    # number of executions of MDPFuzz
    max_num_exec = mdpfuzz_iteration[-1]
    # number of executions of Fuzzer
    if max_num_exec > fuzzer_iteration[-1]:
        print('MDPFuzz performs more executions')
    else:
        # minus 1?
        index = sum(fuzzer_iteration < max_num_exec)
        x = time_ticks[index]
        y = fuzzer_result[0][index]
    return x, y


def mark_timestep(ax, x: float, y: float, marker_prop: Dict = {}, text_prop: Dict = {}):
    x_offset = text_prop.pop('x_offset', 0.0)
    y_offset = text_prop.pop('y_offset', 0.0)
    ax.plot(x, y, '|', **marker_prop)
    ax.text(x + x_offset, y + y_offset, f'{x:0.2f} h', **text_prop)



if __name__ == '__main__':
    from data_manager import load_dict
    fps = [f'results2/results_{i}.json' for i in ['fuzzer', 'fuzzer_iterations', 'mdpfuzz', 'mdpfuzz_iterations']]
    dicts = [load_dict(f) for f in fps]
    key = 'acas'

    fr = dicts[0][key]
    fi = dicts[1][key]
    mi = dicts[3][key]
    x, y = find_time_by_which_a_method_preforms_as_many_tests_as_the_other_one(
        fr, fi, mi
    )
    print(x, y)

# def plot_reproduction_results(
#         use_cases: List[str],
#         results: List[Tuple],
#         color: Tuple,
#         label: str,
#         budget_in_hours: int = 12
#     ):
#     '''
#     Plots the results of a method for each use-case.
#     The first column plots the results over time, while the second column plots the same results over test executions.
#     '''
#     n = len(use_cases)
#     fig, axs = plt.subplots(nrows=n, ncols=2, figsize=(12, 6 * n), gridspec_kw={'width_ratios': [3, 4]}, sharey='row')

#     [ax.grid(axis='y', color='0.9', linestyle='-', linewidth=1) for ax in axs.flat]

#     # x labeling
#     ticks = np.arange(1, budget_in_hours + 1)
#     axs[-1][1].set_xlabel('#Iterations', fontsize=AXIS_LABEL_FONTSIZE)

#     # iterates over the use-cases
#     for u in range(n):
#         data = results[u]
#         # labeling
#         axs[u][0].set_title(use_cases[u], fontsize=TITLE_LABEL_FONTSIZE, loc='left')
#         axs[u][0].set_ylabel(FAULT_LABEL, fontsize=AXIS_LABEL_FONTSIZE - 1)
#         y, perc_25, perc_75 = data
#         # over time
#         x = np.linspace(0, budget_in_hours, num=len(y), endpoint=True)
#         axs[u][0].plot(x, y, color=color, label=label)
#         axs[u][0].fill_between(x, perc_25, perc_75, alpha=0.25, linewidth=0, color=color)
#         axs[u][0].set_xticks(ticks)
#         axs[u][0].set_xticklabels([])
#         # over test cases
#         x = np.arange(len(y))
#         axs[u][1].plot(x, y, color=color, label=label)
#         axs[u][1].fill_between(x, perc_25, perc_75, alpha=0.25, linewidth=0, color=color)
#     # x labeling
#     axs[-1][0].set_xlabel('Time (h)', fontsize=AXIS_LABEL_FONTSIZE)
#     axs[-1][0].set_xticklabels(ticks)

#     return (fig, axs)


# def adds_results_to_axs(
#         axs: np.ndarray,
#         results: List[Tuple],
#         color: Tuple,
#         label: str,
#         budget_in_hours: int = 12
#     ):
#     '''
#     Adds results to axes.
#     It assumes the axes of shape (num_use_cases, 2).
#     '''
#     nrows, ncols = axs.shape
#     assert ncols == 2, f'axs malformed! ({axs.shape})'
#     for r in range(nrows):
#         data = results[r]
#         y, perc_25, perc_75 = data
#         for j, x in enumerate([np.linspace(0, budget_in_hours, num=len(y), endpoint=True), np.arange(len(y))]):
#             axs[r][j].plot(x, y, color=color, label=label)
#             axs[r][j].fill_between(x, perc_25, perc_75, alpha=0.25, linewidth=0, color=color)
#     return axs


# def add_results_for_case(
#         row_axs: np.ndarray,
#         results: List[Tuple],
#         labels: List[str],
#         colors: List[Tuple],
#         linestyle: str,
#         budget_in_hours: int = 12
#     ):
#     for i in range(len(results)):
#         data = results[i]
#         color = colors[i]
#         label = labels[i]
#         y, perc_25, perc_75 = data
#         for j, x in enumerate([np.linspace(0, budget_in_hours, num=len(y), endpoint=True), np.arange(len(y))]):
#             row_axs[j].plot(x, y, color=color, label=label, linestyle=linestyle)
#             row_axs[j].fill_between(x, perc_25, perc_75, alpha=0.25, linewidth=0, color=color)
#     return axs