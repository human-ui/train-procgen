import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from math import ceil
from constants import ENV_NAMES

import seaborn  # sets some style parameters automatically

COLORS = [(57, 106, 177), (218, 124, 48)]


def switch_to_outer_plot(fig):
    ax0 = fig.add_subplot(111, frame_on=False)
    ax0.set_xticks([])
    ax0.set_yticks([])

    return ax0


def ema(data_in, smoothing=0):
    data_out = np.zeros_like(data_in)
    curr = np.nan

    for i in range(len(data_in)):
        x = data_in[i]
        if np.isnan(curr):
            curr = x
        else:
            curr = (1 - smoothing) * x + smoothing * curr

        data_out[i] = curr

    return data_out


def plot_data_mean_std(ax, data_y, color, data_x=None, x_scale=1, smoothing=0, first_valid=0, label=None):
    # color = COLORS[color_idx]
    # hexcolor = '#%02x%02x%02x' % color

    data_y = data_y[:, first_valid:]
    nx, num_datapoint = np.shape(data_y)

    if smoothing > 0:
        for i in range(nx):
            data_y[i, ...] = ema(data_y[i, ...], smoothing)

    if data_x is None:
        data_x = (np.array(range(num_datapoint)) + first_valid) * x_scale
    else:
        data_x = data_x[first_valid:]

    data_mean = np.mean(data_y, axis=0)
    ax.plot(data_x, data_mean, color=color, label=label, linestyle='solid', alpha=1, rasterized=True)

    if data_y.shape[0] > 1:
        data_std = np.std(data_y, axis=0, ddof=1)
        ax.fill_between(data_x, data_mean - data_std, data_mean + data_std, color=color, alpha=.25, linewidth=0.0, rasterized=True)


def read_csv(filename, key_name):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        key_index = -1

        values = []

        for line_num, row in enumerate(csv_reader):
            row = [x.lower() for x in row]
            if line_num == 0:
                idxs = [i for i, val in enumerate(row) if val == key_name]
                key_index = idxs[0]
            else:
                values.append(row[key_index])

    return np.array(values, dtype=np.float32)


def plot_values(ax, all_values, title=None, max_x=0, label=None, data_x=None, is_normalized=False, **kwargs):
    if max_x > 0:
        all_values = all_values[..., :max_x]
        if data_x is not None:
            data_x = data_x[:max_x]

    if ax is not None:
        plot_data_mean_std(ax, all_values, label=label, data_x=data_x, **kwargs)
        ax.set_title(title)
        ax.set_xlim((0, 8e6))
        if is_normalized:
            ax.set_ylim((-0.2, 1))

    return all_values


def plot_experiment(axarr,
                    color='#000000',
                    label='',
                    run_directory_prefix='',
                    titles=None,
                    suffixes=[''],
                    key_name='eprewmean',
                    x_key_name=None,
                    env_names=None,
                    num_runs=3,
                    will_reduce=False,
                    normalization_ranges=None,
                    **kwargs):
    run_folders = [f'{run_directory_prefix}{x}' for x in range(num_runs)]

    env_names = env_names or ENV_NAMES
    num_envs = len(env_names)
    is_normalized = normalization_ranges is not None

    all_values = []

    for env_idx in range(num_envs):
        env_name = env_names[env_idx]

        # print(f'loading results from {env_name}...')
        csv_files = [f"results/{resid}/progress-{env_name}.csv" for resid in run_folders]

        if not will_reduce:
            dimy = len(axarr[0])
            curr_ax = axarr[env_idx // dimy][env_idx % dimy]
        else:
            curr_ax = None

        raw_data = np.array([read_csv(file, key_name) for file in csv_files])

        raw_data_x = None
        if x_key_name:
            raw_data_x = read_csv(csv_files[0], x_key_name)

        if is_normalized:
            game_range = normalization_ranges[env_name]
            game_min = game_range[0]
            game_max = game_range[1]
            raw_data = (np.array(raw_data) - game_min) / (game_max - game_min)

        values = plot_values(curr_ax, raw_data,
                             title=env_name,
                             color=color,
                             label=label if env_idx == 0 else None,  # only label the first graph to avoid legend duplicates,
                             data_x=raw_data_x,
                             is_normalized=is_normalized,
                             **kwargs)
        all_values.append(values)

    if will_reduce:
        normalized_data = np.mean(all_values, axis=0)
        title = 'Mean Normalized Score'
        plot_values(axarr, normalized_data, title=None, color=color, label=label, **kwargs)


def plot_experiments(will_reduce, env_names, args_list):

    env_names = env_names or ENV_NAMES
    num_envs = len(env_names)

    if will_reduce:
        num_visible_plots = 1
        f, axarr = plt.subplots()
    else:
        num_visible_plots = num_envs
        dimx = dimy = ceil(np.sqrt(num_visible_plots))
        f, axarr = plt.subplots(dimx, dimy, sharex=True)

    for idx, args in enumerate(args_list):
        plot_experiment(axarr, env_names=env_names, will_reduce=will_reduce, **args)

    if len(args_list) > 1:
        if num_visible_plots == 1:
            axarr.legend(loc='lower right')
        else:
            f.legend(loc='lower right', bbox_to_anchor=(.5, 0, .5, 1))

    return f, axarr
