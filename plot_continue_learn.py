import pandas as pd
import json

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.stats as stats

import os


class ReinforceMetricV3():
    def __init__(self, evaluate_time):
        self.evaluate_time = evaluate_time
        self.EGO_SIZE = [1.8, 4.9, 1.8]

    def compute_metrics(self, load_path, mode='takeover_eval'):

        print("[EVAL] reinforcement_learning_metric start...")

        if mode == 'expert_eval':
            with open(load_path) as f:
                data_dict = json.load(f)
            return data_dict
        else:
            with open(load_path) as f:
                results = json.load(f)

        reward_metrics = ['r_nbr_col', 'r_bnd_col', 'r_progress',
                          'r_lane_choice', 'r_route_remain',
                          'r_traffic_line']
        traj_modes = ['top_score_plan']

        metric_dict = {}
        takeover_data = './test_takeover_dataset.json'
        with open(takeover_data) as f:
            data_dict = json.load(f)

        # Clip-wise metrics
        if mode == 'takeover_eval':
            metric_dict['clipwise'] = {}
            for m in reward_metrics:
                for traj_mode in traj_modes:
                    list_data = []
                    name = 'rl_' + traj_mode + '_' + m
                    for x in results:
                        if x[name] is not None:
                            # breakpoint()
                            takeover_reason = data_dict[x['dat_name']]['takeover']
                            list_data.append([
                                takeover_reason,
                                x['dat_name'],
                                x['timestamp'],
                                int(x['take_over_index']),
                                bool(x['take_over']),
                                float(x[name][0])])
                    df = pd.DataFrame(list_data, columns=['scenario',
                                                          'dat_name',
                                                          'timestamp',
                                                          'take_over_index',
                                                          'take_over',
                                                          name])

                    # indicator over whole dat (clip)
                    if any(key in name for key in ['r_nbr_col', 'r_bnd_col', 'r_traffic_line', 'r_route_remain']):
                        metric_dict['clipwise'][name] = df.groupby(['dat_name'])[name].min().round(3).to_dict()
                    elif any(key in name for key in ['r_progress', 'r_lane_choice']):
                        metric_dict['clipwise'][name] = df.groupby(['dat_name'])[name].median().round(3).to_dict()
                    else:
                        metric_dict['clipwise'][name] = df.groupby(['dat_name'])[name].mean().round(3).to_dict()

        print("[EVAL] reinforcement_learning_metric finished")
        return metric_dict


def calculate_mean_and_ci(data, confidence=0.95):
    num = data.shape[0]
    mean = np.mean(data, axis=0)
    if num < 2:
        h = np.zeros_like(mean)
    else:
        se = stats.sem(data, axis=0)
        h = se * stats.t.ppf((1 + confidence) / 2, num - 1)
    return mean, h


def plot_continue_learn(results, mode='expert_eval'):

    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=300)
    colors = ['#E64B35', '#4DBBD5']
    labels = ['w/o expert data', 'w/ expert data']
    markers = ['o', 's']
    y_max = 0
    for i, result in enumerate(results):
        sorted_results = dict(sorted(result.items(), key=lambda x: int(x[0].rstrip('k'))))

        means, yerrs = [], []

        for key in sorted_results.keys():
            values = np.array(result[key])
            m, ci = calculate_mean_and_ci(values)
            means.append(m)
            yerrs.append(ci)

        n = len(sorted_results)
        x_vals = np.arange(n)

        means = np.array(means)
        yerrs = np.array(yerrs)
        y_max = max(np.max(means + yerrs), y_max)
        ax.plot(x_vals, means, marker=markers[i], label=labels[i],
                color=colors[i], linewidth=2, markersize=6, zorder=3)

        plt.errorbar(
            x_vals, means, yerr=yerrs, fmt='none',
            capsize=3, markersize=5, linewidth=2,
            ecolor=colors[i], elinewidth=1.0,
            alpha=0.4
        )

    metrics = ['1/5', '2/5', '3/5', '4/5', '1']

    ax.set_xticks(x_vals)
    ax.set_xticklabels(metrics, fontsize=13)
    ax.set_xlabel('Dataset proportion', fontsize=14, fontweight='normal')
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='x', labelsize=13)

    # ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.3)
    ax.set_axisbelow(True)

    if mode == 'expert_eval':
        ax.set_ylabel('ADE@6s [m]', fontsize=14, fontweight='normal')
        ax.legend(loc='upper right', frameon=False, fontsize=12)

        name_png = 'continue_learn_ade.png'
        name_pdf = 'continue_learn_ade.pdf'
    else:
        ax.set_ylabel('Total score', fontsize=14, fontweight='normal')
        name_png = 'continue_learn_reward.png'
        name_pdf = 'continue_learn_reward.pdf'
    plt.tight_layout()
    plt.savefig(name_png, bbox_inches='tight', dpi=600)
    plt.savefig(name_pdf, bbox_inches='tight', dpi=600)


def calculate_runs_res(path_1, path_2, mode='takeover_eval'):
    rl_metrics = ReinforceMetricV3(
        evaluate_time=[1, 2, 3, 4, 5, 6]
    )
    reward_keys = [
        'rl_top_score_plan_r_nbr_col',
        'rl_top_score_plan_r_bnd_col',
        'rl_top_score_plan_r_progress',
        'rl_top_score_plan_r_lane_choice',
        'rl_top_score_plan_r_route_remain',
        'rl_top_score_plan_r_traffic_line'
    ]

    final_res = []
    if mode == 'takeover_eval':
        for path in [path_1, path_2]:
            path_res = {}
            pt_files = [f for f in os.listdir(path) if f.endswith('.json')]
            sorted_files = sorted(pt_files)

            for pt_file in sorted_files:
                total_r = {}
                pt_path = os.path.join(path, pt_file)
                results_dict = rl_metrics.compute_metrics(load_path=pt_path)
                for key in reward_keys:
                    # merge similar metrics
                    if key in ['rl_top_score_plan_r_nbr_col', 'rl_top_score_plan_r_progress']:
                        res = np.array(list(results_dict['clipwise'][key].values()))
                        total_r.update(
                            {key.split('_r_')[-1]: np.round(res, 2)}
                        )
                    if key == 'rl_top_score_plan_r_bnd_col':
                        bnd_col = np.array(list(results_dict['clipwise'][key].values()))
                        traffic_line = np.array(list(results_dict['clipwise']["rl_top_score_plan_r_traffic_line" ].values()))
                        total_r.update(
                            {'onroad': np.round((bnd_col + traffic_line) / 2, 2)}
                        )
                    if key == 'rl_top_score_plan_r_lane_choice':
                        route = np.array(list(results_dict['clipwise'][key].values()))
                        drivable_lane = np.array(list(results_dict['clipwise']["rl_top_score_plan_r_route_remain"].values()))
                        total_r.update(
                            {'route': np.round((route + drivable_lane) / 2, 2)}
                        )
                values = np.array([value for key, value in total_r.items()])
                total_r.update(
                    {'total': np.sum(values, axis=0)}
                )

                path_res.update({pt_file.split('res_')[-1].split('.json')[0]: total_r['total']})
            final_res.append(path_res)

    if mode == 'expert_eval':
        for path in [path_1, path_2]:
            path_res = {}
            pt_files = [f for f in os.listdir(path) if f.endswith('.json')]
            sorted_files = sorted(pt_files, key=lambda x: int(x.split('k')[0]))

            for pt_file in sorted_files:
                iter_res = []
                pt_path = os.path.join(path, pt_file)
                results_dict = rl_metrics.compute_metrics(
                    load_path=pt_path, mode='expert_eval')
                for scene_key in results_dict.keys():
                    if scene_key not in ['gameformer/all', 'gameformer/clipwise', 'gameformer/num_dat', 'data_time', 'time']:
                        value = results_dict[scene_key]['top']['ego_ADE'][-1]
                        iter_res.append(value)
                path_res.update({pt_file.split('.json')[0]: iter_res})

            final_res.append(path_res)
    return final_res


if __name__ == "__main__":

    stats_results = calculate_runs_res(
        './test_ADE_with_takeover_training_data',
        './test_ADE_with_mixed_training_data',
        mode='expert_eval'
    )

    plot_continue_learn(stats_results, mode='expert_eval')

    stats_results = calculate_runs_res(
        './test_SCORE_with_takeover_training_data',
        './test_SCORE_with_mixed_training_data'
    )
    plot_continue_learn(stats_results, mode='takeover_eval')

