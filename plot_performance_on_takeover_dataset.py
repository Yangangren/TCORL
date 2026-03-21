import pandas as pd
import json

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.stats as stats


class ReinforceMetric():
    def __init__(self, mode='evaluate'):
        self.EGO_SIZE = [1.8, 4.9, 1.8]
        self.safety_distance = 0.1
        self.mode = mode,

    def compute_metrics(self, load_path):
        print("[EVAL] Start")

        with open(load_path) as f:
            results = json.load(f)
        reward_metrics = ['r_nbr_col', 'r_bnd_col', 'r_progress',
                            'r_lane_choice', 'r_route_remain',
                            'r_traffic_line', 'r_total']
        traj_modes = [
            'top_score_plan'
        ]

        takeover_data = 'test_takeover_dataset.json'
        with open(takeover_data) as f:
            data_dict = json.load(f)

        # Clip-wise metrics
        metric_dict = {}
        metric_dict['clipwise'] = {}
        threshold_before = -20
        threshold_after = 0

        for m in reward_metrics:
            for traj_mode in traj_modes:
                list_data = []
                name = 'rl_' + traj_mode + '_' + m
                for x in results:
                    if x[name] is not None:
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

                # indicator over -20~3 frames near takeover moment
                df_near_take_over = df[
                    (df['take_over_index'] >= threshold_before) & (df['take_over_index'] <= threshold_after)
                ]
                if len(df_near_take_over) > 0:
                    if any(key in name for key in ['r_nbr_col', 'r_bnd_col', 'r_traffic_line', 'r_route_remain']):
                        metric_dict['clipwise'][name + '_near_takeover'] = df_near_take_over.groupby(['scenario', 'take_over_index'])[name].min().xs('collision', level='scenario').round(3).to_dict()
                    elif any(key in name for key in ['r_progress', 'r_lane_choice']):
                        metric_dict['clipwise'][name + '_near_takeover'] = df_near_take_over.groupby(['scenario', 'take_over_index'])[name].median().xs('collision', level='scenario').round(3).to_dict()
                    else:
                        metric_dict['clipwise'][name + '_near_takeover'] = df_near_take_over.groupby(['scenario', 'take_over_index'])[name].mean().xs('collision', level='scenario').round(3).to_dict()
                else:
                    assert 0, "no takeover moment in your data"

        print("[EVAL] Finished")
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


def plot_takeover(plot_keys, results_pre, results_post):
    scores = {'pretrain': {}, 'postrain': {}}
    performance = {'pretrain': {}, 'postrain': {}}
    for key in plot_keys:
        scores['pretrain'].update({key: np.array(list(results_pre['clipwise'][key].values()))})
        scores['postrain'].update({key: np.array(list(results_post['clipwise'][key].values()))})

        key_to = key + "_near_takeover"
        performance['pretrain'].update({key_to: np.array(list(results_pre['clipwise'][key_to].values()))})
        performance['postrain'].update({key_to: np.array(list(results_post['clipwise'][key_to].values()))})

    # merge similar score
    for key in scores.keys():
        scores[key][plot_keys[1]] = (scores[key][plot_keys[1]] + scores[key][plot_keys[5]]) / 2
        scores[key][plot_keys[3]] = (scores[key][plot_keys[3]] + scores[key][plot_keys[4]]) / 2
        scores[key].pop(plot_keys[5])
        scores[key].pop(plot_keys[4])

    plot_keys_new = plot_keys[:4]

    metrics = ['safety', 'compliance', 'efficiency', 'onroad']
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    bar_width = 0.35
    colors = ['#E64B35', '#4DBBD5']
    labels = ['Pre-deployed model', 'Post-trained model']

    for i, (model_name, label) in enumerate(zip(scores.keys(), labels)):
        means, yerrs = [], []

        for metric in plot_keys_new:
            m, ci = calculate_mean_and_ci(scores[model_name][metric])
            means.append(m)
            yerrs.append(ci)

        pos = x - bar_width / 2 if i == 0 else x + bar_width / 2
        rects = ax.bar(pos, means, bar_width,
                       yerr=yerrs,
                       alpha=0.7,
                       color=colors[i],
                       label=label,
                       edgecolor='black',
                       linewidth=0.8,
                       capsize=4,
                       error_kw={'elinewidth': 1.2, 'ecolor': 'black'}
                    )

    ax.set_ylabel('Score', fontsize=14, fontweight='normal')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=14, fontweight='normal')
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='x', labelsize=13)

    ax.legend(loc='upper right', frameon=False, fontsize=14)
    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    plt.savefig('takeover_all_score.png', bbox_inches='tight', dpi=300)
    plt.savefig('takeover_all_score.pdf', bbox_inches='tight', dpi=300)

    # takeover performance between -20-3 frame
    markers = ['o', 's']
    for metric in ['rl_top_score_plan_r_nbr_col_near_takeover']:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

        for i, (model_name, label) in enumerate(zip(scores.keys(), labels)):
            data = performance[model_name][metric][:-1]
            horizons = - len(data) + np.arange(len(data))
            ax.plot(horizons, data, label=labels[i],
                    color=colors[i], linewidth=2, marker=markers[i],
                    markersize=4, zorder=3)

        ax.set_ylabel('Safety Score', fontsize=14, fontweight='normal')
        ax.set_xlabel('Time step', fontsize=14, fontweight='normal')
        ax.tick_params(axis='y', labelsize=13)
        ax.tick_params(axis='x', labelsize=13)
        ax.set_xlim([-20, 0])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

        x_min, x_max = ax.get_xlim()
        ax.fill_between([x_min, x_max - 0.1], 0.3, 0.4, color='gray', alpha=0.3, edgecolor='none')
        ax.text((x_min + x_max) / 1.8, 0.35, 'Intelligent driving', color='black', fontsize=12,
                ha='center', va='center', alpha=0.8)
        if 'nbr_col' in metric:
            ax.set_ylim([0.2, 1.0])

        ax.plot([-0, -0], [0, 0.4], color='blue', linewidth=6, alpha=1.0)
        ax.annotate(f'takeover\nmoment', xy=(0, 0.35), xytext=(-6, 0.5),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                    ha='center', va='bottom', fontsize=12,
                    color='blue', annotation_clip=False)

        plt.tight_layout()
        plt.savefig('takeover_step_score' + '.png', bbox_inches='tight', dpi=300)
        plt.savefig('takeover_step_score' + '.pdf', bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    rl_metrics = ReinforceMetric(
        mode='evaluate',
    )

    pre_path = './predeploy_model_on_takeover_test_dataset/eval_res.json'
    results_pre = rl_metrics.compute_metrics(load_path=pre_path)

    post_path = './postrain_model_on_takeover_test_dataset/eval_res.json'
    results_post = rl_metrics.compute_metrics(load_path=post_path)

    top_score_keys = [
        'rl_top_score_plan_r_nbr_col',
        'rl_top_score_plan_r_bnd_col',
        'rl_top_score_plan_r_progress',
        'rl_top_score_plan_r_lane_choice',
        'rl_top_score_plan_r_route_remain',
        'rl_top_score_plan_r_traffic_line',
        'rl_top_score_plan_r_total'
    ]
    plot_takeover(top_score_keys, results_pre, results_post)
