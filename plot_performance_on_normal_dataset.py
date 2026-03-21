import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import scipy.stats as stats


normal_data_path_pretrained = ['./predeploy_model_on_normal_test_dataset']
normal_data_path_postrained = ['./postrain_model_on_normal_test_dataset']

ade_file_name = 'eval_results.json'
ade_keys = ['top', 'ego_ADE']

collision_cost_name = 'collision_cost.pkl'

horizons = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
xticklabels = ['1', '2', '3', '4', '5', '6']


def calculate_mean_and_ci(data, confidence=0.95):
    num = data.shape[0]
    mean = np.mean(data, axis=0)
    if num < 2:
        h = np.zeros_like(mean)
    else:
        se = stats.sem(data, axis=0)
        h = se * stats.t.ppf((1 + confidence) / 2, num - 1)
    return mean, h


def load_data(path_all):
    ade_model = []
    collision_model = []
    for path in path_all:
        ade_path = path + '/' + ade_file_name
        coll_path = path + '/' + collision_cost_name
        with open(ade_path, "r", encoding='utf-8') as f:
            loaded_dict = json.load(f)
            for scene_key in loaded_dict.keys():
                if scene_key not in ['gameformer/all', 'gameformer/clipwise', 'gameformer/num_dat']:
                    value = loaded_dict[scene_key][ade_keys[0]][ade_keys[1]]
                    ade_model.append(value)
        with open(coll_path, 'rb') as f:
            loaded_dict = pickle.load(f)
            cost = loaded_dict['top_collision_cost']
            collision_model.append(cost)
    ade_model = np.stack(ade_model, axis=0)
    collision_model = np.concatenate(collision_model, axis=0)
    return ade_model, collision_model


ade_data_pretrained, cost_data_pretrained = load_data(normal_data_path_pretrained)
ade_data_postrained, cost_data_postrained = load_data(normal_data_path_postrained)

data_pretrained_mean, data_pretrained_ci = calculate_mean_and_ci(ade_data_pretrained)
data_postrained_mean, data_postrained_ci = calculate_mean_and_ci(ade_data_postrained)

# ade
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
color_pre = '#E64B35'
color_pos = '#4DBBD5'

ax.plot(horizons, data_pretrained_mean, label='Pre-deployed model',
        color=color_pre, linewidth=2, marker='o', markersize=6, zorder=3)
ax.plot(horizons, data_postrained_mean, label='Post-trained model',
        color=color_pos, linewidth=2, marker='s', markersize=6, zorder=3)
ax.fill_between(horizons, data_pretrained_mean - data_pretrained_ci,
                data_pretrained_mean + data_pretrained_ci,
                color=color_pre, alpha=0.2, edgecolor='none', zorder=2)
ax.fill_between(horizons, data_postrained_mean - data_postrained_ci,
                data_postrained_mean + data_postrained_ci,
                color=color_pos, alpha=0.2, edgecolor='none', zorder=2)

ax.set_xticks(horizons)
ax.set_xticklabels(xticklabels, fontsize=13)
ax.set_xlabel('Horizon [s]', fontsize=14, fontweight='normal')
ax.set_ylabel('ADE [m]', fontsize=14, fontweight='normal')
ax.tick_params(axis='y', labelsize=13)
ax.tick_params(axis='x', labelsize=13)

y_max = np.max(np.concatenate([data_pretrained_mean + data_pretrained_ci,
                               data_postrained_mean + data_postrained_ci]))
ax.set_ylim([0, y_max * 1.1])
ax.set_xlim([0.9, 6.1])

ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.3)
ax.set_axisbelow(True)

ax.legend(loc='upper left', frameon=False, fontsize=14)

plt.tight_layout()
plt.savefig('normal_ade.png', bbox_inches='tight', dpi=300)
plt.savefig('normal_ade.pdf', bbox_inches='tight', dpi=300)


# collision cost
data_all = [cost_data_pretrained, cost_data_postrained]
collision_rate = [np.sum(cost_data_pretrained > 0.1) / len(cost_data_pretrained) * 100,
                  np.sum(cost_data_postrained > 0.1) / len(cost_data_postrained) * 100,]
data_nonzero_all = [cost_data_pretrained[cost_data_pretrained > 0.1], cost_data_postrained[cost_data_postrained > 0.1]]

labels = ['Pre-deployed model', 'Post-trained model']
colors = ['#E64B35', '#4DBBD5']
edge_colors = ['#B0301E', '#3A8FA8']

fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
parts = ax.violinplot(data_nonzero_all, showmeans=False, showmedians=False, showextrema=False)

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor(edge_colors[i])
    pc.set_linewidth(1.0)
    pc.set_alpha(0.7)

ax.boxplot(data_nonzero_all, positions=[1.0, 2.0], widths=0.06,
           patch_artist=True, showfliers=False,
           boxprops=dict(facecolor='white', alpha=1.0, edgecolor='black', linewidth=0.8),
           medianprops=dict(color='black', linewidth=1.5),
           whiskerprops=dict(color='black', linewidth=0.8, linestyle='-'),
           capprops=dict(color='black', linewidth=0.8),
           zorder=3)

y_max = max(np.max(data_nonzero_all[0]), np.max(data_nonzero_all[1]))

for i, rate in enumerate(collision_rate):
    text_str = f"Collision rate: \n{rate:.2f}%"
    ax.text(i + 1, y_max * 1.18, 'Collsion rate',
            ha='center', va='bottom', fontsize=12, fontweight='normal', color='#333333')
    ax.text(i + 1, y_max * 1.08, f'{rate:.2f}%',
            ha='center', va='bottom', fontsize=14, fontweight='normal', color='black')

ax.set_ylabel('Collision Severity', fontsize=14, fontweight='normal')
ax.set_xticks([1.0, 2.0])
ax.set_xticklabels(labels, fontsize=14, fontweight='normal')
ax.tick_params(axis='y', labelsize=13)
ax.set_ylim(0, y_max * 1.35)
plt.tight_layout()
plt.savefig('normal_cost.png', bbox_inches='tight', dpi=300)
plt.savefig('normal_cost.pdf', bbox_inches='tight', dpi=300)
