import matplotlib.pyplot as plt
import numpy as np


data_expert = np.load('./reward_data/human_traj_reward.npy')
data_takeover = np.load('./reward_data/model_traj_reward.npy')
data1 = data_takeover[:, 0]
data2 = data_expert[:, 0]


bins = 15
xmin = min(data1.min(), data2.min())
xmax = max(data1.max(), data2.max())

bin_edges = np.linspace(xmin, xmax, bins + 1)
weight_d = np.ones_like(data1, dtype=float) / len(data1)
weight_h = np.ones_like(data2, dtype=float) / len(data2)


def plot_each_fig(data1, data2, name, maximum=0, atol=1e-12):
    def first_reach_zero_index(x, maximum=0., atol=1e-12):
        hits = np.where(np.isclose(x, maximum, atol=atol))[0]

        return int(hits[0]) if hits.size > 0 else None

    def percentile_at_value(x, v):
        return 100 * np.mean(x < v)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    x_data1 = np.sort(data1)
    y_data1 = np.arange(1, len(x_data1) + 1) / len(x_data1) * 100

    idx1 = first_reach_zero_index(x_data1, maximum, atol=atol)
    if idx1 is None:
        assert 0, "Return value is None!"
    v_1 = x_data1[idx1]
    p_1 = float(percentile_at_value(x_data1, v_1))

    x_data2 = np.sort(data2)
    y_data2 = np.arange(1, len(x_data2) + 1) / len(x_data2) * 100

    idx2 = first_reach_zero_index(x_data2, maximum, atol=atol)
    if idx2 is None:
        assert 0, "Return value is None!"
    v_2 = x_data2[idx2]
    p_2 = float(percentile_at_value(x_data2, v_2))

    ax.plot(x_data1, y_data1, linewidth=3.0, label="Model trajectory", color='blue')
    ax.axhline(p_1, linestyle="--", linewidth=2.0, color='gray')
    ax.scatter([v_1], [p_1], s=30, color='gray', zorder=3)
    x_min, x_max = ax.get_xlim()
    x_mid = (x_min + x_max) / 3
    ax.annotate(
        "{:.2f}%".format(p_1),
        xy=(x_mid, p_1),
        xytext=(0, 6),
        textcoords="offset points",
        fontsize=16,
        color="black",
        zorder=10
    )

    ax.plot(x_data2, y_data2, '--', linewidth=3.0, label="Human trajectory", color='orange')
    ax.axhline(p_2, linestyle="--", linewidth=2.0, color='gray')
    ax.scatter([v_2], [p_2], s=30, color='gray', zorder=3)
    ax.annotate(
        "{:.2f}%".format(p_2),
        xy=(x_mid, p_2),
        xytext=(0, 6),
        textcoords="offset points",
        fontsize=16,
        color="black",
        zorder=10
    )

    if name == "reward_progress":
        ax.set_xlabel('Efficiency reward', fontsize=20, fontweight='light')
        ax.set_xlim(0.2, 1.05)
    elif name == "reward_neighbor_collision":
        ax.set_xlabel('Safety reward', fontsize=20, fontweight='light')
        ax.set_xlim(-10, 0.5)
    elif name == "reward_route":
        ax.set_xlabel('Route-following reward', fontsize=20, fontweight='light')
        ax.set_xlim(0., 1.0)
    elif name == "reward_onroad":
        ax.set_xlabel('Onroad reward', fontsize=20, fontweight='light')
        ax.set_xlim(-4.5, 0.2)

    ax.set_ylabel('Percentile (%)', fontsize=20)
    ax.set_ylim(0., 100)
    ax.tick_params(axis='x', which='major', labelsize=16)
    ax.tick_params(axis='y', which='major', labelsize=16)

    if name == "reward_neighbor_collision":
        ax.legend(frameon=True, fontsize=20)
    plt.tight_layout()

    plt.savefig('./' + name + '.pdf',
                dpi=400,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.savefig('./' + name + '.png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')


## safety
plot_each_fig(data1, data2, 'reward_neighbor_collision', maximum=0.)

## progress
data3 = data_takeover[:, 2]
data4 = data_expert[:, 2]
plot_each_fig(data3, data4, 'reward_progress', maximum=1.0)

## route
data5 = data_takeover[:, 3]
data6 = data_expert[:, 3]
plot_each_fig(data5, data6, 'reward_route', maximum=0.95, atol=0.1)

## on-road
data7 = data_takeover[:, 4]
data8 = data_expert[:, 4]
plot_each_fig(data7, data8, 'reward_onroad', maximum=0.)