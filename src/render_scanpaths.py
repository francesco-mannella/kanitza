import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def filter(g, orig_side=10, side=10, s=0.01):
    s = s or 1 / side
    res = np.zeros((side, side))
    t = np.linspace(0, orig_side - 1, side)
    for x in range(side):
        for y in range(side):
            diff = np.array(g) - [t[x], t[y]]
            res[x, y] = np.exp(-(s**-2) * np.dot(diff, diff))
    return res


df = pd.read_csv("paths.csv")

trials = df.trial.unique()

fig, axes = plt.subplots(6, 3)

axes = axes.flatten()
side = 7

for ax in axes:
    ax.set_axis_off()
    ax.set_xlim(-1, side)
    ax.set_ylim(-1, side)
fig.tight_layout(pad=0.1)

dfp = df.query("precision == 0.7")
for ax, trial in zip(axes, trials):
    ddf = dfp.query(f"trial=='{trial}'")[["goal.x", "goal.y"]]
    ddf = ddf.loc[6:, :]
    ln = ddf.shape[0]

    fddf = np.array([filter(x, side=side) for x in ddf.to_numpy()])
    fddf = fddf.reshape(ln, -1)
    fddf = np.argmax(fddf, -1)
    fddf = np.array([[x // side, x % side] for x in fddf])

    colors = plt.cm.hot(np.linspace(0, 1, ln))
    for t in range(1, ln):
        ax.plot(
            fddf[(t - 1) : (t + 1), 0],
            fddf[(t - 1) : (t + 1), 1],
            c="black",
            alpha=0.3,
        )
        ax.scatter(
            fddf[(t - 1) : (t + 1), 0],
            fddf[(t - 1) : (t + 1), 1],
            c="black",
            alpha=0.5,
            s=2 * t + 1,
        )
        plt.pause(0.001)
