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
side = 10
s = 5
imgs = []
for i, ax in enumerate(axes):
    ax.set_axis_off()
    ax.set_xlim(-1, side)
    ax.set_ylim(-1, side)
    imgs.append(ax.imshow(np.zeros([side, side]), vmin=0, vmax=1))

fig.tight_layout(pad=0.1)

dfp = df.query("precision == 0.8")
for i, (ax, trial) in enumerate(zip(axes, trials)):
    ddf = dfp.query(f"trial=='{trial}'")[["goal.x", "goal.y"]]
    ddf = ddf.loc[6:, :]
    ln = ddf.shape[0]

    fddf = np.array([filter(x, side=side, s=s) for x in ddf.to_numpy()])

    colors = plt.cm.hot(np.linspace(0, 1, ln))
    for t in range(1, ln):
        imgs[i].set_array(
            fddf[t],
        )
        plt.pause(0.1)
