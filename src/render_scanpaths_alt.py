"""Animate test scanpaths as Gaussian blobs on a 10x10 grid.

Usage: python /path/to/src/render_scanpaths_alt.py  (cwd must hold
paths.csv produced by scripts/paths.py).

For each trial (first 18) keeps rows with precision == 0.8, skips the
first 6 rows of the trial and shows each
goal as a Gaussian bump (std 5) in a 10x10 image, one subplot per trial (6x3).

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def filter(g, orig_side=10, side=10, s=0.01):
    """Gaussian bump of a point on a resampled grid.

    Args:
        g (array-like): (2,) point in [0, orig_side - 1] map coordinates.
        orig_side (int): side of the original map.
        side (int): side of the output grid.
        s (float): bump std in map units (1 / side if 0 or None).

    Returns:
        np.ndarray: (side, side) bump, peak 1 at the nearest grid node.
    """
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
    ddf = ddf.iloc[6:]
    ln = ddf.shape[0]

    fddf = np.array([filter(x, side=side, s=s) for x in ddf.to_numpy()])

    colors = plt.cm.hot(np.linspace(0, 1, ln))
    for t in range(1, ln):
        imgs[i].set_array(
            fddf[t],
        )
        plt.pause(0.1)
