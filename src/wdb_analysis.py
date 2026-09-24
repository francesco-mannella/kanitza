"""Plot competence and weight-change statistics of a wandb parameter sweep.

Usage: python /path/to/src/wdb_analysis.py  (any cwd; needs wandb login).

Inputs: ./stats.csv if present, otherwise downloads the history of every
run in francesco-mannella/eye-simulation whose name contains "predgrid"
(decay < 6), parsing decay and local_decay from the "_d_XXXXX_l_XXXXX"
name part (any "_p_..." suffix is ignored), and caches it to
./stats.csv.

Outputs: parameter_exploration.png with four panels (competence at step 499;
moving-average weight change of visual_conditions, visual_effects and
attention maps at step 400) over the decay x local_decay grid.
"""
# %%
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import wandb

def flt(x, win=150):
    """Moving average of x over win samples (same length as x)."""
    return np.convolve(x, np.ones(win) / win, mode="same")


try:
    stats = pd.read_csv("stats.csv")
except FileNotFoundError:

    project_name = "eye-simulation"
    entity_name = "francesco-mannella"
    api = wandb.Api()
    entity, project = entity_name, project_name
    runs = api.runs(entity + "/" + project)

    stats = []
    names = []
    for run in runs:

        if "predgrid" in run.name:
            names.append(run.name)
            decay = float(re.sub(r".*_d_(..)(...)_l_.*", r"\1.\2", run.name))
            local_decay = float(re.sub(r".*_l_(..)(...)(_.*)?$", r"\1.\2", run.name))

            if decay < 6:
                df = run.history()
                df.loc[:, "run"] = run.name
                df.loc[:, "decay"] = decay
                df.loc[:, "local_decay"] = local_decay
                stats.append(df)

    stats = pd.concat(stats)
    stats = stats[
        [
            "run",
            "_step",
            "competence",
            "visual_conditions",
            "visual_effects",
            "attention",
            "decay",
            "local_decay",
        ]
    ]

    stats.to_csv("stats.csv")

# %%

for var_name, orig_var_name in zip(
    ["cond_base", "eff_base", "att_base"],
    ["visual_conditions", "visual_effects", "attention"],
):

    stats.loc[:, var_name] = (
        stats.groupby("run", as_index=False)[orig_var_name]
        .transform(flt)
        .to_numpy()
    )
# %%
sns.set_style("white")
p1 = (
    so.Plot(
        stats.query("_step==499"),
        x="decay",
        y="local_decay",
        pointsize="competence",
    )
    .add(so.Dot(marker="s", color="#444"), legend=False)
    .scale(pointsize=(1, 10))
    .label(title="comp")
    .limit(
        x=(1, 4.5),
        y=(0.0, 4.5),
    )
)


ps = []
for var in ["cond_base", "eff_base", "att_base"]:
    ps.append(
        so.Plot(
            stats.query("_step==400 and decay < 6"),
            x="decay",
            y="local_decay",
            pointsize=var,
        )
        .add(so.Dot(marker="s", color="#444"), legend=False)
        .scale(pointsize=(1, 10))
        .label(title=var)
        .limit(
            x=(1, 4.5),
            y=(0.0, 4.5),
        )
    )
#
fig1, axes = plt.subplots(1, 4, figsize=(14, 4))
for idx, ax in enumerate(axes):
    ax.set_aspect("equal")
    if idx == 0:
        p1.on(ax).plot()
    else:
        ax.set_aspect("equal")
        ps[idx - 1].on(ax).plot()
fig1.tight_layout()
fig1.show()
fig1.savefig("parameter_exploration.png")
