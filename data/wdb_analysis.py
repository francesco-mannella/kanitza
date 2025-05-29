import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import wandb


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

        if run.name.find("predgrid") > 0:
            names.append(run.name)
            decay = float(re.sub(r".*_d_(..)(...)_l_.*", r"\1.\2", run.name))
            local_decay = float(re.sub(r".*_l_(..)(...)$", r"\1.\2", run.name))

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


def flt(x, win=150):
    return np.convolve(x, np.ones(win) / win, mode="same")


for var_name, orig_var_name in zip(
    ["cond_base", "eff_base", "att_base"],
    ["visual_conditions", "visual_effects", "attention"],
):

    stats.loc[:, var_name] = (
        stats.groupby("run", as_index=False)[orig_var_name]
        .transform(flt)
        .to_numpy()
    )

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
