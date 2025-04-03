# %%

from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from seaborn import axes_style

import matplotlib


matplotlib.use("agg")

dirs = "s_1*"

dfs = []

for d in glob(dirs):
    test_dir = d

    file_regex = f"{test_dir}/goals*npy"
    files = glob(file_regex)
    if len(files) > 0:
        ndict = []
        for filename in files:
            np_dict = pd.DataFrame(np.load(filename, allow_pickle=True)[0])
            columns = np_dict.columns
            ndict.append(np_dict)

        df = pd.DataFrame(np.concat(ndict))
        df.columns = columns

        df.loc[:, "sim"] = d
        df.loc[:, "pos.x"] = np.stack(df.position)[:, 0]
        df.loc[:, "pos.y"] = np.stack(df.position)[:, 1]
        df.loc[:, "goal.x"] = np.stack(df.goal)[:, 0, 0]
        df.loc[:, "goal.y"] = np.stack(df.goal)[:, 0, 1]
        df.loc[:, "saccade"] = [
            int(x.replace("0000-", "")) for x in df.saccade_id
        ]
        df.loc[:, "rot"] = df.angle * 180 / np.pi

        dfs.append(df)

        so.Plot.config.theme.update(
            axes_style("whitegrid", {"font_scale": 1, "axes.grid": True})
        )

        so.Plot.config.display["scaling"] = 0.7
        p = (
            so.Plot(
                df,
                x="goal.y",
                y="goal.x",
                linewidth="rot",
                color="world",
            )
            .add(
                so.Path(
                    marker="o",
                ),
            )
            .scale(
                x=so.Continuous().tick(at=np.arange(10)),
                y=so.Continuous().tick(at=np.arange(10)),
            )
            .limit(
                x=(-0.5, 9.5),
                y=(-0.5, 9.5),
            )
            .layout(
                size=(8, 5),
                extent=(0.1, 0.1, 0.75, 0.9),
            )
        )

        sns.set_style("whitegrid")
        fig, ax = plt.subplots()
        p.on(ax).show()

        fig.savefig(f"{test_dir}.png")

dfs = pd.concat(dfs)
dfs.to_csv("paths.csv")
