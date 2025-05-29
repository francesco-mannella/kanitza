# %%

from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import slugify
from seaborn import axes_style


matplotlib.use("agg")

dirs = "*"

dfs = []
sns.set_style("whitegrid")
so.Plot.config.display["scaling"] = 0.7

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
        df.loc[:, "goal.y"] = np.stack(df.goal)[:, 0, 0]
        df.loc[:, "goal.x"] = np.stack(df.goal)[:, 0, 1]
        df.loc[:, "saccade"] = [
            int(x.replace("0000-", "")) for x in df.saccade_id
        ]
        df.loc[:, "Rot"] = df.angle * 180 / np.pi
        df = df.rename(columns={"world": "Object"})
        df = df.query("saccade > 2")
        df.loc[:, "trial"] = [
            slugify.slugify(f"{obj}-{angle}")
            for obj, angle in zip(df.Object, df.angle)
        ]

        dfs.append(df)

        so.Plot.config.theme.update(
            axes_style("whitegrid", {"font_scale": 1, "axes.grid": True})
        )

        fig, ax = plt.subplots()
        p = (
            so.Plot(
                df,
                x="goal.x",
                y="goal.y",
            )
            .add(
                so.Path(
                    marker="o",
                    alpha=0.5,
                ),
                linewidth="Rot",
                color="Object",
                group="trial",
            )
            .scale(
                x=so.Continuous().tick(at=np.arange(10)),
                y=so.Continuous().tick(at=np.arange(10)),
                color=("#ff5555", "#5555ff"),
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

        p.on(ax).plot()

        fig.legends[0].set_bbox_to_anchor((0.78, 0.5))

        fig.savefig(f"{test_dir}.png")

        if test_dir.find("d_03500_l_00500") >= 0:

            for i, obj in enumerate(df.Object.unique()):
                for j, angle in enumerate(df.angle.unique()):
                    fig, ax = plt.subplots()

                    dfline = df.query(f"Object=='{obj}' and angle=={angle}")

                    p.add(
                        so.Path(linewidth=3, alpha=1),
                        data=dfline,
                        legend=False,
                        color="Object",
                    ).on(ax).plot()

                    fig.legends[0].set_bbox_to_anchor((0.78, 0.5))

                    fig.savefig(
                        slugify.slugify(f"{test_dir}_{obj}_{angle}.png")
                    )

            ddf = df.copy()

pddfs = pd.concat(dfs)
pddfs = pddfs[
    [
        "Object",
        "saccade_id",
        "sim",
        "pos.x",
        "pos.y",
        "goal.y",
        "goal.x",
        "saccade",
        "Rot",
        "trial",
    ]
]

pddfs = pddfs.rename(
    columns={
        "Object": "object",
        "saccade": "ts",
        "Rot": "rot",
    }
)


pddfs.loc[:, "saccade_num"] = pddfs.groupby(["trial", "trial", "rot"])[
    "ts"
].transform(lambda x: np.arange(len(x)))

pddfs.to_csv("paths.csv")

# %%

# %%
