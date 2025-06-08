import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import slugify
from seaborn import axes_style


# Configuration
SIMULATION_DIR_PATTERN = "sim*"
OUTPUT_CSV_FILENAME = "paths.csv"
sns.set_style("whitegrid")
so.Plot.config.display["scaling"] = 0.7

# Process simulation directories
simulation_dfs = []
for sim_dir in glob(SIMULATION_DIR_PATTERN):
    print(sim_dir)

    # Load goal data
    goal_files = glob(f"{sim_dir}/goals*npy")
    if not goal_files:
        continue

    goal_dicts = []
    for filename in goal_files:
        np_array = np.load(filename, allow_pickle=True)[0]
        goal_df = pd.DataFrame(np_array)
        goal_dicts.append(goal_df)

    combined_df = pd.concat(goal_dicts, ignore_index=True)
    combined_df.columns = goal_df.columns

    # Extract and transform data
    combined_df["sim"] = sim_dir
    combined_df["pos.x"] = np.stack(combined_df["position"])[:, 0]
    combined_df["pos.y"] = np.stack(combined_df["position"])[:, 1]
    combined_df["goal.y"] = np.stack(combined_df["goal"])[:, 0, 0]
    combined_df["goal.x"] = np.stack(combined_df["goal"])[:, 0, 1]
    combined_df["saccade"] = [
        int(x.replace("0000-", "")) for x in combined_df["saccade_id"]
    ]
    combined_df["Rot"] = combined_df["angle"] * 180 / np.pi
    combined_df = combined_df.rename(columns={"world": "Object"})
    combined_df = combined_df.query("saccade > 2")
    combined_df["trial"] = [
        slugify.slugify(f"{obj}-{angle}")
        for obj, angle in zip(combined_df["Object"], combined_df["angle"])
    ]

    simulation_dfs.append(combined_df)

    # Plotting
    so.Plot.config.theme.update(
        axes_style("whitegrid", {"font_scale": 1, "axes.grid": True})
    )

    fig, ax = plt.subplots()
    plot = (
        so.Plot(
            combined_df,
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
            color=("#5555ff", "#ff5555"),
        )
        .limit(
            x=(-0.5, 9.5),
            y=(-0.5, 9.5),
        )
        .layout(
            size=(8, 5),
            extent=(0.1, 0.1, 0.75, 0.9),
        )
        .label(title=re.sub(r"sim_.*_s_", r"s_", sim_dir))
    )

    plot.on(ax).plot()
    fig.legends[0].set_bbox_to_anchor((0.78, 0.5))
    fig.savefig(f"{sim_dir}.png")


# Concatenate and save dataframes
all_simulations_df = pd.concat(simulation_dfs, ignore_index=True)
processed_df = all_simulations_df[
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

processed_df = processed_df.rename(
    columns={
        "Object": "object",
        "saccade": "ts",
        "Rot": "rot",
    }
)

processed_df["saccade_num"] = processed_df.groupby(["trial", "trial", "rot"])[
    "ts"
].transform(lambda x: np.arange(len(x)))

processed_df.to_csv(OUTPUT_CSV_FILENAME, index=False)
