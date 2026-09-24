"""Scatter plot of final competence over the decay-speed parameter grid.

Usage: run from the directory holding the simulation folders:
    python /path/to/src/analysis.py

Inputs: for every folder matching "*_s_*_m_*": <dir>/loaded_params and the
last "comp: <float>" line of <dir>/log (written only when main.py ran with
-w); folders without such a line are skipped.
Outputs: an interactive seaborn plot, x=decaying_speed,
y=local_decaying_speed, point size=competence.
"""
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from params import Parameters


dirs = glob("*_s_*_m_*/")


def get_pdict(params):
    """Return the parameters of `params` as a dict without param_types."""
    _pdict = vars(params)
    _pdict = {k: _pdict[k] for k in _pdict if k != "param_types"}
    return _pdict


rows = []
for d in dirs:

    with open(f"{d}/log") as f:
        comps = [line for line in f if line.startswith("comp:")]
    if not comps:
        continue

    params = Parameters()
    params.load(f"{d}/loaded_params")
    _dict = get_pdict(params)
    _dict["comp"] = float(comps[-1].replace("comp:", ""))
    rows.append(_dict)

df = pd.DataFrame(rows)
df["comp_scaled"] = (df.comp - df.comp.min()) / (
    df.comp.max() - df.comp.min()
)


p = (
    so.Plot(
        df,
        x="decaying_speed",
        y="local_decaying_speed",
        pointsize="comp",
    )
    .add(so.Dot(marker="s", color="#444"))
    .scale(pointsize=(1, 18))
    .layout(extent=(0.1, 0.1, 0.7, 0.9))
    .limit(
        x=(0, 5.5),
        y=(0.0, 5.5),
    )
)
sns.set_style("white")
fig, ax = plt.subplots()
ax.set_aspect("equal")
p.on(ax).show()
