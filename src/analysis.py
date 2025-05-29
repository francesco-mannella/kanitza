from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from params import Parameters


dirs = glob("s_*")

params = Parameters()


def get_pdict(params):
    _pdict = vars(params)
    _pdict = {k: _pdict[k] for k in _pdict if k != "param_types"}
    return _pdict


df = pd.DataFrame(columns=list(get_pdict(params).keys()) + ["comp"])
df = df.astype(dtype=params.param_types)
df.comp = df.comp.astype(float)

for index, d in enumerate(dirs):

    with open(f"{d}/loaded_params") as f:
        param_lines = f.readlines()
    param_string = ";".join([p.strip() for p in param_lines])
    params.string_to_params(param_string)
    _dict = get_pdict(params)

    with open(f"{d}/log") as f:
        comp = f.readlines()[-1].strip().replace("comp:", "")
        _dict["comp"] = float(comp)

    df.loc[index, :] = _dict
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
