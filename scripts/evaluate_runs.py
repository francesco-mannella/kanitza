"""Score simulation runs on competence, weight stability, topography and
prototype definition.

Usage:
    python /path/to/scripts/evaluate_runs.py PATH [PATH ...] [--csv FILE]
        [--probes N]

Each PATH is a run folder (with off_control_store and loaded_params) or a
folder searched up to two levels deep for run folders. Compare only runs
trained with the same code version: the probes are built with the current
code (e.g. Gabor filters, fovea gain).

For every run it computes:
    - competence: mean of the last 20 logged epochs (data_sim, else the
      "comp:" lines of log), and the mean grid distances of the attention
      and visual-effects winners from the goal (goal_dist_*, logged by
      newer runs; independent of match_std, lower is better);
    - stability: mean relative weight change per epoch over the last 50
      epochs (logged weight-change norm / final weight norm), per map;
    - topography, per map: Spearman correlation between grid distance and
      prototype distance over all unit pairs (1 = perfectly ordered) and
      the topographic error on probe inputs (share of probes whose best
      and second-best units are not grid neighbors);
    - definition, per map: quantization error on the probes (mean distance
      to the best unit / mean distance of the probes from their mean),
      share of dead units (never best on any probe) and, for the visual
      maps, prototype contrast (mean per-prototype std / mean probe std).

Probes: N foveas (default 1000) from the environment and the agent's
get_fovea, with fixed seed, both objects, random poses and retina
positions near the object; built once per distinct visual setup. The
attention map is probed with points uniform in [0.1, 0.9]^2.

Each criterion gets a rank across the evaluated runs (mean rank of its
sub-metrics, 1 = best); "score" is the mean of the four criterion ranks.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr


SRC = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "src")
sys.path.insert(0, SRC)

import EyeSim  # noqa: E402
import gymnasium as gym  # noqa: E402

from model.agent import Agent  # noqa: E402
from params import Parameters  # noqa: E402


_ = EyeSim
MAPS = {
    "visual_conditions": "vc",
    "visual_effects": "ve",
    "attention": "att",
}
PROBE_KEYS = [
    "taskspace_xlim", "taskspace_ylim", "retina_scale", "retina_size",
    "fovea_scale", "fovea_size", "test_fovea", "fovea_gain",
    "gabor_scales", "gabor_orientation_bins", "gabor_frequency",
    "gabor_phase_offset", "gabor_kernel_size", "gabor_sigma_y_multiplier",
    "gabor_rgb_prop", "gabor_bright_prop",
]


def find_runs(paths):
    """Run folders among paths and their subfolders (two levels)."""
    runs = []
    for path in paths:
        for pattern in ["", "*", "*/*"]:
            for d in sorted(glob.glob(os.path.join(path, pattern))):
                if os.path.isfile(os.path.join(d, "off_control_store")) and \
                        os.path.isfile(os.path.join(d, "loaded_params")):
                    runs.append(os.path.normpath(d))
    return list(dict.fromkeys(runs))


def load_params(run):
    params = Parameters()
    params.load(os.path.join(run, "loaded_params"))
    return params


def visual_probes(params, n, seed=0):
    """(n, input_size) fovea probes, scaled like the controller's input."""
    env = gym.make(params.env_name, params=params).unwrapped
    env.set_seed(seed)
    agent = Agent(env, params, seed=seed)
    rng = np.random.RandomState(seed)
    xlim, ylim = env.taskspace_xlim, env.taskspace_ylim
    probes = []
    for i in range(n):
        pos = [xlim[0] + (xlim[1] - xlim[0]) * (0.3 + 0.4 * rng.rand()),
               ylim[0] + (ylim[1] - ylim[0]) * (0.3 + 0.4 * rng.rand())]
        env.init_world(world=i % 2, object_params={"pos": pos, "rot": 2 * np.pi * rng.rand()})
        env.reset()
        target = np.array(pos) + rng.uniform(-15, 15, 2)
        obs = env.step(target - env.retina_sim_pos)[0]
        probes.append(np.asarray(agent.get_fovea(obs), dtype=float).ravel() / 255.0)
    return np.array(probes)


def curve(run, key):
    """Logged values of key per epoch: data_sim (training runs merged by
    step) or, for competence only, the "comp:" lines of log."""
    values = {}
    for f in sorted(glob.glob(os.path.join(run, "data_sim", "run-*", "metrics.jsonl"))):
        try:
            with open(os.path.join(os.path.dirname(f), "config.json")) as c:
                if json.load(c).get("job_type") not in (None, "train"):
                    continue
        except (OSError, ValueError):
            pass
        for line in open(f):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if key in row:
                values[row["_step"]] = row[key]
    if values:
        return np.array([values[k] for k in sorted(values)], dtype=float)
    if key == "competence" and os.path.isfile(os.path.join(run, "log")):
        return np.array([float(l.split(":", 1)[1]) for l in open(os.path.join(run, "log"))
                         if l.startswith("comp:")])
    return np.array([])


def grid_coords(n_units):
    side = int(round(np.sqrt(n_units)))
    rows, cols = np.divmod(np.arange(n_units), side)
    return np.stack([rows, cols], 1)


def pairwise(x):
    sq = (x ** 2).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * x @ x.T, 0))


def map_metrics(prototypes, probes, visual):
    """Topography and definition metrics of one map.

    Args:
        prototypes (np.ndarray): (units, input_size) unit weights.
        probes (np.ndarray): (n, input_size) probe inputs.
        visual (bool): also compute prototype contrast.
    """
    coords = grid_coords(len(prototypes))
    iu = np.triu_indices(len(prototypes), 1)
    rho = spearmanr(pairwise(coords.astype(float))[iu], pairwise(prototypes)[iu])[0]

    d2 = (probes ** 2).sum(1)[:, None] + (prototypes ** 2).sum(1)[None] - 2 * probes @ prototypes.T
    order = np.argsort(d2, 1)
    best, second = order[:, 0], order[:, 1]
    cheb = np.abs(coords[best] - coords[second]).max(1)
    spread = np.linalg.norm(probes - probes.mean(0), axis=1).mean()
    out = dict(
        spearman=rho,
        topo_error=float((cheb > 1).mean()),
        quant_error=float(np.sqrt(np.maximum(d2[np.arange(len(d2)), best], 0)).mean() / spread),
        dead=float(1 - len(np.unique(best)) / len(prototypes)),
    )
    if visual:
        out["contrast"] = float(prototypes.std(1).mean() / probes.std(1).mean())
    return out


def evaluate(run, probe_cache, n_probes):
    params = load_params(run)
    key = json.dumps({k: getattr(params, k, None) for k in PROBE_KEYS}, sort_keys=True)
    if key not in probe_cache:
        probe_cache[key] = visual_probes(params, n_probes)
    probes = probe_cache[key]
    att_probes = 0.1 + 0.8 * np.random.RandomState(1).rand(n_probes, 2)

    store = torch.load(os.path.join(run, "off_control_store"), map_location="cpu",
                       weights_only=False)
    row = dict(run=run, epochs=store["epoch"] + 1)
    comp = curve(run, "competence")
    row["competence"] = comp[-20:].mean() if len(comp) else np.nan
    for k in ["goal_dist_attention", "goal_dist_effects"]:
        c = curve(run, k)
        row[k] = c[-20:].mean() if len(c) else np.nan

    for name, short in MAPS.items():
        weights = store[f"{name}_map_state_dict"]["weights"].numpy().astype(float)
        change = curve(run, name)
        row[f"stab_{short}"] = (change[-50:].mean() / np.linalg.norm(weights)
                                if len(change) else np.nan)
        visual = name != "attention"
        metrics = map_metrics(weights.T, probes if visual else att_probes, visual)
        row.update({f"{m}_{short}": v for m, v in metrics.items()})
    return row


CRITERIA = {
    "competence": [("competence", +1), ("goal_dist_attention", -1), ("goal_dist_effects", -1)],
    "stability": [("stab_vc", -1), ("stab_ve", -1), ("stab_att", -1)],
    "topography": [("spearman_vc", +1), ("spearman_ve", +1), ("spearman_att", +1),
                   ("topo_error_vc", -1), ("topo_error_ve", -1), ("topo_error_att", -1)],
    "definition": [("quant_error_vc", -1), ("quant_error_ve", -1), ("dead_vc", -1),
                   ("dead_ve", -1), ("dead_att", -1), ("contrast_vc", +1), ("contrast_ve", +1)],
}


def add_ranks(rows):
    """Criterion ranks (1 = best) and overall score (mean criterion rank)."""
    def rank(values, sign):
        """Rank 1 = best; ties share their average rank; NaN ranks last."""
        v = np.array(values, dtype=float)
        if np.all(np.isnan(v)):
            return np.full(len(v), np.nan)
        return rankdata(-sign * np.where(np.isnan(v), -sign * np.inf, v))

    for crit, metrics in CRITERIA.items():
        ranks = [rank([r[m] for r in rows], s) for m, s in metrics]
        mean = np.nanmean(np.array(ranks), 0)
        for r, v in zip(rows, mean):
            r[f"rank_{crit}"] = v
    for r in rows:
        r["score"] = np.nanmean([r[f"rank_{c}"] for c in CRITERIA])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--csv", help="Write all metrics to this CSV file.")
    parser.add_argument("--probes", type=int, default=1000)
    args = parser.parse_args()

    runs = find_runs(args.paths)
    if not runs:
        sys.exit("No run folders found.")
    cache, rows = {}, []
    for run in runs:
        print(f"evaluating {run}", file=sys.stderr)
        rows.append(evaluate(run, cache, args.probes))
    add_ranks(rows)
    rows.sort(key=lambda r: r["score"])

    columns = list(rows[0])
    if args.csv:
        with open(args.csv, "w") as f:
            f.write(",".join(columns) + "\n")
            for r in rows:
                f.write(",".join(str(r[c]) for c in columns) + "\n")

    show = ["score", "rank_competence", "rank_stability", "rank_topography",
            "rank_definition", "epochs", "competence", "goal_dist_attention",
            "stab_vc", "spearman_vc", "spearman_ve", "spearman_att", "topo_error_vc",
            "quant_error_vc", "quant_error_ve", "dead_ve", "dead_att", "contrast_vc"]
    names = [os.path.relpath(r["run"]) for r in rows]
    width = max(len(n) for n in names)
    print(f"{'run':{width}s} " + " ".join(f"{c[:11]:>11s}" for c in show))
    for n, r in zip(names, rows):
        print(f"{n:{width}s} " + " ".join(f"{r[c]:11.3f}" for c in show))


if __name__ == "__main__":
    main()
