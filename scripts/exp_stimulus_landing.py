"""Post hoc (2026-10-09), after the Phase 2 paradigm: does the effects region of the activated location follow the true landing?

Usage:
    python scripts/exp_stimulus_landing.py --manifest FILE [--out DIR]

Per run, in fovea pixels: a stimulus at each place p of a 31x31 grid over the 16x16 window (step 0.5)
activates the conditions winner j (hard readout); the saccade d(j) (scaled by fovea_size / fovea_scale)
lands it at q = p - d(j). ce_j is the effects RF centre (spot probe width 3, peak readout 0.5) of the
same location. Only places with q inside the window are testable.
  gap_fwd, gap_stay  mean distance from ce_j to q and to p
  gap_shuf           gap_fwd with the effects units permuted over locations (mean of --perms)
  r, r_null, r_sd, p partial correlation between ce_j and q after removing the part of each that is
                     linear in p (mean over x and y), against the same permutation null
No rule was fixed before running; this is descriptive and labelled post hoc.
"""
import argparse
import csv
import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from evaluate_runs import load_params  # noqa: E402
from exp_predictive_field import rf_centres  # noqa: E402
from exp_remapping_paradigm import spot_images  # noqa: E402
from probe_retinotopy import activations  # noqa: E402

HALF = 7.5
G = np.arange(-HALF, HALF + 0.01, 0.5)
P = np.array([(x, y) for y in G for x in G])


def resid(y, x):
    X = np.c_[x, np.ones(len(x))]
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def partial_r(ce, q, p):
    return np.mean([np.corrcoef(resid(ce[:, k], p), resid(q[:, k], p))[0, 1] for k in (0, 1)])


def evaluate(path, args, rng):
    store = torch.load(os.path.join(path, "off_control_store"), map_location="cpu", weights_only=False)
    w = {n: store[f"{n}_map_state_dict"]["weights"].numpy().astype(float)
         for n in ["visual_conditions", "visual_effects", "attention"]}
    d = (w["attention"].T - 0.5) * args.retina_scale * args.fovea / load_params(path).fovea_scale[0]
    j = activations(spot_images(P, np.percentile(w["visual_conditions"], 99)), w["visual_conditions"])["hard"].argmax(1)
    ce = rf_centres(w["visual_effects"], 3.0, 3000, np.random.RandomState(1), 0.5)
    q = P - d[j]
    t = (np.abs(q) <= HALF).all(1)
    row = dict(run=os.path.basename(path), testable=t.mean())
    if t.sum() < 20:
        return row
    gap = lambda c: np.linalg.norm(c - q, axis=1)[t].mean()
    perm = [rng.permutation(len(ce)) for _ in range(args.perms)]
    r = partial_r(ce[j][t], q[t], P[t])
    null = np.array([partial_r(ce[k][j][t], q[t], P[t]) for k in perm])
    row.update(gap_fwd=gap(ce[j]), gap_stay=np.linalg.norm(ce[j] - P, axis=1)[t].mean(),
               gap_shuf=np.mean([gap(ce[k][j]) for k in perm]),
               r=r, r_null=null.mean(), r_sd=null.std(), p=(1 + (null >= r).sum()) / (len(null) + 1))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--perms", type=int, default=300)
    ap.add_argument("--retina-scale", type=float, default=80)
    ap.add_argument("--fovea", type=float, default=16)
    args = ap.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)
    rng = np.random.RandomState(0)
    rows = [dict(seed=r["seed"], **evaluate(r["run"], args, rng)) for r in csv.DictReader(open(args.manifest))]
    for r in rows:
        print(" ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()), flush=True)
    os.makedirs(args.out, exist_ok=True)
    name = "landing_" + os.path.splitext(os.path.basename(args.manifest))[0] + ".csv"
    with open(os.path.join(args.out, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
