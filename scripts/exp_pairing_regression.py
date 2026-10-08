"""Post hoc (2026-10-08), after the Phase 2 paradigm: is the effects field the conditions field shifted by the saccade?

Usage:
    python scripts/exp_pairing_regression.py --manifest FILE [--out DIR]

Per run, in fovea pixels (saccade vectors d(j) scaled by fovea_size / fovea_scale), with RF centres
cc (conditions) and ce (effects) from the spot probe (width 3), mean readout and peak readout 0.5:
  cos_dd   mean cosine between cc - ce and d, with a null that shuffles d over cells (p, 500 shuffles)
  cos_cc   mean cosine between cc and d (the saccade goes to the conditions field)
  a, b     least-squares coefficients of d ~ a * cc + b * ce over both coordinates, no intercept
Forward remapping predicts a = -b: the effects field is the conditions field shifted by d.
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


def cos(a, b):
    return ((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)).mean()


def evaluate(path, args):
    store = torch.load(os.path.join(path, "off_control_store"), map_location="cpu", weights_only=False)
    w = {n: store[f"{n}_map_state_dict"]["weights"].numpy().astype(float)
         for n in ["visual_conditions", "visual_effects", "attention"]}
    d = (w["attention"].T - 0.5) * args.retina_scale * args.fovea / load_params(path).fovea_scale[0]
    row = dict(run=os.path.basename(path))
    for tag, peak in [("mean", 0.0), ("peak", 0.5)]:
        rng = np.random.RandomState(1)
        cc = rf_centres(w["visual_conditions"], 3.0, 3000, rng, peak)
        ce = rf_centres(w["visual_effects"], 3.0, 3000, rng, peak)
        f = cos(cc - ce, d)
        null = np.array([cos(cc - ce, d[rng.permutation(len(d))]) for _ in range(500)])
        a, b = np.linalg.lstsq(np.stack([cc.ravel(), ce.ravel()], 1), d.ravel(), rcond=None)[0]
        row.update({f"cos_dd_{tag}": f, f"null_sd_{tag}": null.std(), f"p_{tag}": (1 + (null >= f).sum()) / 501,
                    f"cos_cc_{tag}": cos(cc, d), f"a_{tag}": a, f"b_{tag}": b})
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--retina-scale", type=float, default=80.0)
    ap.add_argument("--fovea", type=int, default=16)
    args = ap.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for r in csv.DictReader(open(args.manifest)):
        row = evaluate(r["run"], args)
        rows.append(row)
        print(f"{row['run']:30s} cos_dd {row['cos_dd_mean']:5.2f} p {row['p_mean']:.3f} cos_cc {row['cos_cc_mean']:5.2f} "
              f"| peak a {row['a_peak']:5.2f} b {row['b_peak']:5.2f}", flush=True)
    with open(os.path.join(args.out, "pairing_" + os.path.splitext(os.path.basename(args.manifest))[0] + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
