"""Phase 2.1 (predictive_remapping.md): can the remapping paradigm be tested on the current maps?

Usage:
    python scripts/exp_remap_feasibility.py --manifest FILE [--out DIR]

For each run and cell j: CRF_j is the effects-unit RF centre (post-saccadic field) (spot width 3, soft
readout, mean over spots, in fovea pixels from the fovea centre; as exp_predictive_field.py),
d(j) = (attention prototype - 0.5) * retina_scale * fovea_size / fovea_scale is the saccade vector in fovea pixels, FF_j = CRF_j + d(j) the
forward-predicted future field. A cell is testable if FF_j lies inside the fovea window
(|x|, |y| <= (fovea_size - 1) / 2 pixels). Reported per run: share of cells with |d| > min_move,
the amplitude quantiles of d, the testable fraction, the geometric fraction (CRF uniform over
the window) as a check against the compression of soft RF centres, and the executed
amplitudes at decision steps from replay (both objects, 3 rotations, exclude_fixation on).

Rule fixed before running: if the median testable fraction over the runs is below 0.30, the
wider-window variants of 2.3 are built before any comparison with the literature; otherwise
Phase 2 proceeds on the current runs. PASS = proceed on current runs, FAIL = build 2.3.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from evaluate_runs import load_params  # noqa: E402
from exp_common import CENTRE, Run, set_device  # noqa: E402
from exp_predictive_field import rf_centres  # noqa: E402

ROT = [0, 3, 6]


def executed_amplitudes(path):
    run = Run(path, seed=0, exclude_fixation=True)
    period = run.params.saccade_period
    amps, before = [], {}

    def on_decision(d):
        before["pos"] = np.array(d["run"].env.retina_sim_pos, float)

    for world in ("triangle", "square"):
        for k in ROT:
            def on_step(d):
                if d["t"] % period == 0:
                    amps.append(np.linalg.norm(np.array(d["run"].env.retina_sim_pos, float) - before["pos"]))
            run.test(world, pos=list(CENTRE), rot=2 * np.pi * k / 9, on_decision=on_decision, on_step=on_step)
    return np.array(amps)


def evaluate(path, args):
    store = torch.load(os.path.join(path, "off_control_store"), map_location="cpu", weights_only=False)
    w = {k: store[f"{k}_map_state_dict"]["weights"].numpy().astype(float)
         for k in ["visual_effects", "attention"]}
    rng = np.random.RandomState(0)
    crf = rf_centres(w["visual_effects"], 3.0, 5000, rng, 0.0)
    d = (w["attention"].T - 0.5) * args.retina_scale * args.fovea / load_params(path).fovea_scale[0]
    half = (args.fovea - 1) / 2
    moving = np.linalg.norm(d, axis=1) > args.min_move
    inside = (np.abs(crf + d) <= half).all(1)
    u = rng.uniform(-half, half, (100000, 2))
    geo = (np.abs(u[:, None] + d[None]) <= half).all(2).mean(0)
    amp = np.linalg.norm(d, axis=1)
    ex = executed_amplitudes(path)
    return dict(run=os.path.basename(path), moving=moving.mean(), amp_med=np.median(amp), amp_q10=np.quantile(amp, .1),
                amp_q90=np.quantile(amp, .9), testable=inside.mean(), testable_moving=inside[moving].mean() if moving.any() else np.nan,
                geometric=geo.mean(), exec_med=np.median(ex), exec_q10=np.quantile(ex, .1), exec_q90=np.quantile(ex, .9),
                exec_inside=(ex <= 2 * half).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--retina-scale", type=float, default=80.0)
    ap.add_argument("--fovea", type=int, default=16)
    ap.add_argument("--min-move", type=float, default=2.0)
    args = ap.parse_args()
    set_device()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for r in csv.DictReader(open(args.manifest)):
        row = evaluate(r["run"], args)
        rows.append(row)
        print(f"{row['run']:34s} moving {row['moving']:.2f} |d| med {row['amp_med']:5.1f} testable {row['testable']:.2f} "
              f"geometric {row['geometric']:.2f} executed med {row['exec_med']:5.1f}", flush=True)
    with open(os.path.join(args.out, "feasibility.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    med = np.median([r["testable"] for r in rows])
    print(f"Median testable fraction {med:.2f} (threshold 0.30): {'PASS (proceed on current runs)' if med >= 0.30 else 'FAIL (build 2.3 wider-window variants)'}")


if __name__ == "__main__":
    main()
