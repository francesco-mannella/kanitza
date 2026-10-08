"""Phase-encoded retinotopic mapping of the visual maps (predictive_remapping.md 1.3).

Usage:
    python scripts/probe_phase_encoded.py --verify
    python scripts/probe_phase_encoded.py (--manifest FILE | --run DIR ...) [--out DIR]

As in phase-encoded fMRI mapping, a bar sweeps the 16x16 fovea horizontally
and vertically and a ring expands from the centre, each for --cycles cycles
of --frames frames (the stimulus runs from outside the fovea on one side to
outside on the other, so every cycle has a blank stretch). Each unit's soft
response (as in probe_retinotopy.activations) is Fourier transformed over
time; the phase at the cycle frequency gives its preferred x (horizontal
sweep), y (vertical sweep) and eccentricity (ring).

Per map: rho_phase = Spearman correlation between lattice distance and the
distance between phase-derived (x, y) over valid units (non-zero response
variance in both sweeps); z by unit permutation; gain and p against
--shuffles shuffled pixel layouts of the stimuli; rho_coh is rho_phase over
units with coherence >= 0.3 in both sweeps (coherence = amplitude at the
cycle frequency over the root sum of squared non-DC amplitudes); r_spot_x and
r_spot_y are Spearman correlations between phase-derived and spot-probe
(width 3) RF centres over units; r_ecc is the Spearman correlation between
the ring eccentricity and the eccentricity of the (x, y) centre from the bars.

Rule for --verify (fixed before running): the position-only SOM must give
rho_phase > 0.5, p < 0.05 and r_spot_x, r_spot_y > 0.5; the untrained map
must give rho_phase < 0.2 or p >= 0.05. PASS or FAIL per case.
"""
import argparse
import csv
import os
import sys
import warnings

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from exp_retinotopy_runs import LD, SIDE, untrained  # noqa: E402
from evaluate_runs import pairwise  # noqa: E402
from indirect_retinotopy_test import CASES, rectangles, train_som  # noqa: E402
from probe_retinotopy import activations, spots  # noqa: E402

COH_MIN = 0.3
BAR_SIGMA = 1.5
AX = np.arange(SIDE)


def sweep(kind, frames, cycles):
    centre = (SIDE - 1) / 2
    lo, hi = (-3, SIDE + 2) if kind != "ring" else (-2, np.hypot(centre, centre) + 2)
    pos = np.tile(lo + (hi - lo) * np.arange(frames) / frames, cycles)
    if kind == "ring":
        d = np.hypot(AX[None, :, None] - centre, AX[None, None, :] - centre)
        img = np.exp(-((d - pos[:, None, None]) ** 2) / (2 * BAR_SIGMA**2))
    else:
        g = np.exp(-((AX[None] - pos[:, None]) ** 2) / (2 * BAR_SIGMA**2))
        img = g[:, None, :] * np.ones((1, SIDE, 1)) if kind == "x" else g[:, :, None] * np.ones((1, 1, SIDE))
    return img, lo + (hi - lo) * np.arange(frames) / frames


def to_input(img, amp, layout=None):
    x = np.repeat(amp * img[..., None], 3, axis=3).reshape(len(img), SIDE * SIDE, 3)
    if layout is not None:
        x = x[:, layout]
    return x.reshape(len(img), -1)


def phase_position(resp, frames, cycles, axis_pos):
    f = np.fft.rfft(resp - resp.mean(0), axis=0)
    amp = np.abs(f)
    theta = (-np.angle(f[cycles])) % (2 * np.pi)
    coh = amp[cycles] / np.sqrt((amp[1:] ** 2).sum(0) + 1e-30)
    t = theta / (2 * np.pi) * frames
    return np.interp(t, np.arange(frames + 1), np.append(axis_pos, 2 * axis_pos[-1] - axis_pos[-2])), coh


def maps(weights, args, layout=None):
    amp = np.percentile(weights, 99)
    out = {}
    for kind in ("x", "y", "ring"):
        img, axis_pos = sweep(kind, args.frames, args.cycles)
        resp = activations(to_input(img, amp, layout), weights)["soft"]
        out[kind] = phase_position(resp, args.frames, args.cycles, axis_pos) + (resp.std(0) > 1e-12,)
    return out


def rho_of(m):
    valid = m["x"][2] & m["y"][2]
    if valid.sum() < 10:
        return np.nan, None, valid
    xy = np.stack([m["x"][0], m["y"][0]], 1)[valid]
    iu = np.triu_indices(valid.sum(), 1)
    return spearmanr(LD[np.ix_(valid, valid)][iu], pairwise(xy)[iu])[0], xy, valid


def evaluate(weights, args, rng):
    m = maps(weights, args)
    rho, xy, valid = rho_of(m)
    if xy is None:
        return dict(rho_phase=np.nan)
    iu = np.triu_indices(valid.sum(), 1)
    grid = LD[np.ix_(valid, valid)][iu]
    zu = [spearmanr(grid, pairwise(xy[rng.permutation(len(xy))])[iu])[0] for _ in range(100)]
    null = np.array([rho_of(maps(weights, args, rng.permutation(SIDE * SIDE)))[0] for _ in range(args.shuffles)])
    row = dict(rho_phase=rho, z=(rho - np.mean(zu)) / np.std(zu), gain=rho - np.nanmean(null),
               p=(1 + (null >= rho).sum()) / (len(null) + 1))
    coh = (m["x"][1] >= COH_MIN) & (m["y"][1] >= COH_MIN) & valid
    if coh.sum() >= 10:
        xyc = np.stack([m["x"][0], m["y"][0]], 1)[coh]
        ic = np.triu_indices(coh.sum(), 1)
        row["rho_coh"] = spearmanr(LD[np.ix_(coh, coh)][ic], pairwise(xyc)[ic])[0]
    else:
        row["rho_coh"] = np.nan
    row["coh_units"] = int(coh.sum())
    s, centres = spots(SIDE, 3.0, args.n_probes, np.percentile(weights, 99), rng)
    a = activations(s, weights)["soft"]
    rf = (a.T @ centres) / a.sum(0)[:, None]
    row["r_spot_x"] = spearmanr(m["x"][0][valid], rf[valid, 0])[0]
    row["r_spot_y"] = spearmanr(m["y"][0][valid], rf[valid, 1])[0]
    centre = (SIDE - 1) / 2
    ecc = np.hypot(m["x"][0] - centre, m["y"][0] - centre)
    ok = valid & m["ring"][2]
    row["r_ecc"] = spearmanr(ecc[ok], m["ring"][0][ok])[0] if ok.sum() > 3 else np.nan
    return row


def verify(args):
    rng = np.random.RandomState(0)
    fails = 0
    cases = []
    x = rectangles(args.n_train, *CASES[0][1:], rng)
    cases.append(("position only", train_som(x, rng)))
    x = rectangles(args.n_train, *CASES[1][1:], rng)
    cases.append(("global shapes", train_som(x, rng)))
    cases.append(("untrained", untrained(0, (SIDE * SIDE * 3, 100))))
    for name, w in cases:
        r = evaluate(w, args, rng)
        if name == "position only":
            ok = r["rho_phase"] > 0.5 and r["p"] < 0.05 and r["r_spot_x"] > 0.5 and r["r_spot_y"] > 0.5
        elif name == "untrained":
            ok = r["rho_phase"] < 0.2 or r["p"] >= 0.05
        else:
            ok = True
        fails += not ok
        print(f"{name:14s} rho {r['rho_phase']:5.2f} z {r['z']:5.1f} p {r['p']:.3f} rho_coh {r['rho_coh']:5.2f} "
              f"r_spot_x {r['r_spot_x']:5.2f} r_spot_y {r['r_spot_y']:5.2f} r_ecc {r['r_ecc']:5.2f} "
              f"{'PASS' if ok else 'FAIL'}" + ("" if name != "global shapes" else " (reported, no rule)"))
    print("VERIFY", "PASS" if not fails else "FAIL")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--manifest")
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--out", default=".")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--shuffles", type=int, default=100)
    ap.add_argument("--n-probes", type=int, default=2000)
    ap.add_argument("--n-train", type=int, default=3000)
    args = ap.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)
    if args.verify:
        return verify(args)
    runs = [dict(run=r, sweep="", seed=0) for r in args.run]
    if args.manifest:
        runs += list(csv.DictReader(open(args.manifest)))
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for r in runs:
        store = torch.load(os.path.join(r["run"], "off_control_store"), map_location="cpu", weights_only=False)
        for name, short in [("visual_conditions", "vc"), ("visual_effects", "ve")]:
            w = store[f"{name}_map_state_dict"]["weights"].numpy().astype(float)
            row = dict(sweep=r["sweep"], run=os.path.basename(r["run"]), seed=r["seed"], map=short)
            row.update(evaluate(w, args, np.random.RandomState(0)))
            rows.append(row)
            print(f"{row['run']:34s} {short} rho {row['rho_phase']:5.2f} p {row['p']:.3f} "
                  f"r_spot_x {row['r_spot_x']:5.2f} r_spot_y {row['r_spot_y']:5.2f}", flush=True)
    with open(os.path.join(args.out, "phase_encoded.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
