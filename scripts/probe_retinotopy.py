"""Probe trained maps with Gaussian spots to measure how retinotopic they are."""
import argparse
import os
import sys
import warnings

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from evaluate_runs import MAPS, find_runs, grid_coords, pairwise  # noqa: E402


def spots(side, sigma, n, amp, rng):
    centres = rng.uniform(0, side - 1, (n, 2))
    ax = np.arange(side)
    gx = np.exp(-((ax[None] - centres[:, :1]) ** 2) / (2 * sigma**2))
    gy = np.exp(-((ax[None] - centres[:, 1:]) ** 2) / (2 * sigma**2))
    img = amp * gy[:, :, None] * gx[:, None, :]
    return np.repeat(img[..., None], 3, axis=3).reshape(n, -1), centres


def activations(x, weights):
    d2 = ((x**2).sum(1)[:, None] + (weights**2).sum(0)[None] - 2 * x @ weights).clip(0)
    hard = np.zeros_like(d2)
    hard[np.arange(len(d2)), d2.argmin(1)] = 1
    z = -d2 / (2 * np.median(d2.min(1)))
    soft = np.exp(z - z.max(1, keepdims=True))
    return {"hard": hard, "soft": soft / soft.sum(1, keepdims=True)}


def smoothness(coords, points, perms, rng):
    if len(points) < 3:
        return np.nan, np.nan
    iu = np.triu_indices(len(points), 1)
    grid = pairwise(coords.astype(float))[iu]
    rho = spearmanr(grid, pairwise(points)[iu])[0]
    null = [spearmanr(grid, pairwise(points[rng.permutation(len(points))])[iu])[0]
            for _ in range(perms)]
    return rho, (rho - np.mean(null)) / np.std(null)


def evaluate(run, args):
    store = torch.load(os.path.join(run, "off_control_store"), map_location="cpu",
                       weights_only=False)
    rows = []
    for name, short in MAPS.items():
        weights = store[f"{name}_map_state_dict"]["weights"].numpy().astype(float)
        coords = grid_coords(weights.shape[1])
        rng = np.random.RandomState(args.seed)
        if name == "attention":
            rho, z = smoothness(coords, weights.T, args.perms, rng)
            extent = float((weights.max(1) - weights.min(1)).mean())
            rows.append(dict(run=run, map=short, sigma=np.nan, readout="prototype", rho=rho,
                             z=z, units=weights.shape[1], range=extent, pr=np.nan))
            continue
        side = int(round(np.sqrt(weights.shape[0] / 3)))
        amp = args.amp if args.amp else np.percentile(weights, 99)
        for sigma in args.sigmas:
            x, centres = spots(side, sigma, args.n_probes, amp, rng)
            layout = rng.permutation(side * side)
            shuffled = x.reshape(len(x), side * side, 3)[:, layout].reshape(len(x), -1)
            acts = activations(x, weights)
            acts.update({f"{k}-ctl": v for k, v in activations(shuffled, weights).items()})
            for readout, a in acts.items():
                mass = a.sum(0)
                valid = mass > 0
                rf = (a.T @ centres)[valid] / mass[valid, None]
                rho, z = smoothness(coords[valid], rf, args.perms, rng)
                rows.append(dict(
                    run=run, map=short, sigma=sigma, readout=readout, rho=rho, z=z,
                    units=int(valid.sum()),
                    range=float(((rf.max(0) - rf.min(0)) / (side - 1)).mean()),
                    pr=float((a.sum(1) ** 2 / (a**2).sum(1)).mean()),
                ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=(
        "Probe the maps of trained runs with Gaussian spots at known positions in "
        "the fovea. For each unit the receptive-field centre is the "
        "activation-weighted mean spot position; rho is the Spearman correlation "
        "between grid distance and receptive-field distance over unit pairs, z its "
        "permutation z-score. range is the spread of receptive-field centres as a "
        "fraction of the fovea; pr the participation ratio. The attention map is "
        "a positive control: its prototypes are retinal coordinates. Readouts "
        "ending in -ctl use the same spots with the pixel layout shuffled "
        "(fixed permutation): if rho survives there, it reflects prototype-space "
        "topography, not spatial order. Reference (SOM note, shape-trained): hard "
        "rho 0.18, soft 0.49-0.51."))
    parser.add_argument("paths", nargs="+", help="Run folders, or folders containing them.")
    parser.add_argument("--sigmas", type=float, nargs="+", default=[1, 2, 3, 4],
                        help="Spot widths in fovea pixels.")
    parser.add_argument("--n-probes", type=int, default=5000)
    parser.add_argument("--amp", type=float,
                        help="Spot amplitude (default: 99th percentile of the map's weights).")
    parser.add_argument("--perms", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", help="Write all rows to this CSV file.")
    args = parser.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)

    runs = find_runs(args.paths)
    if not runs:
        sys.exit("No run folders found.")
    rows = []
    for run in runs:
        print(f"probing {run}", file=sys.stderr)
        rows += evaluate(run, args)

    if args.csv:
        columns = list(rows[0])
        with open(args.csv, "w") as f:
            f.write(",".join(columns) + "\n")
            for r in rows:
                f.write(",".join(str(r[c]) for c in columns) + "\n")

    keys = list(dict.fromkeys((r["map"], r["sigma"], r["readout"]) for r in rows))
    print(f"{len(runs)} run(s); mean over runs (std when more than one)")
    print(f"{'map':4s} {'sigma':>5s} {'readout':>9s} {'rho':>14s} {'z':>8s} "
          f"{'units':>6s} {'range':>6s} {'pr':>6s} {'rho-ctl':>8s}")
    for key in keys:
        sel = [r for r in rows if (r["map"], r["sigma"], r["readout"]) == key]
        rho = np.array([r["rho"] for r in sel])
        mean = lambda c: np.nanmean([r[c] for r in sel])  # noqa: E731
        ctl = [r["rho"] for r in rows
               if (r["map"], r["sigma"], r["readout"]) == (key[0], key[1], key[2] + "-ctl")]
        gain = rho.mean() - np.mean(ctl) if ctl and not key[2].endswith("-ctl") else np.nan
        print(f"{key[0]:4s} {key[1]:5.1f} {key[2]:>9s} {rho.mean():7.3f}+-{rho.std():5.3f} "
              f"{mean('z'):8.1f} {mean('units'):6.0f} {mean('range'):6.2f} {mean('pr'):6.1f} "
              f"{gain:8.3f}")


if __name__ == "__main__":
    main()
