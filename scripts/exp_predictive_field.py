"""E3: does the pairing of conditions and effects units look like predictive (future-field) remapping?"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from evaluate_runs import find_runs  # noqa: E402
from probe_retinotopy import activations, spots  # noqa: E402


def rf_centres(weights, sigma, n_probes, rng, peak):
    side = int(round(np.sqrt(weights.shape[0] / 3)))
    x, centres = spots(side, sigma, n_probes, np.percentile(weights, 99), rng)
    a = activations(x, weights)["soft"]
    if peak:
        a = np.where(a >= peak * a.max(0), a, 0)
    return (a.T @ centres) / a.sum(0)[:, None] - (side - 1) / 2


def cosine(a, b):
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    ok = den > 1e-9
    return num[ok] / den[ok]


def statistics(c_cond, c_eff, d, min_move):
    moves = np.linalg.norm(d, axis=1) > min_move
    dd = c_cond - c_eff
    fwd = cosine(dd[moves], d[moves]).mean()
    conv = cosine((dd - d)[moves], -c_eff[moves]).mean()
    return fwd, conv


def evaluate(path, args):
    store = torch.load(os.path.join(path, "off_control_store"), map_location="cpu",
                       weights_only=False)
    w = {k: store[f"{k}_map_state_dict"]["weights"].numpy().astype(float)
         for k in ["visual_conditions", "visual_effects", "attention"]}
    rng = np.random.RandomState(args.seed)
    c_cond = rf_centres(w["visual_conditions"], args.sigma, args.n_probes, rng, args.peak)
    c_eff = rf_centres(w["visual_effects"], args.sigma, args.n_probes, rng, args.peak)
    d = (w["attention"].T - 0.5) * args.retina_scale
    fwd, conv = statistics(c_cond, c_eff, d, args.min_move)
    null = np.array([statistics(c_cond, c_eff[rng.permutation(len(c_eff))], d, args.min_move)
                     for _ in range(args.perms)])
    dd = c_cond - c_eff
    slope = (dd * d).sum() / (d * d).sum()
    return dict(
        run=os.path.basename(path), fwd=fwd, fwd_null95=np.percentile(null[:, 0], 95),
        fwd_p=(1 + (null[:, 0] >= fwd).sum()) / (1 + len(null)), slope=slope,
        conv=conv, conv_null95=np.percentile(null[:, 1], 95),
        conv_p=(1 + (null[:, 1] >= conv).sum()) / (1 + len(null)),
        moving=int((np.linalg.norm(d, axis=1) > args.min_move).sum()),
    )


def main():
    parser = argparse.ArgumentParser(description=(
        "For every lattice cell j the conditions unit j and the effects unit j are probed with "
        "Gaussian spots (soft readout) to get the field c_cond(j) before and c_eff(j) after the "
        "saccade, in fovea pixels from the fovea centre; d(j) is the saccade of the attention "
        "prototype of cell j, (s - 0.5) * retina_scale, one fovea pixel per retina pixel. If the "
        "pairing tracks one stimulus across the saccade (forward remapping), c_cond - c_eff = d. "
        "fwd is the mean cosine between c_cond - c_eff and d over cells with |d| > min_move; "
        "its null shuffles the effects units across cells. conv (exploratory, sensitive to the "
        "compression of soft readout, not in the rule) is the mean cosine between the residual "
        "c_cond - c_eff - d and -c_eff, positive if the fields shift toward the target. Rule "
        "fixed before running: a forward-remapping-like relation is present if fwd is positive "
        "and above the 95th percentile of its null in at least 4 of 5 runs. The first plan "
        "counted 8 of 10 maps; one value per run is computed, so the count is 4 of 5."))
    parser.add_argument("paths", nargs="+", help="Run folders, or folders containing them.")
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--n-probes", type=int, default=5000)
    parser.add_argument("--perms", type=int, default=1000)
    parser.add_argument("--min-move", type=float, default=2.0, help="Fovea pixels.")
    parser.add_argument("--retina-scale", type=float, default=80.0)
    parser.add_argument("--peak", type=float, default=0.0, help=(
        "Exploratory: use only activations above this fraction of a unit's maximum, which "
        "reduces the compression of the receptive-field centres (0 = the pre-stated mean)."))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv")
    args = parser.parse_args()

    runs = find_runs(args.paths)
    if not runs:
        sys.exit("No run folders found.")
    rows = [evaluate(p, args) for p in runs]
    print(f"{'run':36s} {'fwd':>6s} {'null95':>7s} {'p':>6s} {'slope':>6s} {'conv':>6s} "
          f"{'null95':>7s} {'p':>6s} {'cells':>5s}")
    for r in rows:
        print(f"{r['run'][-36:]:36s} {r['fwd']:6.2f} {r['fwd_null95']:7.2f} {r['fwd_p']:6.3f} "
              f"{r['slope']:6.2f} {r['conv']:6.2f} {r['conv_null95']:7.2f} {r['conv_p']:6.3f} "
              f"{r['moving']:5d}")
    hit = [r["fwd"] > 0 and r["fwd"] > r["fwd_null95"] for r in rows]
    need = int(np.ceil(0.8 * len(rows)))
    print(f"Rule (forward remapping-like relation): {sum(hit)}/{len(rows)} runs above the null; "
          f"{'PASS' if sum(hit) >= need else 'FAIL'}")
    chit = [r["conv"] > 0 and r["conv"] > r["conv_null95"] for r in rows]
    print(f"Exploratory (convergent): {sum(chit)}/{len(rows)} runs above the null")
    if args.csv:
        with open(args.csv, "w") as f:
            f.write(",".join(rows[0]) + "\n")
            for r in rows:
                f.write(",".join(str(v) for v in r.values()) + "\n")


if __name__ == "__main__":
    main()
