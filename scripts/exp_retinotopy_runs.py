"""Phase 1 (predictive_remapping.md 1.2, 1.3, 1.5): ground-truth organisation,
black-box spot probe and classification of the visual maps of trained runs.

Usage:
    python scripts/exp_retinotopy_runs.py --verify
    python scripts/exp_retinotopy_runs.py --manifest MANIFEST --foveas DIR --out DIR

Per run and visual map (conditions vc, effects ve), for three kinds of
weights: the trained map, an untrained map (the controller's initial
weights, seeded by the run seed) and a batch SOM trained without anchor on
the run's experienced foveas (the input route alone).

Ground truth: each grid-set fovea (blank ones dropped) is assigned to its
winner; over units that win at least --min-count foveas, rho_off, rho_rot,
rho_world are Spearman correlations between lattice distance and the
distance between unit-mean labels (offset; rotation as (cos, sin); world).
Null: labels shuffled over foveas (--perms), reported as z. The same is
computed on the experienced set (suffix _exp).

Probe: spot width 3, soft readout, 2000 spots; z by unit permutation, gain
and p against --shuffles shuffled pixel layouts (as blackbox() in
indirect_retinotopy_test.py). Widths 2 and 4 and the hard readout are
reported (rho_s2, rho_s4, rho_hard) but not used for the class.

Classes (margin 0.1; rho_other = max(rho_rot, rho_world)):
    proper      rho_off - rho_other > 0.1 and gain p < 0.05
    indirect    z > 3, gain p < 0.05 and rho_other - rho_off > 0.1
    topography  z > 3 and gain p >= 0.05
    none        otherwise (z > 3, p < 0.05 and |rho_off - rho_other| <= 0.1
                is none by this rule; flagged mixed)

--verify (rules before the manifest run): synthetic position-only SOM must
be proper, the global-shapes SOM topography or indirect, untrained maps
none; PASS or FAIL per case. For synthetic maps rho_other is the larger of
the width and colour rho.
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
from evaluate_runs import grid_coords, load_params, pairwise  # noqa: E402
from exp_labelled_foveas import setup_hash  # noqa: E402
from indirect_retinotopy_test import CASES, rectangles, train_som  # noqa: E402
from probe_retinotopy import activations, spots  # noqa: E402

SIDE = 16
MARGIN = 0.1
LD = pairwise(grid_coords(100).astype(float))
IU = np.triu_indices(100, 1)


def winners(x, weights):
    d2 = (x**2).sum(1)[:, None] + (weights**2).sum(0)[None] - 2 * x @ weights
    return d2.argmin(1)


def organisation(win, feat, min_count):
    counts = np.bincount(win, minlength=100)
    units = np.where(counts >= min_count)[0]
    if len(units) < 10:
        return np.nan
    means = np.stack([feat[win == u].mean(0) for u in units])
    iu = np.triu_indices(len(units), 1)
    return spearmanr(LD[np.ix_(units, units)][iu], pairwise(means)[iu])[0]


def ground_truth(win, feats, min_count, perms, rng):
    out = {}
    for name, feat in feats.items():
        rho = organisation(win, feat, min_count)
        null = np.array([organisation(win, feat[rng.permutation(len(feat))], min_count)
                         for _ in range(perms)])
        out[f"rho_{name}"] = rho
        out[f"z_{name}"] = (rho - np.nanmean(null)) / np.nanstd(null)
    return out


def labels_to_feats(labels):
    return {"off": labels["offset"],
            "rot": np.stack([np.cos(labels["rot"]), np.sin(labels["rot"])], 1),
            "world": labels["world"][:, None].astype(float)}


def probe(weights, rng, sigma, readout, shuffles, n_probes):
    amp = np.percentile(weights, 99)
    x, centres = spots(SIDE, sigma, n_probes, amp, rng)

    def rho_of(inp):
        a = activations(inp, weights)[readout]
        valid = a.sum(0) > 0
        rf = (a.T @ centres)[valid] / a.sum(0)[valid, None]
        if valid.sum() < 10:
            return np.nan, rf, a, valid
        iu = np.triu_indices(valid.sum(), 1)
        return spearmanr(LD[np.ix_(valid, valid)][iu], pairwise(rf)[iu])[0], rf, a, valid

    rho, rf, a, valid = rho_of(x)
    if shuffles == 0:
        return dict(rho=rho)
    null = np.array([rho_of(x.reshape(len(x), SIDE * SIDE, 3)[:, rng.permutation(SIDE * SIDE)].reshape(len(x), -1))[0]
                     for _ in range(shuffles)])
    iu = np.triu_indices(valid.sum(), 1)
    grid = LD[np.ix_(valid, valid)][iu]
    zu = [spearmanr(grid, pairwise(rf[rng.permutation(len(rf))])[iu])[0] for _ in range(100)]
    return dict(rho=rho, z=(rho - np.mean(zu)) / np.std(zu), gain=rho - np.nanmean(null),
                p=(1 + (null >= rho).sum()) / (len(null) + 1))


def classify(rho_off, rho_other, z, p):
    if rho_off - rho_other > MARGIN and p < 0.05:
        return "proper"
    if z > 3 and p < 0.05 and rho_other - rho_off > MARGIN:
        return "indirect"
    if z > 3 and p >= 0.05:
        return "topography"
    return "none"


def probe_row(weights, rng, args):
    main = probe(weights, rng, 3.0, "soft", args.shuffles, args.n_probes)
    return dict(probe_rho=main["rho"], probe_z=main["z"], probe_gain=main["gain"], probe_p=main["p"],
                rho_s2=probe(weights, rng, 2.0, "soft", 0, args.n_probes)["rho"],
                rho_s4=probe(weights, rng, 4.0, "soft", 0, args.n_probes)["rho"],
                rho_hard=probe(weights, rng, 3.0, "hard", 0, args.n_probes)["rho"])


def untrained(seed, shape):
    torch.manual_seed(seed)
    w = torch.empty(*shape)
    torch.nn.init.xavier_normal_(w)
    return 1e-4 * w.numpy().astype(float)


def evaluate_weights(weights, grid, exp, args, rng):
    row = {}
    keep = grid["foveas"].max(1) > 0
    gf = grid["foveas"][keep].astype(float)
    glabels = {k: grid[k][keep] for k in ("offset", "rot", "world")}
    row.update(ground_truth(winners(gf, weights), labels_to_feats(glabels), args.min_count, args.perms, rng))
    ef = exp["foveas"].astype(float)
    elabels = {k: exp[k] for k in ("offset", "rot", "world")}
    row.update({k + "_exp": v for k, v in ground_truth(
        winners(ef, weights), labels_to_feats(elabels), args.min_count, args.perms, rng).items()})
    row.update(probe_row(weights, rng, args))
    other = np.nanmax([row["rho_rot"], row["rho_world"]])
    row["class"] = classify(row["rho_off"], other, row["probe_z"], row["probe_p"])
    row["mixed"] = bool(row["probe_z"] > 3 and row["probe_p"] < 0.05 and abs(row["rho_off"] - other) <= MARGIN)
    return row


def other_rho(win, lab, min_count):
    return max(np.nan_to_num(organisation(win, lab[k], min_count), nan=0.0) for k in ("width", "colour"))


def verify(args):
    rng = np.random.RandomState(0)
    fails = 0
    expected = {"position only": {"proper"}, "global shapes": {"topography", "indirect"}}
    for label, spread, content in CASES:
        x, lab = rectangles(args.n_train, spread, content, rng, labels=True)
        weights = train_som(x, rng)
        win = winners(x, weights)
        rho_off = organisation(win, lab["centre"], args.min_count)
        rho_other = other_rho(win, lab, args.min_count)
        p = probe(weights, rng, 3.0, "soft", args.shuffles, args.n_probes)
        cls = classify(rho_off, rho_other, p["z"], p["p"])
        ok = cls in expected[label]
        fails += not ok
        print(f"{label:14s} rho_off {rho_off:5.2f} rho_other {rho_other:5.2f} z {p['z']:5.1f} p {p['p']:.3f} "
              f"-> {cls} (expected {'/'.join(sorted(expected[label]))}) {'PASS' if ok else 'FAIL'}")
    x, lab = rectangles(args.n_train, 1.0, 1.5, rng, labels=True)
    weights = untrained(0, (SIDE * SIDE * 3, 100))
    win = winners(x, weights)
    rho_off = organisation(win, lab["centre"], args.min_count)
    rho_other = other_rho(win, lab, args.min_count)
    p = probe(weights, rng, 3.0, "soft", args.shuffles, args.n_probes)
    cls = classify(rho_off, rho_other, p["z"], p["p"])
    ok = cls == "none"
    fails += not ok
    print(f"{'untrained':14s} rho_off {rho_off:5.2f} rho_other {rho_other:5.2f} z {p['z']:5.1f} p {p['p']:.3f} "
          f"-> {cls} (expected none) {'PASS' if ok else 'FAIL'}")
    print("VERIFY", "PASS" if not fails else "FAIL")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--manifest")
    ap.add_argument("--foveas")
    ap.add_argument("--out", default=".")
    ap.add_argument("--run", action="append", default=[], help="run folder (repeatable), instead of the manifest")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--perms", type=int, default=50)
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
        path = r["run"]
        h = setup_hash(load_params(path))
        grid = np.load(os.path.join(args.foveas, f"foveas_grid_{h}.npz"))
        exp = np.load(os.path.join(args.foveas, f"foveas_exp_{os.path.basename(path)}.npz"))
        store = torch.load(os.path.join(path, "off_control_store"), map_location="cpu", weights_only=False)
        seed = int(r["seed"]) if str(r["seed"]).isdigit() else 0
        for name, short in [("visual_conditions", "vc"), ("visual_effects", "ve")]:
            trained = store[f"{name}_map_state_dict"]["weights"].numpy().astype(float)
            rng = np.random.RandomState(args.seed)
            som = train_som(exp["foveas"].astype(float), rng)
            for kind, weights in [("trained", trained), ("untrained", untrained(seed, trained.shape)), ("som", som)]:
                row = dict(sweep=r["sweep"], run=os.path.basename(path), seed=r["seed"], map=short, kind=kind)
                row.update(evaluate_weights(weights, grid, exp, args, rng))
                rows.append(row)
                print(f"{row['run']:34s} {short} {kind:9s} off {row['rho_off']:5.2f} rot {row['rho_rot']:5.2f} "
                      f"world {row['rho_world']:5.2f} z {row['probe_z']:5.1f} p {row['probe_p']:.3f} {row['class']}",
                      flush=True)
    with open(os.path.join(args.out, "retinotopy_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
