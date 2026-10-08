"""Phase 2.2 (predictive_remapping.md): predictive-remapping paradigm at map level.

Usage:
    python scripts/exp_remapping_paradigm.py --verify
    python scripts/exp_remapping_paradigm.py --manifest FILE [--out DIR]

Column j = conditions unit j, effects unit j, attention prototype j. d(j) = (attention prototype
- 0.5) * retina_scale in retina pixels, converted to fovea pixels (x fovea_size / fovea_scale of
the run). Fields are in fovea pixels from the fovea centre.
  QF_j(p): soft response (as probe_retinotopy.activations) of effects unit j to a width-3 spot at
           p; CRF_j is its response-weighted centre over 2000 spots.
  PF_j(p): pre-saccadic response of column j through the model's read-out path: a Gaussian on the
           lattice (std neighborhood_modulation_baseline) around the conditions winner of the
           spot, evaluated at j; its centre is c_pre_j.
  FF_j = CRF_j + d(j) (forward-predicted future field); ST_j = d(j).
Testable cells: FF_j inside the fovea window. Computed on testable cells only.
P1: cell remaps if PF_j(FF_j) >= alpha * PF_j(c_pre_j) (alpha 0.5; the pre-saccadic response at the
    future field relative to the column's response at its own pre-saccadic centre, because the
    soft effects read-out and the lattice-kernel pre-saccadic read-out have different scales) and
    PF_j(FF_j) > PF_j(CRF_j).
    A field is read at a location as the average of the probe responses weighted by a Gaussian of
    width 1.5 px around it (the lattice is discrete and the centres are estimated).
P2 (trace) holds by construction (the effects activity is carried by the stored goal cell); it
    is not computed and is not evidence.
P3: remapping vector r_j = c_pre_j - CRF_j; FF type if the angle between r_j and d(j) is within
    20 deg, ST type if the angle is larger and c_pre_j is closer to d(j) than CRF_j is, else
    other. Toward if d(j) . CRF_j > 0, away otherwise (geometric stand-in for hemifield).
Controls per run: effects units permuted across cells (mean of --shuffles permutations);
untrained conditions and effects maps with the trained saccade vectors.

Summary over the runs of a manifest (C1, C2 of the plan; C3 is judged across manifests):
C1: median FF-type fraction (all testable cells) within [0.27, 0.82] and above the larger
    control mean by more than the SD of the FF-type fraction across runs.
C2: away minus toward FF-type fraction >= 0.15 and toward minus away ST-type fraction >= 0.15
    (medians over runs). A run needs >= 10 testable cells and >= 5 per direction to count.

--verify (rule fixed before running): synthetic aligned pair (effects unit j = spot at c_j -
d_j, with c_j the conditions centre, |d| 3.5 to 4.5 px so that FF and CRF are resolved by the lattice) must give P1 >= 0.9 and FF-type >= 0.9; the same pair with
the effects units permuted must give P1 < 0.3. PASS or FAIL per case.
"""
import argparse
import csv
import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from evaluate_runs import grid_coords, load_params, pairwise  # noqa: E402
from exp_retinotopy_runs import untrained  # noqa: E402
from probe_retinotopy import activations  # noqa: E402

SIDE = 16
HALF = (SIDE - 1) / 2
ALPHA = 0.5
ANGLE = np.deg2rad(20)
LD = pairwise(grid_coords(100).astype(float))


def spot_images(centres, amp, sigma=3.0):
    ax = np.arange(SIDE) - HALF
    gx = np.exp(-((ax[None] - centres[:, :1]) ** 2) / (2 * sigma**2))
    gy = np.exp(-((ax[None] - centres[:, 1:]) ** 2) / (2 * sigma**2))
    img = amp * gy[:, :, None] * gx[:, None, :]
    return np.repeat(img[..., None], 3, axis=3).reshape(len(centres), -1)


def kernel(std):
    k = np.exp(-0.5 * LD**2 / std**2)
    return k / k.sum(1, keepdims=True)


def pre_response(img, w_cond, k):
    d2 = (img**2).sum(1)[:, None] + (w_cond**2).sum(0)[None] - 2 * img @ w_cond
    return k[d2.argmin(1)]


def centre(resp, centres):
    return (resp.T @ centres) / resp.sum(0)[:, None]


def field_at(resp, probes, loc, bw=1.5):
    w = np.exp(-((probes[:, None] - loc[None]) ** 2).sum(2) / (2 * bw**2))
    return (w * resp).sum(0) / w.sum(0)


def paradigm(w_cond, w_eff, d, std, rng, n_probes=2000):
    n = w_cond.shape[1]
    k = kernel(std)
    probes = rng.uniform(-HALF, HALF, (n_probes, 2))
    amp_c, amp_e = np.percentile(w_cond, 99), np.percentile(w_eff, 99)
    qf = activations(spot_images(probes, amp_e), w_eff)["soft"]
    pf = pre_response(spot_images(probes, amp_c), w_cond, k)
    crf, cpre = centre(qf, probes), centre(pf, probes)
    ff = crf + d
    testable = (np.abs(ff) <= HALF).all(1)
    p_ff, p_crf, p_peak = field_at(pf, probes, ff), field_at(pf, probes, crf), field_at(pf, probes, cpre)
    remap = (p_ff >= ALPHA * p_peak) & (p_ff > p_crf)
    r = cpre - crf
    cos = (r * d).sum(1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(d, axis=1) + 1e-12)
    ff_type = np.arccos(np.clip(cos, -1, 1)) <= ANGLE
    st_type = ~ff_type & (np.linalg.norm(cpre - d, axis=1) < np.linalg.norm(crf - d, axis=1))
    toward = (d * crf).sum(1) > 0
    return dict(testable=testable, remap=remap, ff_type=ff_type, st_type=st_type, toward=toward)


def summarise(res):
    t = res["testable"]
    out = dict(n_testable=int(t.sum()), frac_testable=t.mean())
    ok = t.sum() >= 10
    out["p1"] = res["remap"][t].mean() if ok else np.nan
    out["ff"] = res["ff_type"][t].mean() if ok else np.nan
    out["st"] = res["st_type"][t].mean() if ok else np.nan
    for name, sel in [("toward", res["toward"]), ("away", ~res["toward"])]:
        m = t & sel
        out[f"n_{name}"] = int(m.sum())
        for key in ("ff_type", "st_type"):
            out[f"{key[:2]}_{name}"] = res[key][m].mean() if m.sum() >= 5 else np.nan
    return out


def control_mean(rows):
    return {k: np.nanmean([r[k] for r in rows]) if not all(np.isnan(r[k]) for r in rows) else np.nan
            for k in rows[0]}


def run_row(path, args):
    store = torch.load(os.path.join(path, "off_control_store"), map_location="cpu", weights_only=False)
    w = {k: store[f"{k}_map_state_dict"]["weights"].numpy().astype(float)
         for k in ["visual_conditions", "visual_effects", "attention"]}
    p = load_params(path)
    d = (w["attention"].T - 0.5) * args.retina_scale * SIDE / p.fovea_scale[0]
    std = p.neighborhood_modulation_baseline
    rng = np.random.RandomState(0)
    row = dict(run=os.path.basename(path), kind="trained")
    row.update(summarise(paradigm(w["visual_conditions"], w["visual_effects"], d, std, rng)))
    ctrl = [summarise(paradigm(w["visual_conditions"], w["visual_effects"][:, rng.permutation(100)], d, std, rng))
            for _ in range(args.shuffles)]
    shuf = dict(run=row["run"], kind="shuffled")
    shuf.update(control_mean(ctrl))
    wu = [untrained(s, w[k].shape) for s, k in [(1, "visual_conditions"), (2, "visual_effects")]]
    unt = dict(run=row["run"], kind="untrained")
    unt.update(summarise(paradigm(wu[0], wu[1], d, std, rng)))
    return [row, shuf, unt]


def synthetic(rng, permute):
    g = np.linspace(-3.5, 3.5, 10)
    c = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(100, 2)[:, ::-1]
    ang = rng.uniform(0, 2 * np.pi, 100)
    d = rng.uniform(3.5, 4.5, (100, 1)) * np.stack([np.cos(ang), np.sin(ang)], 1)
    amp = 1.0
    w_cond = spot_images(c, amp).T
    w_eff = spot_images(c - d, amp).T
    if permute:
        w_eff = w_eff[:, rng.permutation(100)]
    return summarise(paradigm(w_cond, w_eff, d, 0.5, rng))


def verify():
    rng = np.random.RandomState(0)
    fails = 0
    for name, permute in [("aligned", False), ("shuffled pairing", True)]:
        s = synthetic(rng, permute)
        ok = (s["p1"] >= 0.9 and s["ff"] >= 0.9) if not permute else s["p1"] < 0.3
        fails += not ok
        print(f"{name:17s} testable {s['n_testable']:3d} P1 {s['p1']:.2f} FF-type {s['ff']:.2f} ST-type {s['st']:.2f} "
              f"{'PASS' if ok else 'FAIL'}")
    print("VERIFY", "PASS" if not fails else "FAIL")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--manifest")
    ap.add_argument("--out", default=".")
    ap.add_argument("--retina-scale", type=float, default=80.0)
    ap.add_argument("--shuffles", type=int, default=50)
    args = ap.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)
    if args.verify:
        return verify()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for r in csv.DictReader(open(args.manifest)):
        for row in run_row(r["run"], args):
            rows.append(row)
            print(f"{row['run']:30s} {row['kind']:9s} testable {row['n_testable']:5.1f} P1 {row['p1']:.2f} "
                  f"FF {row['ff']:.2f} ST {row['st']:.2f} | away FF {row['ff_away']:.2f} ST {row['st_away']:.2f} "
                  f"| toward FF {row['ff_toward']:.2f} ST {row['st_toward']:.2f}", flush=True)
    with open(os.path.join(args.out, "remapping_" + os.path.splitext(os.path.basename(args.manifest))[0] + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tr = [r for r in rows if r["kind"] == "trained"]
    ff = np.array([r["ff"] for r in tr])
    ctrl = max(np.nanmedian([r["ff"] for r in rows if r["kind"] == k]) for k in ("shuffled", "untrained"))
    med = np.nanmedian(ff)
    c1 = 0.27 <= med <= 0.82 and med - ctrl > np.nanstd(ff)
    print(f"C1: median FF-type {med:.2f}, larger control {ctrl:.2f}, SD {np.nanstd(ff):.2f}: {'PASS' if c1 else 'FAIL'}")
    d_ff = np.nanmedian([r["ff_away"] - r["ff_toward"] for r in tr])
    d_st = np.nanmedian([r["st_toward"] - r["st_away"] for r in tr])
    print(f"C2: away-toward FF {d_ff:.2f}, toward-away ST {d_st:.2f}: {'PASS' if d_ff >= 0.15 and d_st >= 0.15 else 'FAIL'}")


if __name__ == "__main__":
    main()
