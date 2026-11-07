"""Three-point remapping paradigm, contingency version: F, T and S as spots, soft competition on the conditions face.

Usage:
    python scripts/exp_contingency.py --manifest FILE [--out DIR] [--tau 0.01 --tau 0.03] [--perms N]

Besides the csv and the viewer json, contingency_maps_<run>.npz holds the response maps of the 100 units on the conditions (Rc) and
effects (Re) faces over the 196 places (places), for the figures of fig_three_spot.py.

Each unit j is a contingency: its conditions prototype is the view at the fovea before the saccade, its effects prototype the
view at the fovea after it. F is the retina centre (centre of the conditions prototype). T = v_j, the saccade vector of its
attention prototype, is where the effects prototype is centred in the pre-saccadic retina. S is a point of the shape seen in both
prototypes: the maximum of (conditions prototype at S) x (effects prototype at S - v_j), at least 8 px from F and from T, with S
and v_j inside the 28 px window, and F, S (conditions field) and F - v_j, S - v_j (effects field) inside the unit's own RFs
(normalised field response >= --rf-thr); units with no such S are dropped. Remapping: Pearson correlation over the grid places between
the conditions map of the unit and its effects map displaced by v; null: the effects map of another unit displaced by the same v. Curve: the
correlation with the unit's own effects map displaced by the v of another unit, by the distance between the two v. No shape is drawn in the display: it holds three spots (sigma 3 px).
Soft competition: the display weighted by the default filter at its trained width, rescaled to the norm of the unfiltered display,
gives the cosine with the conditions prototype of each unit, multiplied by the RF membership of F and S of that unit (geometric mean
of its normalised field response at F and S on the conditions face and at F - v_u and S - v_u on the effects face, power
--rf-weight); weights are softmax(activation / tau).
Reported per tau: weight of the unit j, share of the weight in j and its 8 lattice neighbours, and the same for the displays of
the other contingencies (chance).
Recalled figure (viewer): the training figure (triangle or rectangle, 9 rotations, 9 x 9 retina offsets) whose view before the saccade
has the largest cosine, in the front-end fovea, with the conditions prototype of the unit (the Euclidean nearest view was dropped: it
picks near-empty views of small norm). After the saccade the viewer only translates it by -v. The distance of S from its outline is
classed as corner (nearest vertex within 4 px), edge (outline within 4 px) or off the outline. Also: the cosine of the best view and
the median cosine of all views.
Readout of the unit j (soft readout, no filter): lone-probe responses of the conditions face (view before the saccade) at the stable
field c_j (centre of the unfiltered effects field), the future field c_j + v_j, S, S - v_j, T and F, and of the effects face (view
after it) at c_j, S - v_j, T (the origin) and F (-v_j), as percentile ranks among the probed places. Null: the same places of the
other contingencies. Type of the conditions field: nearest of F, T and S.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from evaluate_runs import load_params  # noqa: E402
from exp_common import Run, set_device  # noqa: E402
import exp_two_spot_paradigm as M  # noqa: E402
import exp_figure_remap as FR  # noqa: E402

C0 = 7.6
PX = 4.0


def proto_map(w):
    return np.linalg.norm(w.reshape(16, 16, 3), axis=2)


def sample(m, pts):
    """Bilinear sample of a 16 x 16 prototype map at retinal points (x, y), zero outside."""
    c = C0 + pts[:, 0] / PX
    r = C0 + pts[:, 1] / PX
    c0, r0 = np.floor(c).astype(int), np.floor(r).astype(int)
    fc, fr = c - c0, r - r0
    out = np.zeros(len(pts))
    for dr, dc, wt in ((0, 0, (1 - fr) * (1 - fc)), (0, 1, (1 - fr) * fc), (1, 0, fr * (1 - fc)), (1, 1, fr * fc)):
        rr, cc = r0 + dr, c0 + dc
        ok = (rr >= 0) & (rr < 16) & (cc >= 0) & (cc < 16)
        out[ok] += wt[ok] * m[rr[ok], cc[ok]]
    return out


def lattice_near(a, b):
    return np.hypot(a // 10 - b // 10, a % 10 - b % 10) <= 1.5


def pick_s(wc_j, we_j, v, n_best=3, ok=None):
    g = np.arange(-M.WIN, M.WIN + 1, 2.0)
    P = np.stack(np.meshgrid(g, g, indexing="xy"), -1).reshape(-1, 2)
    score = sample(proto_map(wc_j), P) * sample(proto_map(we_j), P - v)
    score[(np.linalg.norm(P, axis=1) < 8) | (np.linalg.norm(P - v, axis=1) < 8)] = -1
    if ok is not None:
        score[~ok(P)] = -1
    picks = []
    for i in np.argsort(-score):
        if score[i] <= 0 or len(picks) == n_best:
            break
        if all(np.linalg.norm(P[i] - q) >= 8 for q in picks):
            picks.append(P[i])
    return np.array(picks), score.max()


def seg_dist(p, a, b):
    t = np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + t * (b - a))))


def outline(p, V):
    """Distance of p to the outline of a polygon and to its nearest vertex."""
    V = np.asarray(V, float)
    return min(seg_dist(p, V[i], V[(i + 1) % len(V)]) for i in range(len(V))), float(np.linalg.norm(V - p, axis=1).min())


def where(p, V, tol=4.0):
    e, c = outline(p, V)
    return "corner" if c <= tol else "edge" if e <= tol else "off"


def retrieve(run, wc, att, cand):
    rows = FR.natural_set(run, wc, att)
    X1 = np.stack([q["x"] for q in rows])
    D1 = (X1 @ wc) / np.maximum(np.linalg.norm(X1, axis=1), 1e-12)[:, None] / np.linalg.norm(wc, axis=0)
    out = {}
    for a in cand:
        j = a["j"]
        i1 = int(D1[:, j].argmax())
        r1 = rows[i1]
        out[j] = dict(fig=FR.FIGS[r1["fig"]], rot=int(r1["rot"]), off=[float(x) for x in r1["off"]], V=[[float(x) for x in (p - 40.0)] for p in r1["V0"]],
                      cos=float(D1[i1, j]), med=float(np.median(D1[:, j])))
    return out


def activation(agent, wc, spots):
    ps = agent._saliency(M.render(spots, M.SPOT_SIGMA))[0]
    x = M.masked_fovea(agent, [ps], M.mask_at(agent, (0, 0), 1.0))[0]
    x0 = M.masked_fovea(agent, [ps], np.ones((80, 80)))[0]
    return (x @ wc) / max(np.linalg.norm(x0), 1e-9) / np.linalg.norm(wc, axis=0)


def softmax(a, tau):
    e = np.exp((a - a.max()) / tau)
    return e / e.sum()


def analyse(path, agent, taus, perms, rng, run=None, rfw=1.0, rfthr=0.5):
    std = load_params(path).neighborhood_modulation_baseline
    st = torch.load(os.path.join(path, "off_control_store"), map_location="cpu", weights_only=False)
    wc = st["visual_conditions_map_state_dict"]["weights"].numpy().astype(float)
    we = st["visual_effects_map_state_dict"]["weights"].numpy().astype(float)
    d = (st["attention_map_state_dict"]["weights"].numpy().astype(float).T - 0.5) * 80
    kern = M.kernel(std, wc.shape[1])
    sal = M.probe_saliency(agent, M.PROBE_SIGMA)
    n0 = np.linalg.norm(M.masked_fovea(agent, sal, np.ones((80, 80))), axis=1)
    Vc = M.probe_views(agent, sal, d, M.NOFILTER, wc, n0, "soft")
    Ve = M.probe_views(agent, sal, d, M.NOFILTER, we, n0, "soft")
    vidx = np.where(M.VALID)[0]
    nU = wc.shape[1]
    zf = lambda r: np.clip(r[vidx] - np.median(r[vidx]), 0, None) / max(np.clip(r[vidx] - np.median(r[vidx]), 0, None).max(), 1e-12)
    Rc = np.stack([M.response(Vc, u, kern, "soft")[vidx] for u in range(nU)])
    Re = np.stack([M.response(Ve, u, kern, "soft")[vidx] for u in range(nU)])
    Zc = np.stack([zf(M.response(Vc, u, kern, "soft")) for u in range(nU)])
    Ze = np.stack([zf(M.response(Ve, u, kern, "soft")) for u in range(nU)])
    PL = M.S_GRID[vidx]
    dr = np.round(d)

    def at(Z, pts):
        dd = np.linalg.norm(PL[None] - pts[:, None], axis=2)
        return np.where(dd.min(1) <= 3.0, Z[np.arange(nU), dd.argmin(1)], 0.0)


    def fld(z, pts):
        dd = np.linalg.norm(PL[None] - np.asarray(pts, float)[:, None], axis=2)
        return np.where(dd.min(1) <= 3.0, z[dd.argmin(1)], 0.0)

    cand = []
    for j in range(wc.shape[1]):
        v = np.round(d[j])
        if np.linalg.norm(v) < 16 or (np.abs(v) > M.WIN).any():
            continue
        if fld(Zc[j], [[0, 0]])[0] < rfthr or fld(Ze[j], [-v])[0] < rfthr:
            continue
        picks, sc = pick_s(wc[:, j], we[:, j], v, ok=lambda P, j=j, v=v: (fld(Zc[j], P) >= rfthr) & (fld(Ze[j], P - v) >= rfthr))
        if not len(picks):
            continue
        cand.append(dict(j=j, v=v, S=picks[0], picks=picks, score=sc))
    n = len(cand)
    out = dict(run=os.path.basename(path), n_units=wc.shape[1], n_cand=n, rf_weight=rfw, rf_thr=rfthr)
    def member(S):
        """Per unit: geometric mean of its RF response at F and S (conditions face) and at F - v_u and S - v_u (effects face)."""
        Sn = np.tile(np.asarray(S, float), (nU, 1))
        return (at(Zc, np.zeros((nU, 2))) * at(Zc, Sn) * at(Ze, -dr) * at(Ze, Sn - dr)) ** 0.25

    def act(S, v):
        return activation(agent, wc, [(0, 0), tuple(v), tuple(S)]) * member(S) ** rfw

    A = np.stack([act(a["S"], a["v"]) for a in cand])

    def q8(a):
        return np.round(255 * np.clip(np.asarray(a, float), 0, None) / max(np.max(a), 1e-12)).astype(int).tolist()

    viewer = dict(run=out["run"], places=[[float(x), float(y)] for x, y in M.S_GRID[vidx]], units={}, maps=(Rc, Re))
    if run is not None:
        shapes = retrieve(run, wc, st["attention_map_state_dict"]["weights"].numpy().astype(float).T, cand)
        pre = [where(np.asarray(a["S"]), shapes[a["j"]]["V"]) for a in cand]
        sh = lambda k: float(np.mean([x == k for x in pre]))
        out.update(shape_cos=float(np.median([shapes[a["j"]]["cos"] for a in cand])), shape_med=float(np.median([shapes[a["j"]]["med"] for a in cand])),
                   S_corner=sh("corner"), S_edge=sh("edge"), S_off=sh("off"))
    for i, a in enumerate(cand):
        acts = [act(p, a["v"]) for p in a["picks"]]
        viewer["units"][int(a["j"])] = dict(v=[float(x) for x in a["v"]], picks=[[float(x) for x in p] for p in a["picks"]],
                                            act=[np.round(1000 * x).astype(int).tolist() for x in acts], cond=q8(wc[:, a["j"]]), eff=q8(we[:, a["j"]]))
        if run is not None:
            viewer["units"][int(a["j"])]["shape"] = shapes[a["j"]]
    out.update(act_top=float(A.max(1).mean()), act_median=float(np.median(A)), act_own=float(np.mean([A[i, a["j"]] for i, a in enumerate(cand)])))
    near = np.array([[lattice_near(a["j"], u) for u in range(wc.shape[1])] for a in cand])
    off = ~np.eye(n, dtype=bool)
    rec_rows = []
    for tau in taus:
        Pm = np.stack([softmax(A[i], tau) for i in range(n)])
        own = np.array([[Pm[i, b["j"]] for b in cand] for i in range(n)])
        mass = np.array([[Pm[i][near[k]].sum() for k in range(n)] for i in range(n)])
        top = np.array([[int(Pm[i].argmax() == b["j"]) for b in cand] for i in range(n)])
        rec_rows.append(dict(run=out["run"], tau=tau, n=n, p_own=np.trace(own) / n, p_own_null=own[off].mean(), mass_near=np.trace(mass) / n,
                             mass_near_null=mass[off].mean(), top1=np.trace(top) / n, top1_null=top[off].mean(),
                             top_near=np.mean([near[i][Pm[i].argmax()] for i in range(n)]),
                             top_near_null=np.mean([near[k][Pm[i].argmax()] for i in range(n) for k in range(n) if k != i]),
                             entropy=float(np.mean([-(p * np.log(p + 1e-12)).sum() for p in Pm]))))
    side = int(round(np.sqrt(len(vidx))))
    gy, gx = np.mgrid[0:side, 0:side].astype(float)
    shift = lambda m, v: map_coordinates(m.reshape(side, side), [gy - v[1] / PX, gx - v[0] / PX], order=1, mode="constant", cval=np.nan).reshape(-1)

    def corr(a, b):
        ok = ~np.isnan(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1])

    v_arr = np.array([a["v"] for a in cand])
    C = np.array([[corr(Rc[a["j"]], shift(Re[a["j"]], v_arr[k])) for k in range(n)] for a in cand])
    X = np.array([[corr(Rc[a["j"]], shift(Re[b["j"]], v_arr[i])) for b in cand] for i, a in enumerate(cand)])
    fix = np.array([corr(Rc[a["j"]], Re[a["j"]]) for a in cand])
    rows = []
    obs = np.diag(X)
    pm = np.array([X[np.arange(n), rng.permutation(n)].mean() for _ in range(perms)])
    rows.append(dict(run=out["run"], place="shift", n=n, mean=obs.mean(), null=X[off].mean(), p=(1 + (pm >= obs.mean()).sum()) / (1 + perms),
                     share_above=(obs > np.array([X[i][off[i]].mean() for i in range(n)])).mean()))
    err = np.linalg.norm(v_arr[None] - v_arr[:, None], axis=2)
    for lo in range(0, 56, 8):
        m = off & (err >= lo) & (err < lo + 8)
        rows.append(dict(run=out["run"], place="curve", lo=lo, hi=lo + 8, n=int(m.sum()), mean=C[m].mean() if m.any() else np.nan))
    rows.append(dict(run=out["run"], place="fix", n=n, mean=fix.mean(), null=np.nan, share_above=np.nan))
    units = [dict(run=out["run"], unit=a["j"], v=float(np.linalg.norm(a["v"])), dist_ST=float(np.linalg.norm(a["S"] - a["v"])),
                  dist_SF=float(np.linalg.norm(a["S"])), shift=obs[i], fix=fix[i], shift_null=X[i][off[i]].mean()) for i, a in enumerate(cand)]
    return out, rec_rows, rows, units, viewer


def write(path, data):
    keys = list(dict.fromkeys(k for r in data for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--tau", type=float, action="append")
    ap.add_argument("--probe-sigma", type=float, default=6.0)
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--rf-weight", type=float, default=1.0, help="exponent of the RF membership of F and S in the activation (0 = cosine only)")
    ap.add_argument("--rf-thr", type=float, default=0.5, help="a point is inside an RF when the normalised field response is at least this")
    ap.add_argument("--shapes", action="store_true", help="retrieve the training figure nearest to the prototypes of each unit")
    args = ap.parse_args()
    set_device()
    runs = [r["run"] for r in csv.DictReader(open(args.manifest))]
    agent = Run(runs[0]).agent
    M.PROBE_SIGMA = args.probe_sigma
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(0)
    summ, rec, rows, units = [], [], [], []
    import json
    for path in runs:
        o, r1, r2, r3, vw = analyse(path, agent, args.tau or [0.01, 0.03, 0.1], args.perms, rng, Run(path) if args.shapes else None, args.rf_weight, args.rf_thr)
        summ.append(o)
        rec += r1
        rows += r2
        units += r3
        Rc, Re = vw.pop("maps")
        np.savez(os.path.join(args.out, f"contingency_maps_{o['run']}.npz"), Rc=Rc, Re=Re, places=np.array(vw["places"]))
        json.dump(vw, open(os.path.join(args.out, f"contingency_viewer_{o['run']}.json"), "w"), separators=(",", ":"))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items()}, flush=True)
    for name, data in (("summary", summ), ("recruit", rec), ("places", rows), ("units", units)):
        write(os.path.join(args.out, f"contingency_{name}.csv"), data)


if __name__ == "__main__":
    main()
