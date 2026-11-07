"""Remapping paradigm with figures: the cue is a corner of a trained figure (after Duhamel et al. 1992).

Usage:
    python scripts/exp_figure_remap.py --verify --agent-run RUN
    python scripts/exp_figure_remap.py --manifest FILE [--out DIR] [--scale 0.15 --scale 0.5]

Stimuli: both figures of training (triangle, rectangle called square) at 9 rotations and a 9 x 9 grid of retina offsets (+-20 px)
around the object, as in exp_labelled_foveas.py. For each stimulus the view before the saccade (default centred mask) recruits
the unit j of the conditions map, as in the model; its attention prototype gives the saccade vector v_j (image px), the retina
moves by v_j and the view after the saccade is rendered. The cue is a corner (vertex) of the figure: S_pre is its place in the
view before the saccade (the future field of the unit) and S_post = S_pre - v_j its place after it (the stable RF).
Places are in retina px from the image centre (pixel 40), x to the right, y down; the window of the model is +-28 px.
Alignment: rank of the effects prototype of j among the 100 effects prototypes for the real view after the saccade (Euclid), and
the same for views of other stimuli.
Corner fields: isolated corners (a vertex with a Gaussian window of sigma 6 px, same figure and rotation as the stimulus) are
placed at the places of a grid (step 4 px), passed through the Gabor front end and read with the soft readout of
exp_two_spot_paradigm.py (conditions face for the view before the saccade, effects face for the view after it). The rank of the
field of unit j at S_pre (conditions) and at S_post (effects) is averaged over the cue corners of the unit's stimuli (units with at
least --min-pairs corners); null: the places of the corners of the other units. Filters (--scale x the trained mask sd): none, on F (centre)
and on T (the attention prototype of j) for the conditions face; none and at the centre for the effects face (attention on T after
the saccade, in the frame of the view after it).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from evaluate_runs import load_params  # noqa: E402
from exp_common import CENTRE, Run, set_device  # noqa: E402
import exp_two_spot_paradigm as M  # noqa: E402

LOCAL = {
    "square": [(-20.440325, -19.039692), (11.221761, -19.039692), (11.221761, 7.4436912), (-20.440325, 7.4436912)],
    "triangle": [(-21.371218, 7.4959962), (11.932626, 7.4959962), (-4.7193016, -21.404824)],
}
FIGS = ["triangle", "square"]
N_ROT = 9
GRID = np.linspace(-20, 20, 9)
IDX = np.array([8 + 4 * (a // 2) + 1 + (a % 2) for a in range(32)])
SIGMA_W = 6.0
VREF = np.array([39.0, 40.0])


def vertices_px(label, rot, retina_pos, pos=(40.0, 40.0)):
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c, -s], [s, c]])
    w = np.array(pos) + np.array(LOCAL[label]) @ R.T
    return np.stack([w[:, 0] - retina_pos[0] + 40 - 1, 40 - w[:, 1] + retina_pos[1]], 1)


def fovea32(s32, mask32, agent):
    """Front-end fovea from the 32 x 32 sampled saliency (2 x 2 mean of the pixels the crop resize uses), as the JavaScript port."""
    x = s32 * mask32[..., None]
    x = x.reshape(16, 2, 16, 2, 3).sum((1, 3)) * 0.25 * agent.focus_params.fovea_gain / 255.0
    return x.reshape(-1)


def fovea_batch(S32, mask32, gain):
    x = S32 * mask32[None, ..., None]
    n = len(S32)
    return (x.reshape(n, 16, 2, 16, 2, 3).sum((2, 4)) * 0.25 * gain / 255.0).reshape(n, -1)


def natural_set(run, wc, att, grid=GRID, n_rot=N_ROT):
    env, agent = run.env, run.agent
    agent.set_parameters(np.array([0.5, 0.5]))
    rows = []
    for fi, label in enumerate(FIGS):
        wid = env.world_labels.index(label)
        for k in range(n_rot):
            rot = 2 * np.pi * k / n_rot
            for dy in grid:
                for dx in grid:
                    env.init_world(world=wid, object_params={"pos": [40.0, 40.0], "rot": float(rot)})
                    obs, info = env.reset()
                    env.info = info
                    obs = env.step(CENTRE - np.array([dx, dy]) - env.retina_sim_pos)[0]
                    rp0 = env.retina_sim_pos.copy()
                    x = agent.get_fovea(obs).ravel() / 255.0
                    j = int(M.distances(x[None], wc).argmin(-1)[0])
                    v = (att[j] - 0.5) * 80.0
                    obs2 = env.step(np.array([v[0], -v[1]]))[0]
                    rp1 = env.retina_sim_pos.copy()
                    x2 = agent.get_fovea(obs2).ravel() / 255.0
                    moved = rp1 - rp0
                    rows.append(dict(fig=fi, rot=k, off=(dx, dy), j=j, v=v, moved=np.array([moved[0], -moved[1]]), x=x, x2=x2,
                                     V0=vertices_px(label, rot, rp0), V1=vertices_px(label, rot, rp1)))
    return rows


def corner_bank(run, types, places, rng_sigma=SIGMA_W):
    """Sampled saliency of an isolated corner of every (figure, rotation, vertex) type at every place."""
    env, agent = run.env, run.agent
    gx, gy = np.meshgrid(np.arange(80), np.arange(80))
    win = np.exp(-((gx - VREF[0]) ** 2 + (gy - VREF[1]) ** 2) / (2 * rng_sigma**2))
    bank = {}
    for (fi, k, vi) in types:
        label = FIGS[fi]
        rot = 2 * np.pi * k / N_ROT
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])
        pos = np.array([40.0, 40.0]) - R @ np.array(LOCAL[label][vi])
        env.init_world(world=env.world_labels.index(label), object_params={"pos": [float(pos[0]), float(pos[1])], "rot": float(rot)})
        obs, info = env.reset()
        img = obs["RETINA"].astype(np.float32)
        dark = 255.0 - img
        out = np.empty((len(places), 32, 32, 3), np.float32)
        for q, (sx, sy) in enumerate(places):
            dx, dy = int(1 + sx), int(sy)
            shifted = np.zeros_like(dark)
            ys, yd = (slice(max(0, -dy), min(80, 80 - dy)), slice(max(0, dy), min(80, 80 + dy)))
            xs, xd = (slice(max(0, -dx), min(80, 80 - dx)), slice(max(0, dx), min(80, 80 + dx)))
            shifted[yd, xd] = (dark * win[..., None])[ys, xs]
            im = np.clip(255.0 - shifted, 0, 255).astype(np.uint8)
            out[q] = agent._saliency(im)[0][np.ix_(IDX, IDX)]
        bank[(fi, k, vi)] = out
    return bank


def summarise(path, agent, scales, args, rng):
    run = args._run
    st = torch.load(os.path.join(path, "off_control_store"), map_location="cpu", weights_only=False)
    wc = st["visual_conditions_map_state_dict"]["weights"].numpy().astype(float)
    we = st["visual_effects_map_state_dict"]["weights"].numpy().astype(float)
    att = st["attention_map_state_dict"]["weights"].numpy().astype(float).T
    kern = M.kernel(load_params(path).neighborhood_modulation_baseline, wc.shape[1])
    rows = natural_set(run, wc, att)
    n = len(rows)
    out = dict(run=os.path.basename(path), n_stimuli=n, n_units=len({r["j"] for r in rows}))
    # alignment of the effects prototype with the real view after the saccade
    X2 = np.stack([r["x2"] for r in rows])
    X1 = np.stack([r["x"] for r in rows])
    J = np.array([r["j"] for r in rows])
    D2 = M.distances(X2, we)
    rank_eff = (D2 < D2[np.arange(n), J][:, None]).sum(1) + 1
    perm = rng.permutation(n)
    rank_shuf = (D2[np.arange(n)] < D2[np.arange(n), J[perm]][:, None]).sum(1) + 1
    Dc = M.distances(X1, wc)
    rank_cond_post = (Dc < Dc[np.arange(n), J][:, None]).sum(1) + 1
    D2c = M.distances(X2, wc)
    rank_cond_for_post = (D2c < D2c[np.arange(n), J][:, None]).sum(1) + 1
    out.update(rank_eff_median=float(np.median(rank_eff)), top1_eff=float((rank_eff == 1).mean()), top5_eff=float((rank_eff <= 5).mean()),
               rank_eff_shuffled_median=float(np.median(rank_shuf)), rank_cond_for_post_median=float(np.median(rank_cond_for_post)),
               top5_cond_for_post=float((rank_cond_for_post <= 5).mean()), cond_winner_share=float((rank_cond_post == 1).mean()))
    # cue corners
    pairs = []
    for si, r in enumerate(rows):
        for vi in range(len(r["V0"])):
            s0 = r["V0"][vi] - 40.0
            s1 = r["V1"][vi] - 40.0
            if (np.abs(s0) <= M.WIN).all() and (np.abs(s1) <= M.WIN).all():
                pairs.append(dict(si=si, j=r["j"], t=(r["fig"], r["rot"], vi), S0=s0, S1=s1, v=r["v"], dT=float(np.linalg.norm(s0 - r["v"]))))
    out.update(n_pairs=len(pairs))
    types = sorted({p["t"] for p in pairs})
    places = M.S_GRID[M.VALID]
    bank = corner_bank(run, types, places)
    gain = agent.focus_params.fovea_gain
    ones32 = np.ones((32, 32))
    n0 = {t: np.linalg.norm(fovea_batch(bank[t], ones32, gain), axis=1) for t in types}
    wcn, wen = np.linalg.norm(wc, axis=0), np.linalg.norm(we, axis=0)
    full = np.zeros(len(M.S_GRID))
    vidx = np.where(M.VALID)[0]
    units = sorted({p["j"] for p in pairs})
    by_unit = {j: [p for p in pairs if p["j"] == j] for j in units}
    units = [j for j in units if len(by_unit[j]) >= args.min_pairs]
    ctr = M.mask_at(agent, (0, 0), 1.0)[np.ix_(IDX, IDX)]

    def field(face, j, t, mask32):
        W, Wn = (wc, wcn) if face == "c" else (we, wen)
        x = fovea_batch(bank[t], mask32, gain)
        sim = (x @ W) / np.maximum(n0[t], 1e-9)[:, None] / Wn
        r = np.zeros(len(M.S_GRID))
        r[vidx] = sim @ kern[j]
        return r

    def rank_at(r, place):
        return float(M.place_pct(r[None], np.asarray(place, float)[None])[0])

    summ, unit_rows, examples = [], [], {}
    all_S0 = np.array([p["S0"] for p in pairs])
    all_S1 = np.array([p["S1"] for p in pairs])
    owner = np.array([p["j"] for p in pairs])
    for sc in scales:
        res = {}
        for j in units:
            ps = by_unit[j]
            vj = att[j] - 0.5
            vj = vj * 80.0
            masks = {"none": ones32, "F": M.mask_at(agent, (0, 0), sc)[np.ix_(IDX, IDX)], "T": M.mask_at(agent, vj, sc)[np.ix_(IDX, IDX)],
                     "centre": M.mask_at(agent, (0, 0), sc)[np.ix_(IDX, IDX)]}
            cache = {}
            def get(face, t, mk):
                key = (face, t, mk)
                if key not in cache:
                    cache[key] = field(face, j, t, masks[mk])
                return cache[key]
            cols = {f"c_{m}": [] for m in ("none", "F", "T")}
            cols.update({f"e_{m}": [] for m in ("none", "centre")})
            nulls = {k: [] for k in cols}
            others = np.where(owner != j)[0]
            for p in ps:
                for m in ("none", "F", "T"):
                    r = get("c", p["t"], m)
                    cols[f"c_{m}"].append(rank_at(r, p["S0"]))
                    for q in rng.choice(others, size=args.null_draws, replace=False):
                        nulls[f"c_{m}"].append(rank_at(r, all_S0[q]))
                for m in ("none", "centre"):
                    r = get("e", p["t"], m)
                    cols[f"e_{m}"].append(rank_at(r, p["S1"]))
                    for q in rng.choice(others, size=args.null_draws, replace=False):
                        nulls[f"e_{m}"].append(rank_at(r, all_S1[q]))
            g_err, g_null, s0_d, s0_null, s1_d, s1_null = [], [], [], [], [], []
            for p in ps:
                c1 = M.field_centre(get("c", p["t"], "none"), j, kern, "soft")
                c2 = M.field_centre(get("e", p["t"], "none"), j, kern, "soft")
                if np.isnan(c1).any() or np.isnan(c2).any():
                    continue
                g_err.append(np.linalg.norm(c1 - c2 - vj))
                s0_d.append(np.linalg.norm(c1 - p["S0"]))
                s1_d.append(np.linalg.norm(c2 - p["S1"]))
                for q in rng.choice(others, size=args.null_draws, replace=False):
                    g_null.append(np.linalg.norm(c1 - c2 - pairs[q]["v"]))
                    s0_null.append(np.linalg.norm(c1 - all_S0[q]))
                    s1_null.append(np.linalg.norm(c2 - all_S1[q]))
            far = np.array([p["dT"] >= args.far for p in ps])
            row = dict(run=out["run"], scale=sc, unit=j, n_pairs=len(ps), n_far=int(far.sum()), vnorm=float(np.linalg.norm(vj)),
                       mean_dT=float(np.mean([p["dT"] for p in ps])))
            row.update(geom_err=float(np.median(g_err)), geom_null=float(np.median(g_null)), c1_to_S0=float(np.median(s0_d)), c1_to_S0_null=float(np.median(s0_null)),
                       c2_to_S1=float(np.median(s1_d)), c2_to_S1_null=float(np.median(s1_null)))
            for k in cols:
                row[k] = float(np.mean(cols[k]))
                row[k + "_null"] = float(np.mean(nulls[k]))
                row[k + "_far"] = float(np.mean(np.array(cols[k])[far])) if far.any() else np.nan
            # type of the conditions field with the filter on T for the far cues: nearest to F, T or S
            kind = []
            for p in ps:
                if p["dT"] < args.far:
                    continue
                r = get("c", p["t"], "T")
                c = M.field_centre(r, j, kern, "soft")
                if np.isnan(c).any():
                    continue
                kind.append(int(np.argmin([np.linalg.norm(c), np.linalg.norm(c - vj), np.linalg.norm(c - p["S0"])])))
            row.update(kind_F=float(np.mean(np.array(kind) == 0)) if kind else np.nan, kind_T=float(np.mean(np.array(kind) == 1)) if kind else np.nan,
                       kind_S=float(np.mean(np.array(kind) == 2)) if kind else np.nan, n_kind=len(kind))
            unit_rows.append(row)
        sub = [r for r in unit_rows if r["scale"] == sc and r["run"] == out["run"]]
        summ.append(dict(run=out["run"], scale=sc, n_units=len(sub)))
    out["units_used"] = len(units)
    viewer = dict(run=out["run"], scales=list(scales), units={}, examples=[])
    pos_part = lambda a: np.clip(a, 0, None)
    def q8(a):
        a = pos_part(np.asarray(a, float))
        return np.round(255 * a / max(a.max(), 1e-12)).astype(int).tolist()
    def zfield(r):
        v = r[vidx]
        z = np.clip(v - np.median(v), 0, None)
        return np.round(255 * z / max(z.max(), 1e-12)).astype(int).tolist()
    order = sorted(units, key=lambda j: -len(by_unit[j]))[: args.viewer_units]
    for j in order:
        viewer["units"][int(j)] = dict(v=[float(x) for x in (att[j] - 0.5) * 80.0], cond=q8(wc[:, j]), eff=q8(we[:, j]), n_pairs=len(by_unit[j]))
    used = set()
    for j in order:
        seen_fig = set()
        for p in by_unit[j]:
            r = rows[p["si"]]
            if p["si"] in used or r["fig"] in seen_fig:
                continue
            seen_fig.add(r["fig"])
            used.add(p["si"])
            ex = dict(unit=int(j), fig=FIGS[r["fig"]], rot=int(r["rot"]), off=[float(x) for x in r["off"]], v=[float(x) for x in r["v"]],
                      V0=[[float(x) for x in (a - 40.0)] for a in r["V0"]], V1=[[float(x) for x in (a - 40.0)] for a in r["V1"]],
                      pre=q8(r["x"]), post=q8(r["x2"]), cues=[])
            for pp in [pp for pp in by_unit[j] if pp["si"] == p["si"]]:
                cue = dict(vertex=int(pp["t"][2]), S0=[float(x) for x in pp["S0"]], S1=[float(x) for x in pp["S1"]], fields={})
                for sc in scales:
                    vj = (att[j] - 0.5) * 80.0
                    mk = {"none": ones32, "F": M.mask_at(agent, (0, 0), sc)[np.ix_(IDX, IDX)], "T": M.mask_at(agent, vj, sc)[np.ix_(IDX, IDX)]}
                    cue["fields"][str(sc)] = dict(
                        c_none=zfield(field("c", j, pp["t"], mk["none"])), c_F=zfield(field("c", j, pp["t"], mk["F"])), c_T=zfield(field("c", j, pp["t"], mk["T"])),
                        e_none=zfield(field("e", j, pp["t"], mk["none"])), e_centre=zfield(field("e", j, pp["t"], mk["F"])),
                        rank=dict(c_none=rank_at(field("c", j, pp["t"], mk["none"]), pp["S0"]), c_F=rank_at(field("c", j, pp["t"], mk["F"]), pp["S0"]), c_T=rank_at(field("c", j, pp["t"], mk["T"]), pp["S0"]),
                                  e_none=rank_at(field("e", j, pp["t"], mk["none"]), pp["S1"]), e_centre=rank_at(field("e", j, pp["t"], mk["F"]), pp["S1"])))
                ex["cues"].append(cue)
            viewer["examples"].append(ex)
    viewer["places"] = [[float(a), float(b)] for a, b in places]
    return out, unit_rows, rows, pairs, viewer


def verify(args):
    run = Run(args.agent_run)
    env, agent = run.env, run.agent
    agent.set_parameters(np.array([0.5, 0.5]))
    # (a) sampled front end equals the agent's fovea with a mask
    env.init_world(world=env.world_labels.index("square"), object_params={"pos": [40.0, 40.0], "rot": 0.7})
    obs, _ = env.reset()
    sal = agent._saliency(obs["RETINA"])[0]
    err = 0.0
    for sc in (1.0, 0.15):
        mask = M.mask_at(agent, (7, -12), sc)
        ref = M.masked_fovea(agent, [sal], mask)[0] * 255.0 / 255.0
        mine = fovea_batch(sal[np.ix_(IDX, IDX)][None], mask[np.ix_(IDX, IDX)], agent.focus_params.fovea_gain)[0]
        err = max(err, float(np.abs(ref - mine).max() / max(np.abs(ref).max(), 1e-9)))
    ok_a = err < 1e-4
    print(f"sampled front end vs agent fovea: max relative error {err:.2e}", "PASS" if ok_a else "FAIL")
    # (b) the corner of the figure moves by -v when the retina moves by v
    st = torch.load(os.path.join(args.agent_run, "off_control_store"), map_location="cpu", weights_only=False)
    wc = st["visual_conditions_map_state_dict"]["weights"].numpy().astype(float)
    att = st["attention_map_state_dict"]["weights"].numpy().astype(float).T
    rows = natural_set(run, wc, att, grid=np.linspace(-20, 20, 3), n_rot=3)
    d = max(float(np.abs((r["V1"] - r["V0"]) + r["moved"]).max()) for r in rows)
    ok_b = d < 1e-6
    print(f"corner displacement equals minus the executed saccade over {len(rows)} stimuli: max error {d:.2e}", "PASS" if ok_b else "FAIL")
    # (c) projected vertices lie on the boundary of the rendered figure
    errs = []
    for label in FIGS:
        for rot in (0.0, 0.7, 2.1, 4.0):
            for ret in ((40, 40), (30, 50), (55, 35)):
                env.init_world(world=env.world_labels.index(label), object_params={"pos": [40.0, 40.0], "rot": float(rot)})
                obs, _ = env.reset()
                obs = env.step(np.array(ret, float) - env.retina_sim_pos)[0]
                mask = np.abs(obs["RETINA"].astype(int) - 255).sum(-1) > 60
                ys, xs = np.nonzero(mask)
                for v in vertices_px(label, rot, env.retina_sim_pos):
                    if 0 <= v[0] < 80 and 0 <= v[1] < 80:
                        errs.append(float(np.min(np.hypot(xs - v[0], ys - v[1]))))
    ok_c = max(errs) <= 2.0
    print(f"projected vertices on the figure: max distance {max(errs):.2f} px over {len(errs)} vertices", "PASS" if ok_c else "FAIL")
    print("VERIFY", "PASS" if ok_a and ok_b and ok_c else "FAIL")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest")
    ap.add_argument("--out", default=".")
    ap.add_argument("--scale", type=float, action="append")
    ap.add_argument("--min-pairs", type=int, default=8)
    ap.add_argument("--null-draws", type=int, default=20)
    ap.add_argument("--far", type=float, default=15.0)
    ap.add_argument("--viewer-units", type=int, default=24)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--agent-run")
    ap.add_argument("--probe-sigma", type=float, default=6.0)
    args = ap.parse_args()
    set_device()
    M.PROBE_SIGMA = args.probe_sigma
    if args.verify:
        return verify(args)
    rng = np.random.RandomState(0)
    runs = [r["run"] for r in csv.DictReader(open(args.manifest))] if args.manifest else [args.agent_run]
    os.makedirs(args.out, exist_ok=True)
    allunits, summaries = [], []
    for path in runs:
        run = Run(path)
        args._run = run
        o, unit_rows, rows, pairs, viewer = summarise(path, run.agent, args.scale or [0.15], args, rng)
        with open(os.path.join(args.out, f"figure_viewer_{os.path.basename(path)}.json"), "w") as f:
            json.dump(viewer, f, separators=(",", ":"))
        summaries.append(o)
        allunits += unit_rows
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items()}, flush=True)
    for name, data in (("summary", summaries), ("units", allunits)):
        keys = list(dict.fromkeys(k for r in data for k in r))
        with open(os.path.join(args.out, f"figure_{name}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(data)


if __name__ == "__main__":
    main()
