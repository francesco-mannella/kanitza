"""Two-spot predictive-remapping paradigm, recruitment through the conditions map (plan: predictive_remapping.md, 2.7).

Usage:
    python scripts/exp_two_spot_paradigm.py --verify --agent-run RUN
    python scripts/exp_two_spot_paradigm.py --manifest FILE [--out DIR] [--displays N]

Display: spot F at the gaze centre and spot T at v = T - F (integer px, image coordinates, x = column, y = row) on the 80x80
retina (positions from -40 to 40). Spots are red Gaussian blobs (sigma 3 px, peak red) on the white retina, like the objects of training, drawn through
the Gabor front end and the fovea crop (Agent.get_fovea). White blobs on a black or grey retina give a blank fovea: the front end
is colour-opponent.
T is drawn from a probability cloud: the mean of the conditions prototypes of the run (positive part, summed over colour
channels) read as a probability map over the 16 x 16 pixels of the fovea (4 retina px each, uniform jitter inside the
pixel, offset from the gaze centre, rounded to integer px). Draws closer than 16 px to F are rejected; --displays draws
are kept per run.
Recruitment, as in the SOM simulation: the view with both spots (F and T together, before the saccade) is matched to the
conditions prototypes; the winner j is the recruited cell, and its activity is a Gaussian on the lattice (std
neighborhood_modulation_baseline of the run) around it. The attention prototype of the winner is not used; its distance to
v is reported as the recruitment diagnostic act_err.
Receptive fields of the recruited cell, mapped with a lone probe spot S (sigma 3 px, no F and no T) at the places of a 20 x 20
grid (4 px steps) over the whole retina. A probe at s is presented to a map and won by one unit w(s); the recruited cell
responds as far as w(s) lies in its lattice neighbourhood: R(s) = G[j, w(s)], G the Gaussian lattice kernel. The probe
therefore drives the recruited cell only inside its receptive field.
  c1: conditions RF: w(s) is the winner of the conditions map for S at s (before the saccade).
  c2: effects RF: w(s) is the winner of the effects map for the same probe S at s, before the saccade.
A probe that is not well inside the 64 x 64 px window the model sees (more than 28 px from the gaze; the window
edge is at 32 px and the spot has sigma 3 px) gives a blank or nearly blank view that one fixed unit wins; such places have
no response (R = 0).
Field centre: response-weighted centre of the grid places with R >= 0.5 of the maximum; the field is undefined if no place
is won within the recruited cell's neighbourhood (maximum below half the kernel peak of the cell).
Statistics: where the two RFs sit relative to the two spots. With F at the centre and T at v, the view before the
saccade holds F at 0 and T at v; the view after the saccade holds T at the centre (always) and F at -v. The conditions
prototypes of a cell recruited by the two spots therefore hold T at v, and its effects prototypes hold F at -v.
  c1 vs T before the saccade: median distance of the conditions RF centre c1 from v.
  c2 vs F after the saccade: median distance of the effects RF centre c2 from -v.
Testable displays: both RFs defined and |v| <= 28 px on each axis (so that v and -v are inside the probed area). Null for
each: the saccade vectors shuffled across the testable displays (--perms shuffles), which keeps every display's own c1 and
c2; closeness index = 1 - median distance / mean null median distance (0 = as close as chance, 1 = exact) and
p = (1 + number of shuffles with median distance <= observed) / (1 + --perms). Displays share recruited cells, so the
displays are not independent. --max-act-err keeps only the displays whose recruited cell's own saccade is within that many
px of v (the statistics and their nulls are then computed on those displays).
Place responses (fixation-point narrative). The recruited location holds one conditions prototype while the attentional
filter is still centred on F, and the effects prototype once the filter is on T: before the shift F is at the centre (0) and T
at v, after it T is at the centre and F at -v. For each face and each place (0, v, -v) the response of the lone-probe map R at
the place (maximum over the grid cells within 3 px) is converted to a percentile rank among the probed places of that map
(1 = strongest, 0.5 = a random place). Predicted: conditions face at 0 and v, effects face at 0 and -v; the conditions face at -v
and the effects face at v are the specificity checks, and the contrast is the effects percentile minus the conditions
percentile at -v. Mean over testable displays with a 95% bootstrap interval over recruited cells (--boots); for the places that
depend on v, p is the share of --perms shuffles of v across displays with a mean percentile at least as high. The place 0 is
shared by the prototypes of all locations (all hold F before and T after at the centre), so under the winner read-out it is not
selective and is reported without a criterion.
The rest of the analysis is descriptive: recruitment (cells recruited, distance between the recruited cell's own saccade
and v) and the number of displays with both fields defined.
--verify (readout only: the recruited cell is cell i for display i, so the check does not depend on recruitment): synthetic
maps whose unit j holds a lone spot at c_j, on a regular 10 x 10 block of probe-grid places (+-18 px), for both the
conditions and the effects map (lattice std 0.3). For the 8 x 8 interior cells (the edge cells also win the outer places),
c1 and c2 must both equal c_j within 1 px, in at least 95% of the cells. Place statistics: 100 cells with a distinct v_j (multiples of 4 px up to 24 px); conditions unit j holds a lone spot at v_j
and effects unit j a lone spot at -v_j (displays i for cell i). The median distance of c1 from v and of c2 from -v must be
<= 4 px (the probe grid step; the irregular nearest-neighbour RF regions of the synthetic spots shift their centres by up
to a step), the closeness index >= 0.8 and p <= 0.02, for both.
Place percentiles: with two-spot prototypes (conditions unit j holds spots at 0 and v_j, effects unit j spots at 0 and -v_j), the
mean percentile of the conditions face at v and of the effects face at -v must be >= 0.9 and above the mean of the v-shuffled
null with p <= 0.02 (the first version required the shuffled mean to be <= 0.6; it was 0.72 and 0.73 because the many tied
near-zero responses of the maps raise the percentile of any place, so the shuffled mean itself is the comparator); the
percentiles at 0 are reported without a criterion.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from evaluate_runs import grid_coords, load_params, pairwise  # noqa: E402
from exp_common import Run, set_device  # noqa: E402

GX = GY = np.arange(-38, 39, 4).astype(float)
S_GRID = np.stack(np.meshgrid(GX, GY, indexing="xy"), -1).reshape(-1, 2)
SIGMA = 3.0
LIM = 74
PROBE_SIGMA = SIGMA
SPOT_SIGMA = SIGMA
MASK_SD_SCALE = 1.0
FACE_GAIN = 1.0
WIN = 28
YY, XX = np.mgrid[0:80, 0:80]


def render(spots, sigma=SIGMA):
    g = np.zeros((80, 80))
    for x, y in spots:
        g += np.exp(-((XX - 40 - x) ** 2 + (YY - 40 - y) ** 2) / (2 * sigma**2))
    img = np.full((80, 80, 3), 255.0)
    img[..., 1] = img[..., 2] = 255 * (1 - np.clip(g, 0, 1))
    return img.astype(np.uint8)


def fovea(agent, spots, sigma=SIGMA):
    return agent.get_fovea({"RETINA": render(spots, sigma)}).reshape(-1).astype(np.float32) / 255.0


def spot_table(agent, sigma=SIGMA):
    n = 2 * LIM + 1
    out = np.empty((n, n, 768), np.float32)
    for iy in range(n):
        for ix in range(n):
            out[iy, ix] = fovea(agent, [(ix - LIM, iy - LIM)], sigma)
    return out


def distances(x, w):
    return ((x**2).sum(-1)[..., None] + (w**2).sum(0) - 2 * x @ w).clip(0)


def winners(table, w):
    return distances(table.reshape(-1, 768), w).argmin(-1).reshape(table.shape[:2])


def pair_views(agent, v_list):
    pre = np.stack([fovea(agent, [(0, 0), tuple(v)]) for v in v_list])
    post = np.stack([fovea(agent, [(-v[0], -v[1]), (0, 0)]) for v in v_list])
    return pre, post


def sample_targets(w_cond, n, rng):
    m = w_cond.mean(1).reshape(16, 16, 3).clip(0).sum(-1)
    p = (m / m.sum()).ravel()
    out = []
    while len(out) < n:
        k = rng.choice(256, p=p)
        v = np.round((np.array([k % 16, k // 16]) - 7.5) * 4 + rng.uniform(-2, 2, 2))
        if np.linalg.norm(v) >= 16:
            out.append(v)
    return np.array(out)


def kernel(std, n_units=100):
    ld = pairwise(grid_coords(n_units).astype(float))
    k = np.exp(-0.5 * ld**2 / std**2)
    return k / k.sum(1, keepdims=True)


def centre(r):
    w = np.where(r >= 0.5 * r.max(), r, 0)
    return (w[:, None] * S_GRID).sum(0) / w.sum()


def pick(m, i):
    return m[i] if isinstance(m, list) else m


def field(kern, j, win_map, shift):
    if win_map.ndim == 1:
        r = kern[j, win_map].copy()
        r[(np.abs(S_GRID - shift) > WIN).any(1)] = 0
        return r, (centre(r) if r.max() >= 0.5 * kern[j].max() else np.full(2, np.nan))
    xs = (S_GRID[:, 0] - shift[0]).astype(int) + LIM
    ys = (S_GRID[:, 1] - shift[1]).astype(int) + LIM
    r = kern[j, win_map[ys, xs]]
    r[(np.abs(S_GRID - shift) > WIN).any(1)] = 0
    return r, (centre(r) if r.max() >= 0.5 * kern[j].max() else np.full(2, np.nan))


def place_stat(wc, we, win, v_list, kern, nperm, rng):
    c1 = np.array([field(kern, j, pick(wc, i), (0, 0))[1] for i, j in enumerate(win)])
    c2 = np.array([field(kern, j, pick(we, i), (0, 0))[1] for i, j in enumerate(win)])
    ok = ~(np.isnan(c1).any(1) | np.isnan(c2).any(1)) & (np.abs(v_list) <= WIN).all(1)
    c1, c2, v = c1[ok], c2[ok], v_list[ok]
    out = dict(n_test=int(ok.sum()))
    if not ok.any():
        for name in "12":
            out.update({f"e{name}": np.nan, f"null{name}": np.nan, f"index{name}": np.nan, f"p{name}": np.nan})
        return out
    for name, c, sign in (("1", c1, 1), ("2", c2, -1)):
        err = lambda vv: np.median(np.linalg.norm(c - sign * vv, axis=1))
        e = err(v)
        null = np.array([err(v[rng.permutation(len(v))]) for _ in range(nperm)])
        out.update({f"e{name}": e, f"null{name}": null.mean(), f"index{name}": 1 - e / null.mean(),
                    f"p{name}": (1 + (null <= e).sum()) / (1 + nperm)})
    return out


VALID = (np.abs(S_GRID) <= WIN).all(1)


def place_pct(r, place):
    near = np.linalg.norm(S_GRID[None] - place[:, None], axis=2) <= 3
    rp = np.where(near, r, -1).max(1)[:, None]
    rv = r[:, VALID]
    return ((rv < rp).sum(1) + 0.5 * (rv == rp).sum(1)) / VALID.sum()


def place_percentiles(r1, r2, v):
    zero = np.zeros_like(v)
    cols = {"cond_0": place_pct(r1, zero), "cond_v": place_pct(r1, v), "cond_mv": place_pct(r1, -v),
            "eff_0": place_pct(r2, zero), "eff_v": place_pct(r2, v), "eff_mv": place_pct(r2, -v)}
    cols["contrast_mv"] = cols["eff_mv"] - cols["cond_mv"]
    return cols


def place_data(wc, we, win, v_list, kern):
    r1 = np.array([field(kern, j, pick(wc, i), (0, 0))[0] for i, j in enumerate(win)])
    r2 = np.array([field(kern, j, pick(we, i), (0, 0))[0] for i, j in enumerate(win)])
    c1 = np.array([field(kern, j, pick(wc, i), (0, 0))[1] for i, j in enumerate(win)])
    c2 = np.array([field(kern, j, pick(we, i), (0, 0))[1] for i, j in enumerate(win)])
    ok = ~(np.isnan(c1).any(1) | np.isnan(c2).any(1)) & (np.abs(v_list) <= WIN).all(1)
    return r1, r2, ok


def place_response(wc, we, win, v_list, kern, nperm, nboot, rng):
    r1, r2, ok = place_data(wc, we, win, v_list, kern)
    return place_response_from(r1, r2, ok, win, v_list, nperm, nboot, rng)


def place_response_from(r1, r2, ok, win, v_list, nperm, nboot, rng):
    r1, r2, v, cells = r1[ok], r2[ok], v_list[ok], np.asarray(win)[ok]
    cols = place_percentiles(r1, r2, v)
    out = dict(n_test=int(ok.sum()))
    if not ok.any():
        for k in cols:
            out.update({f"pr_{k}": np.nan, f"pr_{k}_lo": np.nan, f"pr_{k}_hi": np.nan})
        for k in ("cond_v", "cond_mv", "eff_v", "eff_mv", "cond_0", "eff_0"):
            out.update({f"pr_{k}_null": np.nan, f"pr_{k}_p": np.nan, f"pr_{k}_plo": np.nan})
        return out
    uc = np.unique(cells)
    idx = [np.where(cells == c)[0] for c in uc]
    for k, x in cols.items():
        boot = [x[np.concatenate([idx[i] for i in rng.randint(len(uc), size=len(uc))])].mean() for _ in range(nboot)]
        out.update({f"pr_{k}": x.mean(), f"pr_{k}_lo": np.percentile(boot, 2.5), f"pr_{k}_hi": np.percentile(boot, 97.5)})
    nulls = {k: [] for k in ("cond_v", "cond_mv", "eff_v", "eff_mv")}
    for _ in range(nperm):
        vs = v[rng.permutation(len(v))]
        nulls["cond_v"].append(place_pct(r1, vs).mean())
        nulls["cond_mv"].append(place_pct(r1, -vs).mean())
        nulls["eff_v"].append(place_pct(r2, vs).mean())
        nulls["eff_mv"].append(place_pct(r2, -vs).mean())
    for k, nl in nulls.items():
        out[f"pr_{k}_null"] = np.mean(nl)
        out[f"pr_{k}_p"] = (1 + (np.array(nl) >= cols[k].mean()).sum()) / (1 + nperm)
        out[f"pr_{k}_plo"] = (1 + (np.array(nl) <= cols[k].mean()).sum()) / (1 + nperm)
    rs, rr = S_GRID[VALID], np.random.RandomState(3)
    for k, r in (("cond_0", r1), ("eff_0", r2)):
        nl = np.array([place_pct(r, rs[rr.randint(len(rs), size=len(v))]).mean() for _ in range(nperm)])
        out[f"pr_{k}_null"] = nl.mean()
        out[f"pr_{k}_p"] = (1 + (nl >= cols[k].mean()).sum()) / (1 + nperm)
        out[f"pr_{k}_plo"] = (1 + (nl <= cols[k].mean()).sum()) / (1 + nperm)
    return out


def paradigm(w_cond, w_eff, d, pre, post, v_list, std, table, win=None, nperm=500, max_act_err=None, nboot=1000, wc=None, we=None):
    kern = kernel(std, w_cond.shape[1])
    if wc is None:
        wc, we = winners(table, w_cond), winners(table, w_eff)
    if win is None:
        win = distances(pre, w_cond).argmin(-1)
    n_fields = 0
    for i, j in enumerate(win):
        c1 = field(kern, j, pick(wc, i), (0, 0))[1]
        c2 = field(kern, j, pick(we, i), (0, 0))[1]
        n_fields += not (np.isnan(c1).any() or np.isnan(c2).any())
    out = dict(n_displays=len(win), n_recruited=len(set(win)), n_fields=n_fields,
               act_err=np.median(np.linalg.norm(d[win] - v_list, axis=1)))
    keep = np.ones(len(win), bool) if max_act_err is None else np.linalg.norm(d[win] - v_list, axis=1) <= max_act_err
    out["n_kept"] = int(keep.sum())
    kept = np.nonzero(keep)[0]
    wc_k = [wc[i] for i in kept] if isinstance(wc, list) else wc
    we_k = [we[i] for i in kept] if isinstance(we, list) else we
    out.update(place_stat(wc_k, we_k, win[keep], v_list[keep], kern, nperm, np.random.RandomState(1)))
    out.update(place_response(wc_k, we_k, win[keep], v_list[keep], kern, nperm, nboot, np.random.RandomState(2)))
    return out


def probe_saliency(agent, sigma=SIGMA):
    return np.stack([agent._saliency(render([tuple(s)], sigma))[0] for s in S_GRID])


def mask_at(agent, d_px, scale=None):
    agent.set_parameters(0.5 + np.asarray(d_px, float) / 80)
    return agent.attentional_mask.copy() ** (1 / (MASK_SD_SCALE if scale is None else scale) ** 2)


def masked_fovea(agent, sal, mask):
    return np.stack([agent._fovea({}, x * mask[..., None]).reshape(-1) for x in sal]).astype(np.float32) * FACE_GAIN / 255.0


def masked_views(agent, w, d, v_list, mode="split"):
    """Views through the attentional filter. Recruitment always uses the filter at fixation. split: conditions probes at
    fixation, effects probes at the attention prototype nearest to T; before: all probes at fixation; after: all probes
    at that prototype."""
    sal = probe_saliency(agent, PROBE_SIGMA)
    near = np.linalg.norm(d[None] - v_list[:, None], axis=2).argmin(1)
    key_c = np.full(len(v_list), -1) if mode in ("split", "before") else near
    key_e = near if mode in ("split", "after") else np.full(len(v_list), -1)
    centre = lambda k: (0, 0) if k < 0 else d[k]
    pairs = [agent._saliency(render([(0, 0), tuple(v)]))[0] for v in v_list]
    pre = np.stack([masked_fovea(agent, [pairs[i]], mask_at(agent, (0, 0)))[0] for i in range(len(v_list))])
    cache = {}
    for name, keys, wm in (("c", key_c, w["visual_conditions"]), ("e", key_e, w["visual_effects"])):
        for k in np.unique(keys):
            cache[name, k] = distances(masked_fovea(agent, sal, mask_at(agent, centre(k))), wm).argmin(-1)
    return pre, [cache["c", k] for k in key_c], [cache["e", k] for k in key_e]


BLANK_FLOOR = 0.25
NOFILTER = -2
FIELD_MIN_V = 16


def recruit(agent, w_cond, v_list):
    pairs = [agent._saliency(render([(0, 0), tuple(v)], SPOT_SIGMA))[0] for v in v_list]
    pre = np.stack([masked_fovea(agent, [p], mask_at(agent, (0, 0), 1.0))[0] for p in pairs])
    pre0 = np.stack([masked_fovea(agent, [p], np.ones((80, 80)))[0] for p in pairs])
    pre = pre / np.maximum(np.linalg.norm(pre, axis=1, keepdims=True), 1e-9) * np.linalg.norm(pre0, axis=1, keepdims=True)
    return distances(pre, w_cond).argmin(-1)


def probe_views(agent, sal, d, key, w_cond, n0, readout):
    """Probe views of the conditions map through the filter at key (-1: fixation, k: attention prototype k, NOFILTER: none).
    soft: overlap of each view with every prototype, divided by the norm of the unfiltered probe view and of the prototype, so the
    attenuation by the filter is kept. winner: nearest prototype of the view rescaled to the unfiltered norm; -1 if the view is
    below BLANK_FLOOR of that norm."""
    mask = np.ones((80, 80)) if key == NOFILTER else mask_at(agent, (0, 0) if key < 0 else d[key])
    x = masked_fovea(agent, sal, mask)
    if readout == "soft":
        return (x @ w_cond) / np.maximum(n0, 1e-9)[:, None] / np.linalg.norm(w_cond, axis=0)
    nx = np.linalg.norm(x, axis=1)
    win = distances(x / np.maximum(nx, 1e-9)[:, None] * n0[:, None], w_cond).argmin(-1)
    win[nx < BLANK_FLOOR * n0] = -1
    return win


def response(views, j, kern, readout):
    if readout == "soft":
        r = views @ kern[j]
    else:
        r = kern[j, np.where(views < 0, 0, views)]
        r[views < 0] = 0
    r[~VALID] = 0
    return r


def field_centre(r, j, kern, readout):
    if readout == "soft":
        z = np.where(VALID, r - np.median(r[VALID]), 0)
        return centre(np.where(z >= 0.5 * z.max(), z, 0)) if z.max() > 0 else np.full(2, np.nan)
    return centre(r) if r.max() >= 0.5 * kern[j].max() else np.full(2, np.nan)


def conditions_rows(path, args, agent):
    std = args.std or load_params(path).neighborhood_modulation_baseline
    store = torch.load(os.path.join(path, "off_control_store"), map_location="cpu", weights_only=False)
    w_cond = store["visual_conditions_map_state_dict"]["weights"].numpy().astype(float)
    w_eff = store["visual_effects_map_state_dict"]["weights"].numpy().astype(float)
    d = (store["attention_map_state_dict"]["weights"].numpy().astype(float).T - 0.5) * 80
    kern = kernel(std, w_cond.shape[1])
    v_list = sample_targets(w_cond, args.displays, np.random.RandomState(0))
    sal = probe_saliency(agent, PROBE_SIGMA)
    n0 = np.linalg.norm(masked_fovea(agent, sal, np.ones((80, 80))), axis=1)
    win = recruit(agent, w_cond, v_list)
    near = np.linalg.norm(d[None] - v_list[:, None], axis=2).argmin(1)
    base = dict(run=os.path.basename(path), readout=args.readout, scale=MASK_SD_SCALE, n_displays=len(win),
                n_recruited=len(set(win)), act_err=np.median(np.linalg.norm(d[win] - v_list, axis=1)))
    rows, centres = [], {}
    for state, keys in (("off", np.full(len(win), NOFILTER)), ("before", np.full(len(win), -1)), ("after", near)):
        cache = {k: probe_views(agent, sal, d, k, w_cond, n0, args.readout) for k in np.unique(keys)}
        r = np.stack([response(cache[k], j, kern, args.readout) for k, j in zip(keys, win)])
        r2 = r
        if args.effects:
            cache_e = {k: probe_views(agent, sal, d, k, w_eff, n0, args.readout) for k in np.unique(keys)}
            r2 = np.stack([response(cache_e[k], j, kern, args.readout) for k, j in zip(keys, win)])
        c = np.array([field_centre(ri, j, kern, args.readout) for ri, j in zip(r, win)])
        centres[state] = c
        ok = ~np.isnan(c).any(1) & (np.abs(v_list) <= WIN).all(1)
        pr = place_response_from(r, r2, ok, win, v_list, args.perms, args.boots, np.random.RandomState(2))
        if not args.effects:
            pr = {k: x for k, x in pr.items() if "eff" not in k and "contrast" not in k}
        rows.append(dict(base, state=state, **pr))
    far = (np.linalg.norm(v_list, axis=1) >= FIELD_MIN_V) & ~np.isnan(centres["before"]).any(1) & ~np.isnan(centres["after"]).any(1)
    cb, ca, v = centres["before"][far], centres["after"][far], v_list[far]
    move = dict(n_move=int(far.sum()), b_to_F=np.median(np.linalg.norm(cb, axis=1)), a_to_T=np.median(np.linalg.norm(ca - v, axis=1)),
                closer_T_after=(np.linalg.norm(ca - v, axis=1) < np.linalg.norm(cb - v, axis=1)).mean(),
                F_then_T=((np.linalg.norm(cb, axis=1) < np.linalg.norm(cb - v, axis=1)) & (np.linalg.norm(ca - v, axis=1) < np.linalg.norm(ca, axis=1))).mean())
    for row in rows:
        row.update(move if row["state"] == "after" else {k: np.nan for k in move})
    return rows


def verify_conditions(args, agent):
    global MASK_SD_SCALE
    ax = GX[5:15]
    c = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1).reshape(100, 2)
    kern = kernel(0.3)
    sal = probe_saliency(agent, PROBE_SIGMA)
    n0 = np.linalg.norm(masked_fovea(agent, sal, np.ones((80, 80))), axis=1)
    w = np.stack([fovea(agent, [tuple(c[j])], PROBE_SIGMA) for j in range(100)], 1).astype(float)
    views = probe_views(agent, sal, None, NOFILTER, w, n0, "soft")
    e = np.array([np.linalg.norm(field_centre(response(views, j, kern, "soft"), j, kern, "soft") - c[j]) for j in range(100)])
    inner = np.array([(abs(c[j]) <= ax[-2]).all() for j in range(100)])
    print(f"soft readout: centre within 1 px of c_j in {(e[inner] <= 1).mean():.2f} of {inner.sum()} interior cells")
    ok1 = (e[inner] <= 1).mean() >= 0.95
    pts = np.array([(x, y) for x in range(-6, 7) for y in range(-6, 7)]) * 4.0
    v = pts[np.random.RandomState(0).choice(len(pts), 100, replace=False)]
    w2 = np.stack([fovea(agent, [(0, 0), tuple(v[j])], PROBE_SIGMA) for j in range(100)], 1).astype(float)
    allok = np.ones(100, bool)
    MASK_SD_SCALE = args.mask_sd_scale
    res = {}
    for state, keyf in (("off", lambda j: NOFILTER), ("before", lambda j: -1), ("after", lambda j: j)):
        cache = {}
        r = []
        for j in range(100):
            k = keyf(j)
            if k not in cache:
                cache[k] = probe_views(agent, sal, v, k, w2, n0, "soft")
            r.append(response(cache[k], j, kern, "soft"))
        res[state] = place_response_from(np.stack(r), np.stack(r), allok, np.arange(100), v, 100, 200, np.random.RandomState(2))
        print(f"planted two-spot maps, filter {state:6s}: place 0 {res[state]['pr_cond_0']:.2f}, place v {res[state]['pr_cond_v']:.2f} "
              f"(shuffled {res[state]['pr_cond_v_null']:.2f}, p {res[state]['pr_cond_v_p']:.3f}), place -v {res[state]['pr_cond_mv']:.2f}")
    ok2 = res["off"]["pr_cond_v"] >= 0.9 and res["off"]["pr_cond_v_p"] <= 0.02
    ok3 = (res["before"]["pr_cond_0"] >= 0.9 and res["before"]["pr_cond_v"] <= res["before"]["pr_cond_v_null"] + 0.05
           and res["after"]["pr_cond_v"] >= 0.9 and res["after"]["pr_cond_0"] <= res["before"]["pr_cond_0"] - 0.2)
    print("VERIFY", "PASS" if ok1 and ok2 and ok3 else "FAIL")


def run_row(path, args, agent, table):
    std = args.std or load_params(path).neighborhood_modulation_baseline
    store = torch.load(os.path.join(path, "off_control_store"), map_location="cpu", weights_only=False)
    w = {k: store[f"{k}_map_state_dict"]["weights"].numpy().astype(float)
         for k in ["visual_conditions", "visual_effects", "attention"]}
    d = (w["attention"].T - 0.5) * 80
    v_list = sample_targets(w["visual_conditions"], args.displays, np.random.RandomState(0))
    if args.filter != "off":
        pre, wc, we = masked_views(agent, w, d, v_list, {"on": "split"}.get(args.filter, args.filter))
        post = None
    else:
        pre, post = pair_views(agent, v_list)
        wc = we = None
    row = paradigm(w["visual_conditions"], w["visual_effects"], d, pre, post, v_list, std, table, nperm=args.perms,
                   max_act_err=args.max_act_err, nboot=args.boots, wc=wc, we=we)
    return [dict(run=os.path.basename(path), **row)]


def verify(args, agent, table):
    ax = GX[5:15]
    c = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1).reshape(100, 2)
    w = np.stack([fovea(agent, [tuple(c[j])]) for j in range(100)], 1).astype(float)
    kern = kernel(0.3)
    wc, we = winners(table, w), winners(table, w)
    inner = np.array([(abs(c[j]) <= ax[-2]).all() for j in range(100)])
    e1 = np.array([np.linalg.norm(field(kern, j, wc, (0, 0))[1] - c[j]) for j in range(100)])[inner]
    e2 = np.array([np.linalg.norm(field(kern, j, we, (0, 0))[1] - c[j]) for j in range(100)])[inner]
    ok = (e1 <= 1).mean() >= 0.95 and (e2 <= 1).mean() >= 0.95
    print(f"aligned: c1 within 1 px of c_j {(e1 <= 1).mean():.2f}; c2 within 1 px of c_j {(e2 <= 1).mean():.2f} "
          f"({inner.sum()} interior cells)")
    pts = np.array([(x, y) for x in range(-6, 7) for y in range(-6, 7)]) * 4.0
    v = pts[np.random.RandomState(0).choice(len(pts), 100, replace=False)]
    wcv = np.stack([fovea(agent, [tuple(v[j])]) for j in range(100)], 1).astype(float)
    wev = np.stack([fovea(agent, [tuple(-v[j])]) for j in range(100)], 1).astype(float)
    s = place_stat(winners(table, wcv), winners(table, wev), np.arange(100), v, kern, 200, np.random.RandomState(1))
    ok2 = all(s[f"e{k}"] <= 4 and s[f"index{k}"] >= 0.8 and s[f"p{k}"] <= 0.02 for k in "12")
    print(f"place statistics: {s['n_test']} testable, c1 to v {s['e1']:.2f} px (null {s['null1']:.1f}, index {s['index1']:.2f}, "
          f"p {s['p1']:.3f}), c2 to -v {s['e2']:.2f} px (null {s['null2']:.1f}, index {s['index2']:.2f}, p {s['p2']:.3f})")
    wc2 = np.stack([fovea(agent, [(0, 0), tuple(v[j])]) for j in range(100)], 1).astype(float)
    we2 = np.stack([fovea(agent, [(0, 0), tuple(-v[j])]) for j in range(100)], 1).astype(float)
    pr = place_response(winners(table, wc2), winners(table, we2), np.arange(100), v, kern, 100, 200, np.random.RandomState(2))
    ok3 = pr["pr_cond_v"] >= 0.9 and pr["pr_eff_mv"] >= 0.9 and pr["pr_cond_v_p"] <= 0.02 and pr["pr_eff_mv_p"] <= 0.02
    print(f"place percentiles: conditions at v {pr['pr_cond_v']:.2f} (shuffled {pr['pr_cond_v_null']:.2f}), effects at -v "
          f"{pr['pr_eff_mv']:.2f} (shuffled {pr['pr_eff_mv_null']:.2f}); at 0: conditions {pr['pr_cond_0']:.2f}, effects {pr['pr_eff_0']:.2f}")
    ones = masked_fovea(agent, probe_saliency(agent), np.ones((80, 80)))
    same = np.allclose(ones, table[(S_GRID[:, 1] + LIM).astype(int), (S_GRID[:, 0] + LIM).astype(int)], atol=1e-5)
    print(f"uniform mask reproduces the unmasked probe views: {same}")
    ok4 = same
    if args.filter == "on":
        sal = probe_saliency(agent)
        fix = mask_at(agent, (0, 0))
        wc = distances(masked_fovea(agent, sal, fix), wcv).argmin(-1)
        we = [distances(masked_fovea(agent, sal, mask_at(agent, v[j])), wev).argmin(-1) for j in range(100)]
        pm = place_response(wc, we, np.arange(100), v, kern, 100, 200, np.random.RandomState(2))
        print(f"filtered planted maps: {pm['n_test']} testable; conditions at v {pm['pr_cond_v']:.2f} (shuffled "
              f"{pm['pr_cond_v_null']:.2f}), effects at -v {pm['pr_eff_mv']:.2f} (shuffled {pm['pr_eff_mv_null']:.2f})")
        ok4 = ok4 and pm["pr_cond_v"] >= 0.9 and pm["pr_eff_mv"] >= 0.9 and pm["pr_cond_v_p"] <= 0.02 and pm["pr_eff_mv_p"] <= 0.02
    print("VERIFY", "PASS" if ok and ok2 and ok3 and ok4 else "FAIL")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--agent-run", help="run whose front end draws the views (default: first run of the manifest)")
    ap.add_argument("--manifest")
    ap.add_argument("--out", default=".")
    ap.add_argument("--displays", type=int, default=300)
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--boots", type=int, default=1000)
    ap.add_argument("--filter", choices=["on", "off", "before", "after"], default="off",
                    help="on: views through the attentional filter (fixation for the conditions side, attention prototype nearest to T for the effects probes); before / after: every view through the filter at fixation / at that prototype")
    ap.add_argument("--readout", choices=["soft", "winner", "legacy"], default="soft",
                    help="soft / winner: conditions face only, filter states off, before and after (--filter is ignored); legacy: the earlier readout of both faces")
    ap.add_argument("--effects", action="store_true", help="soft / winner readout: also read the effects face, through the same filter state")
    ap.add_argument("--spot-sigma", type=float, default=SIGMA, help="sigma (px) of the spots F and T (soft / winner readouts)")
    ap.add_argument("--probe-sigma", type=float, default=SIGMA, help="sigma (px) of the lone probe spot S; F and T keep sigma 3")
    ap.add_argument("--mask-sd-scale", type=float, default=1.0, help="factor on the attentional mask sd (analysis only)")
    ap.add_argument("--face-gain", type=float, default=1.0, help="factor on the masked fovea before recruitment and probing (analysis only)")
    ap.add_argument("--std", type=float, help="lattice neighbourhood std (default: the run's neighborhood_modulation_baseline)")
    ap.add_argument("--max-act-err", type=float, help="keep only displays whose recruited cell's saccade is within this many px of v")
    args = ap.parse_args()
    set_device()
    runs = [r["run"] for r in csv.DictReader(open(args.manifest))] if args.manifest else []
    agent = Run(args.agent_run or runs[0]).agent
    global PROBE_SIGMA, MASK_SD_SCALE, FACE_GAIN, SPOT_SIGMA
    SPOT_SIGMA = args.spot_sigma
    MASK_SD_SCALE, FACE_GAIN = args.mask_sd_scale, args.face_gain
    PROBE_SIGMA = args.probe_sigma
    legacy = args.readout == "legacy"
    table = spot_table(agent, args.probe_sigma) if legacy and (args.filter == "off" or args.verify) else None
    if args.verify:
        return verify(args, agent, table) if legacy else verify_conditions(args, agent)
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for path in runs:
        for row in (run_row(path, args, agent, table) if legacy else conditions_rows(path, args, agent)):
            rows.append(row)
            if not legacy:
                print(f"{row['run']:22s} {row['state']:6s} recruited {row['n_recruited']:3d} saccade vs v {row['act_err']:5.1f} px | testable {row['n_test']:3d} | "
                      + "; ".join(f"{k} {row['pr_' + k]:.2f} (shuffled {row['pr_' + k + '_null']:.2f}, p {row['pr_' + k + '_p']:.3f})"
                                  for k in ("cond_0", "cond_v", "cond_mv")) + (f" | field: {row['n_move']} displays, centre {row['b_to_F']:.1f} px from F before, "
                      f"{row['a_to_T']:.1f} px from T after, closer to T after {row['closer_T_after']:.2f}, F then T {row['F_then_T']:.2f}" if row["state"] == "after" else ""), flush=True)
                continue
            print(f"{row['run']:22s} recruited {row['n_recruited']:3d} fields {row['n_fields']:3d} of {row['n_displays']} "
                  f"saccade vs v {row['act_err']:5.1f} px | kept {row['n_kept']:3d} testable {row['n_test']:3d} | c1 to v {row['e1']:5.1f} px "
                  f"(null {row['null1']:5.1f}, index {row['index1']:.2f}, p {row['p1']:.3f}) | c2 to -v {row['e2']:5.1f} px "
                  f"(null {row['null2']:5.1f}, index {row['index2']:.2f}, p {row['p2']:.3f})", flush=True)
            print("   place percentiles (95% CI over cells): " + "; ".join(
                f"{k} {row['pr_' + k]:.2f} [{row['pr_' + k + '_lo']:.2f}, {row['pr_' + k + '_hi']:.2f}]"
                for k in ("cond_0", "cond_v", "cond_mv", "eff_0", "eff_v", "eff_mv", "contrast_mv")), flush=True)
            print("   v-shuffle null (p): " + "; ".join(
                f"{k} {row['pr_' + k + '_null']:.2f} (p {row['pr_' + k + '_p']:.3f})" for k in ("cond_v", "cond_mv", "eff_v", "eff_mv")),
                flush=True)
    with open(os.path.join(args.out, "twospot_" + os.path.splitext(os.path.basename(args.manifest))[0] + ".csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
