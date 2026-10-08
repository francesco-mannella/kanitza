"""Black-box retinotopy probe on maps with known organisation, then on trained runs."""
import argparse
import os
import sys
import warnings

import matplotlib
import numpy as np
import torch
from scipy.stats import spearmanr

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from evaluate_runs import find_runs, grid_coords, pairwise  # noqa: E402
from probe_retinotopy import activations, spots  # noqa: E402

SIDE = 16
CASES = [("position only", 1.0, 0.0), ("global shapes", 1.0, 1.5)]
LATTICE = grid_coords(100)


def rectangles(n, spread, content, rng, labels=False):
    r = 5 * spread
    cx, cy = SIDE / 2 + r * rng.uniform(-1, 1, (2, n))
    w = 4 * (1 + content * rng.uniform(-0.5, 1.0, (2, n)))
    colour = (1 - 0.7 * content * rng.uniform(0, 1, (n, 3))) * np.exp(
        3 * content * rng.uniform(-1, 1, (n, 1)))
    ax = np.arange(SIDE)

    def cover(c, size):
        return np.clip(np.minimum(ax[None] + 0.5, (c + size / 2)[:, None])
                       - np.maximum(ax[None] - 0.5, (c - size / 2)[:, None]), 0, 1)

    img = cover(cy, w[1])[:, :, None] * cover(cx, w[0])[:, None, :]
    x = (img[..., None] * colour[:, None, None, :]).reshape(n, -1)
    if labels:
        return x, dict(centre=np.stack([cx, cy], 1), width=w.T, colour=colour)
    return x


def train_som(x, rng, epochs=30, sigma0=5.0, sigma1=0.7):
    w = x[rng.choice(len(x), 100, replace=False)].copy()
    for t in range(epochs):
        sigma = sigma0 * (sigma1 / sigma0) ** (t / (epochs - 1))
        d2 = (x**2).sum(1)[:, None] + (w**2).sum(1)[None] - 2 * x @ w.T
        bmu = d2.argmin(1)
        phi = np.exp(-pairwise(LATTICE.astype(float))[bmu] ** 2 / (2 * sigma**2))
        w = (phi.T @ x) / (phi.sum(0)[:, None] + 1e-9)
    return w.T


def soft_rho(x, centres, weights, layout=None):
    if layout is not None:
        x = x.reshape(len(x), SIDE * SIDE, 3)[:, layout].reshape(len(x), -1)
    a = activations(x, weights)["soft"]
    rf = (a.T @ centres) / a.sum(0)[:, None]
    iu = np.triu_indices(len(rf), 1)
    return spearmanr(pairwise(LATTICE.astype(float))[iu], pairwise(rf)[iu])[0], rf, a


def blackbox(weights, args, rng, sigma=3.0):
    amp = np.percentile(weights, 99)
    x, centres = spots(SIDE, sigma, args.n_probes, amp, rng)
    rho, rf, a = soft_rho(x, centres, weights)
    null = np.array([soft_rho(x, centres, weights, rng.permutation(SIDE * SIDE))[0]
                     for _ in range(args.shuffles)])
    iu = np.triu_indices(len(rf), 1)
    grid = pairwise(LATTICE.astype(float))[iu]
    z_units = [spearmanr(grid, pairwise(rf[rng.permutation(len(rf))])[iu])[0]
               for _ in range(100)]
    return dict(
        rho=rho, z=(rho - np.mean(z_units)) / np.std(z_units),
        gain=rho - null.mean(), p=(1 + (null >= rho).sum()) / (len(null) + 1),
        range=float(((rf.max(0) - rf.min(0)) / (SIDE - 1)).mean()),
        pr=float((a.sum(1) ** 2 / (a**2).sum(1)).mean()), null=null, rf=rf,
    )


COLORS = {"position only": "#0072B2", "global shapes": "#E69F00", "project maps": "#009E73"}


def group_of(kind):
    return kind if kind in COLORS else "project maps"


def plot(rows, folder):
    os.makedirs(folder, exist_ok=True)
    groups = list(COLORS)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    for ax, (key, label) in zip(axes, [("rho", r"Spearman $\rho$"), ("z", "z (unit permutation)"),
                                      ("gain", r"gain over shuffled layout"),
                                      ("range", "RF-centre range (fovea fraction)")]):
        rng = np.random.RandomState(0)
        for i, g in enumerate(groups):
            v = np.array([r[key] for r in rows if group_of(r["kind"]) == g])
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=18, color=COLORS[g], alpha=0.8)
            ax.hlines(v.mean(), i - 0.25, i + 0.25, color="black", lw=1.2)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(["position\nonly", "global\nshapes", "project\nmaps"], fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].axhline(0, color="gray", lw=0.6, ls="--")
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "retinotopy_probe_stats.pdf"))
    plt.close(fig)

    project = sorted((r for r in rows if group_of(r["kind"]) == "project maps"),
                     key=lambda r: r["gain"])
    picks = [("position only", next(r for r in rows if r["kind"] == "position only")),
             ("global shapes", next(r for r in rows if r["kind"] == "global shapes")),
             ("project maps", project[len(project) // 2] if project else None)]
    picks = [(g, r) for g, r in picks if r is not None]
    fig, axes = plt.subplots(1, len(picks), figsize=(3.6 * len(picks), 2.8), sharey=True)
    for ax, (g, r) in zip(np.atleast_1d(axes), picks):
        ax.hist(r["null"], bins=15, color="lightgray", edgecolor="gray")
        ax.axvline(r["rho"], color=COLORS[g], lw=2)
        ax.set_title(f"{g}: rho {r['rho']:.2f}, p {r['p']:.3f}", fontsize=9)
        ax.set_xlabel(r"$\rho$ with shuffled pixel layouts", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("shuffles", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "retinotopy_probe_null.pdf"))
    plt.close(fig)

    fig, axes = plt.subplots(2, len(picks), figsize=(3.2 * len(picks), 5.6))
    for j, (g, r) in enumerate(picks):
        for i, name in enumerate(["x", "y"]):
            ax = axes[i, j]
            im = ax.imshow(r["rf"][:, i].reshape(10, 10), cmap="viridis")
            fig.colorbar(im, ax=ax, fraction=0.046, label="pixel" if j == len(picks) - 1 else "")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{g}: RF centre {name}" if i == 0 else f"RF centre {name}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "retinotopy_probe_maps.pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=(
        "Does a black-box retinotopy probe find retinotopy in maps organised over global "
        "shapes? Rules, formulated after a first run (post hoc). R1: every run and visual "
        "map shows significant retinotopy by the spot-probe protocol (spot probe, soft "
        "readout, unit-permutation z > 3). R2: that order is spatial, beyond prototype "
        "topography: soft rho exceeds the 95th percentile of rho under shuffled pixel "
        "layouts. R3: it is local, not global: receptive-field centres span at most half "
        "the fovea."))
    parser.add_argument("paths", nargs="*", help="Run folders of trained models.")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-probes", type=int, default=2000)
    parser.add_argument("--shuffles", type=int, default=100)
    parser.add_argument("--csv", help="Write per-map rows to this CSV file.")
    parser.add_argument("--plot", help="Write the figures to this folder.")
    args = parser.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)

    rows = []
    for label, spread, content in CASES:
        for seed in range(args.seeds):
            rng = np.random.RandomState(1000 * seed + int(100 * content) + int(10 * spread))
            x = rectangles(args.n_train, spread, content, rng)
            rows.append(dict(kind=label, seed=seed, **blackbox(train_som(x, rng), args, rng)))
    runs = find_runs(args.paths) if args.paths else []
    for run in runs:
        store = torch.load(os.path.join(run, "off_control_store"), map_location="cpu",
                           weights_only=False)
        for name, short in [("visual_conditions", "vc"), ("visual_effects", "ve")]:
            weights = store[f"{name}_map_state_dict"]["weights"].numpy().astype(float)
            rows.append(dict(kind=f"{os.path.basename(run)} {short}", seed=0,
                             **blackbox(weights, args, np.random.RandomState(0))))

    print("Spot-probe protocol (spot probe, soft readout, width 3); gain and p are against "
          "shuffled pixel layouts")
    print(f"{'map':40s} {'rho':>6s} {'z':>6s} {'gain':>6s} {'p':>6s} {'range':>6s} {'pr':>6s}")
    for label, _, _ in CASES:
        s = [r for r in rows if r["kind"] == label]
        m = lambda k: np.mean([r[k] for r in s])  # noqa: E731
        print(f"{label + f' (mean of {len(s)})':40s} {m('rho'):6.2f} {m('z'):6.1f} "
              f"{m('gain'):6.2f} {np.mean([r['p'] < 0.05 for r in s]):6.2f} "
              f"{m('range'):6.2f} {m('pr'):6.1f}")
    trained = [r for r in rows if r["kind"] not in [c[0] for c in CASES]]
    for r in trained:
        print(f"{r['kind']:40s} {r['rho']:6.2f} {r['z']:6.1f} {r['gain']:6.2f} "
              f"{r['p']:6.3f} {r['range']:6.2f} {r['pr']:6.1f}")
    if trained:
        for name, test in [("R1 retinotopy found (z > 3)", lambda r: r["z"] > 3),
                           ("R2 beyond prototype topography (p < 0.05)", lambda r: r["p"] < 0.05),
                           ("R3 local, range <= 0.5", lambda r: r["range"] <= 0.5)]:
            n = sum(test(r) for r in trained)
            print(f"{name}: {n}/{len(trained)} maps; {'PASS' if n == len(trained) else 'FAIL'}")

    if args.csv:
        columns = [c for c in rows[0] if c not in ("null", "rf")]
        with open(args.csv, "w") as f:
            f.write(",".join(columns) + "\n")
            for r in rows:
                f.write(",".join(str(r[c]) for c in columns) + "\n")

    if args.plot:
        plot(rows, args.plot)


if __name__ == "__main__":
    main()
