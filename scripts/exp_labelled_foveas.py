"""Labelled foveas for the Phase 1 retinotopy analysis (predictive_remapping.md 1.1).

Usage:
    python scripts/exp_labelled_foveas.py --out DIR [--manifest FILE] [--seed N]

Grid set: both objects (triangle, square), 9 rotations (2*pi*k/9), a 9x9 grid
of retina offsets around the object (object fixed at the task-space centre,
retina moved to centre - offset). One file per distinct fovea setup,
foveas_grid_<hash>.npz, shared by all runs with that setup. Labels: world (0
triangle, 1 square), rot_index, rot (rad), offset (2, object minus retina
position after the step, so retina clipping is accounted for).

Experienced set (--manifest): each run is replayed with Run.test on both
objects and the 9 rotations; at every step the fovea and the same labels are
logged, plus the step index and whether it is a decision step. One file per
run, foveas_exp_<run name>.npz.

Rule: the data set is accepted if regenerating the grid set with the same
seed gives identical foveas and labels (printed PASS or FAIL).
"""
import argparse
import csv
import hashlib
import os
import sys

import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from evaluate_runs import PROBE_KEYS, load_params  # noqa: E402
from exp_common import CENTRE, Run, set_device  # noqa: E402
from model.agent import Agent  # noqa: E402

N_ROT = 9
GRID = np.linspace(-20, 20, 9)
WORLDS = ["triangle", "square"]


def setup_hash(params):
    key = repr([getattr(params, k) for k in PROBE_KEYS]).encode()
    return hashlib.md5(key).hexdigest()[:8]


def grid_set(params, seed=0):
    env = gym.make(params.env_name, params=params).unwrapped
    env.set_seed(seed)
    agent = Agent(env, seed=seed, focus_params=params)
    foveas, world, rot_index, offset = [], [], [], []
    for w, label in enumerate(WORLDS):
        world_id = env.world_labels.index(label)
        for k in range(N_ROT):
            for dy in GRID:
                for dx in GRID:
                    env.init_world(world=world_id, object_params={"pos": list(CENTRE), "rot": 2 * np.pi * k / N_ROT})
                    env.reset()
                    obs = env.step(CENTRE - np.array([dx, dy]) - env.retina_sim_pos)[0]
                    foveas.append(np.asarray(agent.get_fovea(obs), dtype=np.float32).ravel() / 255.0)
                    world.append(w)
                    rot_index.append(k)
                    offset.append(CENTRE - env.retina_sim_pos)
    rot_index = np.array(rot_index)
    return dict(foveas=np.array(foveas), world=np.array(world), rot_index=rot_index,
                rot=2 * np.pi * rot_index / N_ROT, offset=np.array(offset, dtype=np.float32))


def experienced_set(path, seed=0):
    run = Run(path, seed=seed)
    rows = {"foveas": [], "world": [], "rot_index": [], "offset": [], "step": [], "decision": []}
    period = run.params.saccade_period
    for w, label in enumerate(WORLDS):
        for k in range(N_ROT):
            def on_step(d, w=w, k=k):
                env = d["run"].env
                rows["foveas"].append(np.asarray(d["run"].agent.get_fovea(d["obs"]), dtype=np.float32).ravel() / 255.0)
                rows["world"].append(w)
                rows["rot_index"].append(k)
                rows["offset"].append(np.asarray(env.info["position"]) - env.retina_sim_pos)
                rows["step"].append(d["t"])
                rows["decision"].append(d["t"] % period == 0)
            run.test(label, pos=list(CENTRE), rot=2 * np.pi * k / N_ROT, on_step=on_step)
    out = {k: np.array(v) for k, v in rows.items()}
    out["rot"] = 2 * np.pi * out["rot_index"] / N_ROT
    out["foveas"] = out["foveas"].astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--run", action="append", default=[], help="run folder (repeatable); added to the manifest runs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_device()
    os.makedirs(args.out, exist_ok=True)
    runs = list(args.run)
    if args.manifest:
        runs += [r["run"] for r in csv.DictReader(open(args.manifest))]
    done = set()
    for path in runs:
        params = load_params(path)
        h = setup_hash(params)
        if h not in done:
            f = os.path.join(args.out, f"foveas_grid_{h}.npz")
            if not os.path.isfile(f):
                data = grid_set(params, args.seed)
                again = grid_set(params, args.seed)
                same = all(np.array_equal(data[k], again[k]) for k in data)
                print(f"grid {h}: {len(data['foveas'])} foveas, regeneration identical: {'PASS' if same else 'FAIL'}")
                np.savez_compressed(f, **data)
            done.add(h)
        f = os.path.join(args.out, f"foveas_exp_{os.path.basename(path)}.npz")
        if not os.path.isfile(f):
            data = experienced_set(path, args.seed)
            np.savez_compressed(f, **data)
            print(f"experienced {os.path.basename(path)}: {len(data['foveas'])} foveas")


if __name__ == "__main__":
    main()
