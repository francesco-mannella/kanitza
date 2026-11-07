"""Two demo GIFs of the attentional filter on one object.

Usage:
    python scripts/demo_attention_gifs.py OUT_DIR [--world triangle] [--seed 15]
        [--n-saccades 8] [--precision 0.9] [--no-new-front-end]
    python scripts/demo_attention_gifs.py OUT_DIR --run RUN_DIR [--world square]
        [--rot 0] [--n-saccades 12] [--seed 0]

With --run, test_<world>_<rot>.gif replays the trained controller of RUN_DIR (its
off_control_store and final_parameters): top row visual field, fovea, salience
and attentional filter; bottom row conditions map, effects map and the map of
attentional templates, one tile per unit showing that unit's attentional filter.

no_filter.gif: uniform attentional mask (agent.set_parameters(None)); every
saccade goes to a salient point of the retina. with_filter.gif: the filter
moves along a ring of centers, stays put for HOLD frames, and a saccade is
made only when it moves. Both GIFs have the same canvas and panel geometry
(the filter panel of no_filter.gif is empty), so the first three panels
overlap exactly when the two images are placed at the same position and scale.
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import EyeSim  # noqa: E402
import gymnasium as gym  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "src"))
from model.agent import Agent  # noqa: E402
from params import Parameters  # noqa: E402
from plotter import FoveaPlotter, MapsPlotter  # noqa: E402
from exp_common import Run, set_device  # noqa: E402
from fig_slide_trajectories import replay, reseed  # noqa: E402

_ = EyeSim
matplotlib.rcParams.update(
    {
        "savefig.dpi": 140,
        "font.size": 10,
        "axes.edgecolor": "#8a8f98",
        "axes.linewidth": 0.8,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlepad": 6,
        "xtick.color": "#555",
        "ytick.color": "#555",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    }
)

DPI = 140
SIDE, GAP, LEFT, RIGHT, TOP, BOTTOM = 300, 34, 62, 10, 38, 36
WIDTH = LEFT + 4 * SIDE + 3 * GAP + RIGHT
HEIGHT = TOP + SIDE + BOTTOM
ROW_GAP = 74
HEIGHT2 = TOP + SIDE + ROW_GAP + SIDE + 10
HIDDEN = np.array([-100.0, -100.0])
GAMMA = 0.5
FRAME_MS = 600
HOLD = 3


class DemoPlotter(FoveaPlotter):
    """Compact four-panel figure with a fixation cross at the retina center."""

    def __init__(self, env, *args, show_filter=True, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.fig.set_dpi(DPI)
        self.fig.set_size_inches(WIDTH / DPI, HEIGHT / DPI)
        self.fig.subplots_adjust(
            left=LEFT / WIDTH,
            right=1 - RIGHT / WIDTH,
            wspace=GAP / SIDE,
            top=1 - TOP / HEIGHT,
            bottom=BOTTOM / HEIGHT,
        )
        self.highlight_dot.set_sizes([230])
        self.highlight_dot.set_facecolor("none")
        self.highlight_dot.set_edgecolor("#ff2222")
        self.highlight_dot.set_linewidths(2.2)
        self.vm.duration = FRAME_MS
        if not show_filter:
            self.filter_ax.set_visible(False)

    def _set_titles_and_labels(self):
        self.env_ax.set_title("Visual field")
        self.fovea_ax.set_title("Fovea")
        self.saliency_ax.set_title("Salience")
        self.filter_ax.set_title("Attentional filter")
        self.env_ax.set_xticks([0, 40, 80])
        self.env_ax.set_yticks([0, 40, 80])
        self.env_ax.grid(True, color="#d9dce1", lw=0.6, ls=":")
        for ax in (self.fovea_ax, self.saliency_ax, self.filter_ax):
            ax.set_xticks([])
            ax.set_yticks([])

    def _update_rect_positions(self):
        super()._update_rect_positions()
        self.retina_pos.set(ec="#6b7280", lw=1.0, ls="--")
        self.fovea_pos.set(ec="#2563eb", lw=1.6)
        if not hasattr(self, "fixation"):
            (self.fixation,) = self.env_ax.plot(
                [], [], marker="x", ms=9, mew=2.0, color="#111827", zorder=5,
                path_effects=[pe.withStroke(linewidth=4, foreground="white")],
            )
        pos = self.env.retina_sim_pos
        self.fixation.set_data([pos[0]], [pos[1]])


def make_params(precision, new_front_end):
    p = Parameters()
    p.fovea_scale = [64, 64]
    p.fovea_size = [16, 16]
    p.gabor_scales = [1.0]
    p.gabor_orientation_bins = 5
    p.gabor_frequency = 0.09
    p.gabor_phase_offset = -np.pi / 2
    p.gabor_kernel_size = 5
    p.gabor_sigma_y_multiplier = 1.0
    p.gabor_rgb_prop = 10.0
    p.gabor_bright_prop = 0.0
    p.attention_max_variance = 6.0
    p.attention_fixed_variance_prop = 0.3
    p.attention_center_distance_variance_prop = 0.7
    p.attention_center_distance_slope = 2.0
    p.agent_sampling_precision = precision
    if new_front_end:
        p.gabor_corner_gain = 2.0
        p.gabor_border_dilation = 1
        p.gabor_fill_prop = 0.15
    return p


def run(params, world, seed, centers, n_saccades, out):
    env = gym.make("EyeSim/EyeSim-v0", params=params).unwrapped
    env.set_seed(seed)
    agent = Agent(env, seed=seed, focus_params=params)
    world_id = env.world_labels.index(world)
    env.init_world(world=world_id, object_params={"pos": [40.0, 40.0], "rot": 0.5})
    env.reset()
    plotter = DemoPlotter(env, offline=True, show_filter=centers is not None)

    for k in range(n_saccades):
        agent.set_parameters(None if centers is None else centers[k % len(centers)])
        observation, *_ = env.step([0, 0])
        action, saliency, point, fovea = agent.get_action(observation)
        plotter.step(fovea, saliency, point, agent.attentional_mask)
        observation, *_ = env.step(action)
        _, saliency, _, fovea = agent.get_action(observation)
        for _ in range(HOLD if centers is not None else 1):
            plotter.step(fovea, saliency, HIDDEN, agent.attentional_mask)
    plotter.close(out)


def px_rect(col, row):
    left = LEFT + col * (SIDE + GAP)
    top = TOP + row * (SIDE + ROW_GAP)
    return [left / WIDTH, 1 - (top + SIDE) / HEIGHT2, SIDE / WIDTH, SIDE / HEIGHT2]


class TemplateMaps(MapsPlotter):
    """Conditions and effects maps plus the map of attentional templates, on given axes."""

    def __init__(self, env, controller, axes, templates):
        self.controller = controller
        self.env = env
        self.env_height, self.env_width = env.observation_space["RETINA"].shape[:-1]
        self.offline = True
        self.params = controller.params
        self.side = int(np.sqrt(self.params.maps_output_size))
        self.fovea_size = int(env.fovea_size[0])
        self.pad = 0
        self.block = self.fovea_size
        self.att_palette = self._create_palette(True)
        (
            self.visual_conditions_map_ax,
            self.visual_effects_map_ax,
            self.attention_map_ax,
        ) = axes
        self.templates = templates
        self.saccade = None
        self._pointers = {}
        both = (controller.visual_conditions_map.weights, controller.visual_effects_map.weights)
        self.lo = min(float(w.min()) for w in both)
        self.hi = max(float(w.max()) for w in both)
        self._initialize_maps()

    def _normalize_weights(self, weights):
        return ((weights - self.lo) / (self.hi - self.lo)).clip(0, 1) ** GAMMA

    def _initialize_attention_map_traces(self):
        self.attention_map_im = self.attention_map_ax.imshow(
            self._frame_tiles(self.templates), zorder=0
        )
        self.attention_map_focus = plt.Rectangle(
            (1e100, 1e100), self.block, self.block, fill=False, ec="#d00", lw=1.2, zorder=2
        )
        self.attention_map_ax.add_patch(self.attention_map_focus)
        extent = self.side * self.block
        self.attention_map_ax.set_xlim(0, extent)
        self.attention_map_ax.set_ylim(0, extent)

    def _set_axis(self):
        for ax in (
            self.visual_conditions_map_ax,
            self.visual_effects_map_ax,
            self.attention_map_ax,
        ):
            ax.set_axis_off()
            extent = self.side * self.block
            ax.set_xlim(-14, extent + 0.5)
            ax.set_ylim(-14, extent + 0.5)
            style = dict(ls="", color="#ff1f1f", mec="white", mew=1.0, ms=10, zorder=3)
            self._pointers[ax] = (
                ax.plot([], [], marker=">", **style)[0],
                ax.plot([], [], marker="^", **style)[0],
            )
        for focus in (
            self.visual_conditions_map_focus,
            self.visual_effects_map_focus,
            self.attention_map_focus,
        ):
            focus.set(
                lw=2.4,
                ec="#ff1f1f",
                path_effects=[pe.withStroke(linewidth=4.8, foreground="white")],
            )

    def _update_focus(self, saccade_pointer):
        super()._update_focus(saccade_pointer)
        row_marker, col_marker = self._pointers[saccade_pointer.axes]
        if self.saccade is None:
            row_marker.set_data([], [])
            col_marker.set_data([], [])
            return
        row, col = self.saccade.ravel()
        half = self.block / 2 - 0.5
        row_marker.set_data([-7], [row * self.block + half])
        col_marker.set_data([col * self.block + half], [-7])

    def _update_attention_map_weights(self):
        self._update_focus(self.attention_map_focus)


class TestPlotter(DemoPlotter):
    """DemoPlotter with a second row: conditions, effects and attentional templates."""

    def __init__(self, env, controller, templates):
        super().__init__(env, offline=True)
        self.fig.set_size_inches(WIDTH / DPI, HEIGHT2 / DPI)
        for col, ax in enumerate(self.axes):
            ax.set_position(px_rect(col, 0))
        axes = [self.fig.add_axes(px_rect(col, 1)) for col in range(3)]
        self.maps = TemplateMaps(env, controller, axes, templates)
        for ax, title in zip(axes, ("Conditions", "Effects", "Attentional templates")):
            ax.set_title(title)


def attention_templates(run, tile=16):
    """(side * tile, side * tile, 3) tiles: the attentional filter of every unit of the map."""
    env, off, params = run.env, run.off, run.params
    side = int(np.sqrt(params.maps_output_size))
    agent = Agent(env, seed=0, focus_params=params)
    tiles = np.zeros((side, tile, side, tile, 3))
    for a in range(side):
        for b in range(side):
            focus = off.get_saccade_from_representation(
                torch.tensor([[a, b]], dtype=torch.float32)
            )
            agent.set_parameters(focus)
            mask = cv2.resize(
                agent.attentional_mask.astype(np.float32),
                (tile, tile),
                interpolation=cv2.INTER_AREA,
            )
            mask = ((mask - mask.min()) / (mask.max() - mask.min() + 1e-12)) ** 3
            tiles[a, :, b, :] = plt.cm.viridis(mask[::-1])[..., :3]
    return tiles.reshape(side * tile, side * tile, 3)


def run_test(run_dir, world, rot, n_saccades, out, seed=0):
    set_device()
    run = Run(run_dir, seed=seed)
    env, agent, off, p = run.env, run.agent, run.off, run.params
    reseed(run, seed)
    off.reset_states()
    env.init_world(
        world=env.world_labels.index(world), object_params={"pos": [40.0, 40.0], "rot": rot}
    )
    obs, info = env.reset()
    env.info = info
    off.reset_goal_inhibition()
    if p.orienting_saccade:
        agent.set_parameters(None)
        obs, *_ = env.step(agent.get_action(obs)[0])
    plotter = TestPlotter(env, off, attention_templates(run))
    plotter.vm.duration = FRAME_MS // 3

    followed = []
    for t in range(p.saccade_period * n_saccades):
        phase = t % p.saccade_period
        if phase == 0:
            saccade, goal = off.get_action_from_condition(agent.get_fovea(obs))
            followed.append(goal.ravel()[:2])
            agent.set_parameters(saccade)
            plotter.maps.saccade = goal.ravel().astype(int)
        elif phase == 1:
            agent.set_parameters(np.array([0.5, 0.5]))
        action, saliency, point, fovea = agent.get_action(
            obs, exclude_fixation=p.exclude_fixation and phase == 0
        )
        if p.hold_fixation and phase != 0:
            action = np.zeros(p.action_size)
        if phase in (0, 1):
            plotter.maps._update_maps()
            for _ in range(1 if phase == 0 else HOLD):
                plotter.step(fovea, saliency, point if phase == 0 else HIDDEN, agent.attentional_mask)
        obs, *_ = env.step(action)
    plotter.close(out)
    again = replay(Run(run_dir, seed=seed), world, rot, seed)[["goal.y", "goal.x"]].to_numpy()[: len(followed)]
    print("gif follows the trajectory of fig_slide_trajectories.py:", bool(np.array_equal(again, np.array(followed))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--world", default="triangle")
    ap.add_argument("--seed", type=int, help="default 15 for the demos, 0 with --run")
    ap.add_argument("--n-saccades", type=int, default=8)
    ap.add_argument("--precision", type=float, default=0.9)
    ap.add_argument("--no-new-front-end", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--rot", type=float, default=0.0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    run_dir = os.path.abspath(a.run) if a.run else None
    os.chdir(a.out_dir)
    if run_dir:
        run_test(run_dir, a.world, a.rot, a.n_saccades, f"test_{a.world}_{a.rot}", 0 if a.seed is None else a.seed)
        sys.exit()
    a.seed = 15 if a.seed is None else a.seed

    angles = np.linspace(0, 2 * np.pi, a.n_saccades, endpoint=False)
    ring = 0.5 + 0.3 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    params = make_params(a.precision, not a.no_new_front_end)
    run(params, a.world, a.seed, None, a.n_saccades, "no_filter")
    run(params, a.world, a.seed, ring, a.n_saccades, "with_filter")
