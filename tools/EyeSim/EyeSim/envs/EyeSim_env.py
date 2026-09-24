"""Gymnasium environment: a moving retina over a 2D scene of polygons."""
import os
from importlib import resources

import gymnasium as gym
import numpy as np
from EyeSim.envs.Simulator import Box2DSim as Sim
from EyeSim.envs.Simulator import TestPlotter, VisualSensor
from gymnasium import spaces
from skimage.transform import resize


def DefaultRewardFun(observation):
    """Constant zero reward. Unused."""
    return 0


def get_resource(package, module, filename, text=True):
    """Absolute path of a data file shipped inside a package module."""
    with resources.path(f"{package}.{module}", filename) as rp:
        return rp.absolute()


class EyeSimEnv(gym.Env):
    """A single VisualField simulator.

    The scene (task space taskspace_xlim x taskspace_ylim units, default
    80x80) holds the colored bodies of models/worlds.json; all are parked
    at 1e10 except the one selected by `world` (0 red triangle, 1 blue
    square, 2 green circle), placed at reset. The retina is a retina_size
    px view (default 80x80) of a retina_scale units window (default 80x80)
    centered on retina_sim_pos; the FOVEA observation is its central
    fovea_scale px crop resized to fovea_size px (default 16x16, no
    resizing).

    Observation: {"RETINA": (*retina_size, 3) uint8,
    "FOVEA": (*fovea_size, 3) uint8}.
    Action: (2,) displacement of the retina center in task-space units
    (x, y), clipped so the center stays inside the task space.
    step() returns the gymnasium 5-tuple (observation, reward=0,
    terminated=False, truncated=False, info={}).
    """

    metadata = {"render_modes": ["human", "offline"], "render_fps": 25}

    def __init__(self, render_mode=None, params=None):
        """
        Args:
            render_mode (str, optional): None, "human" or "offline".
            params (Parameters, optional): provides taskspace_xlim,
                taskspace_ylim, retina_scale, retina_size, fovea_scale and
                fovea_size; defaults (80x80 space and retina, fovea_scale
                and fovea_size 16x16) if None.
        """

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if params is None:
            self.taskspace_xlim = np.array([0, 80])
            self.taskspace_ylim = np.array([0, 80])
            self.retina_scale = np.array([80, 80])
            self.retina_size = np.array([80, 80])
            self.fovea_scale = np.array([16, 16])
            self.fovea_size = np.array([16, 16])
        else:
            self.taskspace_xlim = np.array(params.taskspace_xlim)
            self.taskspace_ylim = np.array(params.taskspace_ylim)
            self.retina_scale = np.array(params.retina_scale)
            self.retina_size = np.array(params.retina_size)
            self.fovea_scale = np.array(params.fovea_scale)
            self.fovea_size = np.array(params.fovea_size)

        self.retina_sim = None
        self.retina_sim_pos = None
        self.world_labels = ["triangle", "square", "circle"]
        self.world_files = [
            "worlds.json",
        ]
        self.world_objects = [
            "red_triangle",
            "blue_square",
            "green_circle",
        ]

        self.world = 0

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:

        max_action = np.max(self.retina_scale)
        self.action_space = spaces.Box(-max_action, max_action, [2], dtype=float)

        self.observation_space = gym.spaces.Dict(
            {
                "RETINA": gym.spaces.Box(0, 255, [*self.retina_size, 3], dtype=np.uint8),
                "FOVEA": gym.spaces.Box(0, 255, [*self.fovea_size, 3], dtype=np.uint8),
            }
        )

        self.init_world()
        self.set_seed()

        # Renderer parameters
        self.rendererType = TestPlotter
        self.renderer = None
        self.renderer_figsize = (3, 3)

        self.reset()

    def init_world(self, world=None, object_params=None):
        """Select the object and its pose for the next reset.

        Args:
            world (int, optional): index in world_labels; keeps the current
                one if None.
            object_params (dict, optional): {"pos": [x, y], "rot": rad};
                random pose (40-60% of the task space, any angle) if None.
        """
        if world is not None:
            self.world = world
        self.world_file = get_resource("EyeSim", "models", self.world_files[0])
        self.world_dict = Sim.loadWorldJson(self.world_file)
        self.object_params = object_params

    def set_seed(self, seed=None):
        """Seed self.rng (random seed from os.urandom if None)."""
        self.seed = seed
        if self.seed is None:
            self.seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]
        self.rng = np.random.RandomState(self.seed)

    def update_position_and_rotation(self, position=None, rotation=None, obj=None):
        """Move a body (default: last in z-order) to position / rotation.

        None values leave the corresponding property unchanged.
        """
        self.sim.move(angle=rotation, pos=position, obj=obj)

    def get_center(self, obj_name=None):
        """World center of mass of a body (default: current object)."""

        if obj_name is None:
            obj_name = self.world_objects[self.world]

        center = np.array(self.sim.bodies[obj_name].worldCenter)

        return center

    def get_position_and_rotation(self, obj_name=None):
        """Return (position (2,), angle rad) of a body (default: current
        object)."""

        if obj_name is None:
            obj_name = self.world_objects[self.world]

        # Set the angle and position of the first body
        rotation = self.sim.bodies[obj_name].transform.angle
        position = np.array(self.sim.bodies[obj_name].transform.position)

        return position, rotation

    def step(self, action, position=None, rotation=None):
        """Move the retina, optionally move the last body, render.

        Args:
            action (array-like): (2,) retina displacement (task units).
            position, rotation: optional new pose of the last body in
                z-order (see update_position_and_rotation).

        Returns:
            tuple: (observation, 0, False, False, {}).
        """

        self.retina_sim_pos = self.retina_sim_pos + action

        x_limits = self.taskspace_xlim
        y_limits = self.taskspace_ylim
        limits = np.column_stack((x_limits, y_limits))
        self.retina_sim_pos = np.clip(self.retina_sim_pos, *limits)

        self.update_position_and_rotation(position, rotation)

        self.sim.step()
        retina = self.retina_sim.step(self.retina_sim_pos)
        fovea_start = (self.retina_size - self.fovea_scale) // 2
        fovea_end = fovea_start + self.fovea_scale
        fovea = retina[
            fovea_start[0] : fovea_end[0],
            fovea_start[1] : fovea_end[1],
        ]
        if not np.array_equal(self.fovea_scale, self.fovea_size):
            fovea = resize(
                fovea, (*self.fovea_size, 3), preserve_range=True, anti_aliasing=True
            ).astype(np.uint8)
        self.observation = {
            "RETINA": retina,
            "FOVEA": fovea,
        }
        # compute reward
        reward = 0

        # compute end of task
        terminated = False
        truncated = False

        # other info
        info = dict()

        return self.observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, mode=None):
        """Rebuild the scene, place the current object and center the retina.

        Args:
            seed: forwarded to gym.Env.reset (self.rng is not reseeded).
            mode (str, optional): "human" or "offline" to create a renderer.

        Returns:
            tuple: (observation, info) with info keys "world" (label),
            "angle" (rad) and "position" ((2,) task units).
        """
        super().reset(seed=seed)

        self.sim = Sim(world_dict=self.world_dict)

        # Generate a random angle between 0 and 2π radians
        angle = (
            self.object_params["rot"]
            if self.object_params is not None
            else self.rng.rand() * 2 * np.pi
        )

        # Calculate the range of possible x and y positions
        x_range = self.taskspace_xlim[1] - self.taskspace_xlim[0]
        y_range = self.taskspace_ylim[1] - self.taskspace_ylim[0]

        # Calculate a random position within defined central band of task space
        # Position is calculated to be between 40% to 60% of the task space
        # range
        if self.object_params is not None:
            position = np.array(self.object_params["pos"])
        else:
            position = np.array(
                [
                    self.taskspace_xlim[0]
                    + x_range * (0.4 + 0.2 * self.rng.rand()),  # Calculate x position
                    self.taskspace_ylim[0]
                    + y_range * (0.4 + 0.2 * self.rng.rand()),  # Calculate y position
                ]
            )

        obj_name = self.world_objects[self.world]

        # Set the angle and position of the first body
        self.sim.bodies[obj_name].transform.angle = angle
        self.sim.bodies[obj_name].transform.position = position

        self.retina_sim = VisualSensor(
            self.sim,
            shape=self.retina_size,
            rng=self.retina_scale,
        )

        self.retina_sim_pos = np.array(
            [
                self.taskspace_xlim[1] // 2,
                self.taskspace_ylim[1] // 2,
            ]
        )

        self.render_init(mode)

        if self.renderer is not None:
            self.renderer.reset()

        observation, *_, info = self.step(np.zeros(2))

        info["world"] = self.world_labels[self.world]
        info["angle"] = angle
        info["position"] = position

        return observation, info

    def render_init(self, mode):
        """Close any renderer and create a new one for mode, or none."""
        if self.renderer is not None:
            self.renderer.close()
        if mode == "human":
            self.renderer = self.rendererType(
                self,
                xlim=self.taskspace_xlim,
                ylim=self.taskspace_ylim,
                figsize=self.renderer_figsize,
            )
        elif mode == "offline":
            self.renderer = self.rendererType(
                self,
                xlim=self.taskspace_xlim,
                ylim=self.taskspace_ylim,
                offline=True,
                figsize=self.renderer_figsize,
            )
        else:
            self.renderer = None

    def render_check(self, mode):
        """Recreate the renderer if it does not match mode."""
        if (
            mode is None
            or (
                mode == "offline" and (self.renderer is None or not self.renderer.offline)
            )
            or (mode == "human" and (self.renderer is None or self.renderer.offline))
        ):
            self.render_init(mode)

    def render(self, mode=None):
        """Draw the current scene ("human": window, "offline": frame)."""
        self.render_check(mode)
        if self.renderer is not None:
            self.renderer.step()
