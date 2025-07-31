import os
from importlib import resources

import gymnasium as gym
import numpy as np
from EyeSim.envs.Simulator import Box2DSim as Sim
from EyeSim.envs.Simulator import TestPlotter, VisualSensor
from gymnasium import spaces


def DefaultRewardFun(observation):
    return 0


def get_resource(package, module, filename, text=True):
    with resources.path(f"{package}.{module}", filename) as rp:
        return rp.absolute()


class EyeSimEnv(gym.Env):
    """A single VisualField simulator"""

    metadata = {"render_modes": ["human", "offline"], "render_fps": 25}

    def __init__(self, render_mode=None, colors=False):

        assert (
            render_mode is None or render_mode in self.metadata["render_modes"]
        )
        self.render_mode = render_mode

        self.taskspace_xlim = np.array([0, 80])
        self.taskspace_ylim = np.array([0, 80])
        self.retina_scale = np.array([70, 70])
        self.retina_size = np.array([80, 80])
        self.fovea_scale = np.array([10, 10])
        self.fovea_size = np.array([16, 16])
        self.retina_sim = None
        self.retina_sim_pos = None
        self.world_labels = ["triangle", "square", "circle"]
        self.world_files = [
            "worlds.json",
        ]
        if colors:
            self.world_objects = [
                "red_triangle",
                "blue_square",
                "green_circle",
            ]
        else:
            self.world_objects = [
                "triangle",
                "square",
                "circle",
            ]

        self.world = 0

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:

        max_action = np.max(self.retina_scale)
        self.action_space = spaces.Box(
            -max_action, max_action, [2], dtype=float
        )

        self.observation_space = gym.spaces.Dict(
            {
                "RETINA": gym.spaces.Box(
                    0, 255, [*self.retina_size, 3], dtype=np.uint8
                ),
                "FOVEA": gym.spaces.Box(
                    0, 255, [*self.fovea_size, 3], dtype=np.uint8
                ),
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
        if world is not None:
            self.world = world
        self.world_file = get_resource(
            "EyeSim", "models", self.world_files[0]
        )
        self.world_dict = Sim.loadWorldJson(self.world_file)
        self.object_params = object_params

    def set_seed(self, seed=None):
        self.seed = seed
        if self.seed is None:
            self.seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]
        self.rng = np.random.RandomState(self.seed)

    def update_position_and_rotation(self, position=None, rotation=None, obj=None):
        self.sim.move(angle=rotation, pos=position, obj=obj)

    def get_center(self, obj_name=None):

        if obj_name is None:
            obj_name = self.world_objects[self.world]

        center = np.array(
            self.sim.bodies[obj_name].worldCenter
        )

        return center

    def get_position_and_rotation(self, obj_name=None):

        if obj_name is None:
            obj_name = self.world_objects[self.world]

        # Set the angle and position of the first body
        rotation = self.sim.bodies[obj_name].transform.angle
        position = np.array(
            self.sim.bodies[obj_name].transform.position
        )

        return position, rotation

    def step(self, action, position=None, rotation=None):

        self.retina_sim_pos = self.retina_sim_pos + action

        x_limits = self.taskspace_xlim
        y_limits = self.taskspace_ylim
        limits = np.column_stack((x_limits, y_limits))
        self.retina_sim_pos = np.clip(self.retina_sim_pos, *limits)

        self.update_position_and_rotation(position, rotation)

        self.sim.step()
        retina = self.retina_sim.step(self.retina_sim_pos)
        fovea_start = (self.retina_size - self.fovea_size) // 2
        fovea_end = self.retina_size - fovea_start
        fovea = retina[
            fovea_start[0] : fovea_end[0],
            fovea_start[1] : fovea_end[1],
        ]
        self.observation = {
            "RETINA": retina,
            "FOVEA": fovea,
        }
        # compute reward
        reward = 0

        # compute end of task
        done = False

        # other info
        info = dict()

        return self.observation, reward, done, info

    def reset(self, *, seed=None, mode=None):
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
                    + x_range
                    * (0.4 + 0.2 * self.rng.rand()),  # Calculate x position
                    self.taskspace_ylim[0]
                    + y_range
                    * (0.4 + 0.2 * self.rng.rand()),  # Calculate y position
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

        observation, reward, done, info = self.step(np.zeros(2))

        info["world"] = self.world_labels[self.world]
        info["angle"] = angle
        info["position"] = position

        return observation, info

    def render_init(self, mode):
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
        if (
            mode is None
            or (
                mode == "offline"
                and (self.renderer is None or not self.renderer.offline)
            )
            or (
                mode == "human"
                and (self.renderer is None or self.renderer.offline)
            )
        ):
            self.render_init(mode)

    def render(self, mode=None):
        self.render_check(mode)
        if self.renderer is not None:
            self.renderer.step()
