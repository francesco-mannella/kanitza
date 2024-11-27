import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .Simulator import (
    Box2DSim as Sim,
    TestPlotter,
    VisualSensor,
)
from importlib import resources



def DefaultRewardFun(observation):
    return 0 

def get_resource(package, module, filename, text=True):
    with resources.path(f'{package}.{module}', filename) as rp:
        return rp.absolute()


class EyeSimEnv(gym.Env):
    """A single VisualField simulator"""

    metadata = {'render_modes': ['human', 'offline'], 'render_fps': 25}

    def __init__(self):

        super(EyeSimEnv, self).__init__()

        self.taskspace_xlim = np.array([0, 80])
        self.taskspace_ylim = np.array([0, 80])
        self.retina_scale = np.array([50, 50])
        self.retina_size = np.array([80, 80])
        self.fovea_scale = np.array([10, 10])
        self.fovea_size = np.array([16, 16])
        self.retina_sim = None
        self.retina_sim_pos = None
        self.world_files = [
                "eyesim_triangle.json",
                "eyesim_square.json",
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
                'RETINA': gym.spaces.Box(
                    0, 255, [*self.retina_size, 3], dtype=np.uint8
                ),
                'FOVEA': gym.spaces.Box(
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


    def init_world(self, world=None):
        if world is not None: self.world = world
        self.world_file =  get_resource('EyeSim', 'models',self.world_files[self.world])
        self.world_dict = Sim.loadWorldJson(self.world_file)

    def set_seed(self, seed=None):
        self.seed = seed
        if self.seed is None:
            self.seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]
        self.rng = np.random.RandomState(self.seed)

    def update_position_and_rotation(self, position=None, rotation=None):
        self.sim.move(angle=rotation, pos=position)

    def get_position_and_rotation(self):

        # Get the first body from the simulation's bodies dictionary
        first_body_name = list(self.sim.bodies.keys())[0]

        # Set the angle and position of the first body
        rotation = self.sim.bodies[first_body_name].transform.angle
        position = np.array(self.sim.bodies[first_body_name].transform.position)

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
        fovea_start = (self.retina_size - self.fovea_size)//2
        fovea_end = self.retina_size  - fovea_start
        fovea = retina[
                fovea_start[0]:fovea_end[0],
                fovea_start[1]:fovea_end[1],
                ]
        self.observation = {
            'RETINA': retina,
            'FOVEA': fovea,
        }
        # compute reward
        reward = 0

        # compute end of task
        done = False

        # other info
        info = dict()

        return self.observation, reward, done, info

    def reset(self,*,seed=None, options=None):
        super().reset(seed=seed)
        
        self.sim = Sim(world_dict=self.world_dict)


        # Generate a random angle between 0 and 2π radians
        angle = self.rng.rand() * 2 * np.pi 

        # Calculate the range of possible x and y positions
        x_range = self.taskspace_xlim[1] - self.taskspace_xlim[0]
        y_range = self.taskspace_ylim[1] - self.taskspace_ylim[0]

        # Calculate a random position within defined central band of task space
        # Position is calculated to be between 40% to 60% of the task space range
        position = np.array([
            self.taskspace_xlim[0] + x_range * (0.4 + 0.2 * self.rng.rand()), # Calculate x position
            self.taskspace_ylim[0] + y_range * (0.4 + 0.2 * self.rng.rand())  # Calculate y position
        ])

        # Get the first body from the simulation's bodies dictionary
        first_body_name = list(self.sim.bodies.keys())[0]

        # Set the angle and position of the first body
        self.sim.bodies[first_body_name].transform.angle = angle
        self.sim.bodies[first_body_name].transform.position = position

        self.retina_sim = VisualSensor(
            self.sim,
            shape=self.retina_size,
            rng=self.retina_scale,
        )
        
        self.retina_sim_pos = np.array([
            self.taskspace_xlim[1]//2,
            self.taskspace_ylim[1]//2,
            ])

        if self.renderer is not None:
            self.renderer.reset()

        observation, reward, done, info  = self.step( np.zeros(2) )

        info["angle"] = angle
        info["position"] = position

        return observation, info

    def render_init(self, mode):
        if self.renderer is not None:
            self.renderer.close()
        if mode == 'human':
            self.renderer = self.rendererType(
                self,
                xlim=self.taskspace_xlim,
                ylim=self.taskspace_ylim,
                figsize=self.renderer_figsize,
            )
        elif mode == 'offline':
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
                mode == 'offline'
                and (self.renderer is None or not self.renderer.offline)
            )
            or (
                mode == 'human'
                and (self.renderer is None or self.renderer.offline)
            )
        ):
            self.render_init(mode)

    def render(self, mode='human'):
        self.render_check(mode)
        if self.renderer is not None:
            self.renderer.step()

