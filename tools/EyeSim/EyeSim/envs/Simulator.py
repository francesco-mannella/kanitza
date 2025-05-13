import matplotlib.pyplot as plt
import numpy as np
from Box2D import b2ContactListener
from matplotlib.patches import Polygon
from matplotlib.path import Path

from EyeSim.envs import JsonToPyBox2D as json2d
from .mkvideo import vidManager


class ContactListener(b2ContactListener):
    def __init__(self, bodies):
        b2ContactListener.__init__(self)
        self.contact_db = {}
        self.bodies = bodies

        for h in bodies.keys():
            for k in bodies.keys():
                self.contact_db[(h, k)] = 0

    def BeginContact(self, contact):
        for name, body in self.bodies.items():
            if body == contact.fixtureA.body:
                bodyA = name
            elif body == contact.fixtureB.body:
                bodyB = name

        self.contact_db[(bodyA, bodyB)] = len(contact.manifold.points)

    def EndContact(self, contact):
        for name, body in self.bodies.items():
            if body == contact.fixtureA.body:
                bodyA = name
            elif body == contact.fixtureB.body:
                bodyB = name

        self.contact_db[(bodyA, bodyB)] = 0

    def PreSolve(self, contact, oldManifold):
        pass

    def PostSolve(self, contact, impulse):
        pass


# ------------------------------------------------------------------------------


class Box2DSim(object):
    """2D physics using box2d and a json conf file"""

    @staticmethod
    def loadWorldJson(world_file):
        jsw = json2d.load_json_data(world_file)
        return jsw

    def __init__(
        self,
        world_file=None,
        world_dict=None,
        dt=1 / 80.0,
        vel_iters=30,
        pos_iters=2,
    ):
        """
        Args:

            - world_file (string): the json file from which all objects are
              created
            - world_dict (dict): the json object from which all objects are
              created
            - dt (float): the amount of time to simulate, this should not vary.
            - pos_iters (int): for the velocity constraint solver.
            - vel_iters (int): for the position constraint solver.

        """
        if world_file is not None:
            world, bodies, joints = json2d.createWorldFromJson(world_file)
        else:
            world, bodies, joints = json2d.createWorldFromJsonObj(world_dict)

        self.contact_listener = ContactListener(bodies)

        self.dt = dt
        self.vel_iters = vel_iters
        self.pos_iters = pos_iters
        self.world = world
        self.world.contactListener = self.contact_listener
        self.bodies = bodies

    def contacts(self, bodyA, bodyB):
        """Read contacts between two parts of the simulation

        Args:

            bodyA (string): the name of the object A
            bodyB (string): the name of the object B

        Returns:

            (int): number of contacts
        """
        c1 = 0
        c2 = 0
        db = self.contact_listener.contact_db
        if (bodyA, bodyB) in db.keys():
            c1 = self.contact_listener.contact_db[(bodyA, bodyB)]
        if (bodyB, bodyA) in db.keys():
            c2 = self.contact_listener.contact_db[(bodyB, bodyA)]

        return c1 + c2

    def move(self, pos=None, angle=None, body=None):
        """translate ans rotate

        Args:

            pos (float, float): the new translation position
            angle (float): the new angle position

        """

        if body is None:
            first = list(self.bodies.keys())[0]
        else:
            first = body
        if pos is not None:
            self.bodies[first].position = pos
        if angle is not None:
            self.bodies[first].angle = angle

    def step(self):
        """A simulation step"""
        self.world.Step(self.dt, self.vel_iters, self.pos_iters)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class VisualSensor:
    """Compute the retina state at each ste of simulation"""

    def __init__(self, sim, shape, rng):
        """
        Args:

            sim (Box2DSim): a simulator object
            shape (int, int): width, height of the retina in pixels
            rng (float, float): x and y range in the task space

        """

        self.shape = list(shape)
        self.n_pixels = self.shape[0] * self.shape[1]

        # make a canvas with coordinates
        x = np.arange(-self.shape[0] // 2, self.shape[0] // 2) + 1
        y = np.arange(-self.shape[1] // 2, self.shape[1] // 2) + 1
        X, Y = np.meshgrid(x, y[::-1])
        self.grid = np.vstack((X.flatten(), Y.flatten())).T
        self.scale = np.array(rng) / shape
        self.radius = np.mean(np.array(rng) / shape)
        self.retina = np.zeros(self.shape + [3])
        self.sim = sim

        self.reset(sim)

    def reset(self, sim):
        self.sim = sim

    def step(self, saccade):
        """Run a single simulator step

        Args:

            sim (Box2DSim): a simulator object
            saccade (float, float): x, y of visual field center

        Returns:

            (np.ndarray): a rescaled retina state
        """

        self.retina *= 0
        for key in self.sim.bodies.keys():
            body = self.sim.bodies[key]
            vercs = np.vstack(body.fixtures[0].shape.vertices)
            vercs = vercs[np.arange(len(vercs)) + [0]]
            data = [body.GetWorldPoint(vercs[x]) for x in range(len(vercs))]
            body_pixels = self.path2pixels(data, saccade)
            if body.color is None:
                body.color = [0.5, 0.5, 0.5]
            color = np.array(body.color)
            body_pixels = body_pixels.reshape(body_pixels.shape + (1,)) * (
                1 - color
            )
            self.retina += body_pixels
        self.retina = np.maximum(0, 1 - (self.retina))
        self.retina = 255 * self.retina
        return self.retina.astype(np.uint8)

    def path2pixels(self, vertices, saccade):

        points = self.grid * self.scale + saccade

        path = Path(vertices)  # make a polygon
        points_in_path = path.contains_points(points, radius=self.radius)
        img = 1.0 * points_in_path.reshape(*self.shape, order="F").T  # pixels

        return img


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class TestPlotter:
    """Plotter of simulations
    Builds a simple matplotlib graphic environment
    and render single steps of the simulation within it

    """

    def __init__(
        self,
        env,
        ax=None,
        xlim=None,
        ylim=None,
        figsize=None,
        offline=False,
        video_frame_duration=200,
    ):
        """
        Args:
            env (Box2DSim): a emulator object

        """

        self.env = env
        self.offline = offline
        self.video_frame_duration = video_frame_duration
        self.xlim = xlim if xlim is not None else env.taskspace_xlim
        self.ylim = ylim if ylim is not None else env.taskspace_ylim

        self.ax = ax
        if ax is None:
            if figsize is None:
                self.fig = plt.figure()
            else:
                self.fig = plt.figure(figsize=figsize)
        else:
            self.fig = ax.get_figure()

        self.reset()

    def close(self, name=None):
        plt.close(self.fig)
        if self.offline and name is not None:
            self.vm.mk_video(name=name, dirname=".")
        self.vm = None

    def reset(self):

        if self.offline:
            self.vm = vidManager(
                self.fig, name="frame", duration=self.video_frame_duration
            )

        if self.ax is None:
            self.ax = self.fig.add_subplot(111, aspect="equal")
        self.ax.clear()
        self.polygons = {}
        for key in self.env.sim.bodies.keys():
            self.polygons[key] = Polygon(
                [[0, 0]],
                ec=self.env.sim.bodies[key].color + [1],
                fc=self.env.sim.bodies[key].color + [1],
                closed=True,
            )

            self.ax.add_artist(self.polygons[key])

        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        if not self.offline:
            self.fig.show()
        else:
            self.ts = 0

    def step(self):
        """Run a single emulator step"""

        for key in self.polygons:
            body = self.env.sim.bodies[key]
            vercs = np.vstack(body.fixtures[0].shape.vertices)
            data = np.vstack(
                [body.GetWorldPoint(vercs[x]) for x in range(len(vercs))]
            )
            self.polygons[key].set_xy(data)

        if not self.offline:
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()
        else:
            self.fig.canvas.draw()
            self.vm.save_frame()
            self.ts += 1
