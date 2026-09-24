# EyeSim 

A simple [gym](http://gym.openai.com/) environment using [pybox2d](https://github.com/pybox2d/pybox2d/wiki/manual) as the physics engine and [matplotlib](https://matplotlib.org/) for graphics.

## Table of contents
* [Install](#install)
* [Basic usage](#basic-usage)

## Install

    pip install -e .

## Basic usage

    import gymnasium as gym
    import EyeSim

    env = gym.make("EyeSim/EyeSim-v0").unwrapped  # or params=<Parameters>
    env.set_seed(0)
    env.init_world(world=0, object_params={"pos": [40.0, 40.0], "rot": 0.5})
    observation, info = env.reset()

    for t in range(10):
        observation, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        env.render(mode="human")

`world` indexes `env.world_labels` (`["triangle", "square", "circle"]`).
Without `object_params` the object gets a random position (40-60% of the
task space) and rotation.

#### rendering

The two possible values of the argument to be passed to env.render() are:
* "human": open a matplotlib figure and update it at each call.
* "offline": store a frame in memory at each call; `env.renderer.close(name)` saves them as `<name>.gif`.


#### Observations

A dict with "RETINA", a (*retina_size, 3) uint8 view of a retina_scale window of the task space centered on the eye position (default (80, 80, 3) over 80x80 units), and "FOVEA", its central (*fovea_size, 3) crop (default (16, 16, 3)). The sizes come from the optional `params` object passed to `gym.make` (taskspace_xlim/ylim, retina_scale, retina_size, fovea_scale, fovea_size).

#### Actions

A (2,) displacement of the eye position (x, y) in task-space units; the position is clipped to the task space.

#### Reward

Always 0.


#### Done

`terminated` and `truncated` are always False.


#### Info

`step` returns an empty dict; `reset` returns "world" (label), "angle" (rad) and "position" ((2,) task units).

