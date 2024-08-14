# EyeSim 

A simple [gym](http://gym.openai.com/) environment using [pybox2d](https://github.com/pybox2d/pybox2d/wiki/manual) as the physics engine and [matplotlib](https://matplotlib.org/) for graphics.

## Table of contents
* [Install](#install)
* [Basic usage](#basic-usage)

## Install

    pip install -e .

## Basic usage

### One-arm scenario

    import gym
    import TemplateLib

    env = gym.make('EyeSim-v0')

    for t in range(10):  
      env.render()
      observation = env.step(env.action_space.sample())

#### rendering

The two possible values of the argument to be passed to env.render() are:
* "human": open a matplotlib figure and update it at each call.
* "offline": save a frame into a png file at each call. Files are saved into the local folder 'frames'. This  folder is created if it does not exist.


#### Observations

#### Reward


#### Done


#### Info

