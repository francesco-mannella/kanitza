"""Shared harness for the model experiments: load a trained run, replay the test decision loop."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from evaluate_runs import find_runs, load_params  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
from model.agent import Agent  # noqa: E402
from model.offline_controller import OfflineController  # noqa: E402
from params import Parameters  # noqa: E402

CENTRE = np.array([40.0, 40.0])


def set_device():
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")


class Run:
    def __init__(self, path, seed=0, params_file="auto", **overrides):
        self.path = path
        self.seed = seed
        if params_file == "auto":
            params = load_params(path)
        else:
            params = Parameters()
            params.load(os.path.join(path, params_file))
        params.epochs = 1
        params.episodes = 1
        params.saccade_num = 16
        params.saccade_period = 4
        for k, v in overrides.items():
            setattr(params, k, v)
        self.params = params
        torch.manual_seed(seed)
        env = gym.make(params.env_name, params=params).unwrapped
        env.set_seed(seed)
        env.rotation = 0.0
        self.env = env
        self.agent = Agent(env, seed=seed, focus_params=params)
        self.off = OfflineController.load(
            os.path.join(path, "off_control_store"), env, params, seed
        )

    def test(self, world, pos=None, rot=None, start=None, policy="model", rng=None,
             on_decision=None, on_step=None):
        env, agent, off, p = self.env, self.agent, self.off, self.params
        off.reset_states()
        off.goals = {"world": [], "position": [], "angle": [], "saccade_id": [], "goal": []}
        world_id = [i for i, label in enumerate(env.world_labels) if label == world][0]
        env.init_world(
            world=world_id,
            object_params=None if pos is None else {"pos": pos, "rot": rot},
        )
        obs, info = env.reset()
        env.info = info
        if start is not None:
            obs = env.step(CENTRE + np.asarray(start, float) - env.retina_sim_pos)[0]
        off.reset_goal_inhibition()
        if p.orienting_saccade:
            agent.set_parameters(None)
            obs, *_ = env.step(agent.get_action(obs)[0])
        side = int(round(np.sqrt(p.maps_output_size)))
        saccade = None
        for t in range(p.saccade_time * p.saccade_num):
            decision = t % p.saccade_period == 0
            if decision:
                condition = agent.get_fovea(obs)
                if on_decision is not None:
                    on_decision(dict(t=t, obs=obs, condition=condition, run=self))
                if policy == "random":
                    goal = np.array([[rng.randint(side), rng.randint(side)]], dtype=np.float32)
                    saccade = off.get_saccade_from_representation(torch.tensor(goal))
                else:
                    saccade, goal = off.get_action_from_condition(condition)
                    if policy == "salience":
                        saccade = None
                agent.set_parameters(saccade)
                off.goals["world"].append(env.info["world"])
                off.goals["angle"].append(env.info["angle"])
                off.goals["position"].append(env.info["position"])
                off.goals["saccade_id"].append(f"0000-{t:04d}")
                off.goals["goal"].append(goal)
            elif t % p.saccade_period == 1:
                if saccade is not None and not np.array_equal(saccade, np.array([0.5, 0.5])):
                    saccade = np.array([0.5, 0.5])
                    agent.set_parameters(saccade)
            action = agent.get_action(
                obs, exclude_fixation=p.exclude_fixation and decision
            )[0]
            if p.hold_fixation and not decision:
                action = np.zeros(p.action_size)
            obs, *_ = env.step(action)
            if on_step is not None:
                on_step(dict(t=t, obs=obs, run=self))
        return off.goals
