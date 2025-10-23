from functools import partial
import gym
import numpy as np
from gym.spaces import Box
from gym.wrappers import TimeLimit

from .multiagentenv import MultiAgentEnv
from .obsk import build_obs, get_joints_at_kdist, get_parts_and_edges


# using code from https://github.com/ikostrikov/pytorch-ddpg-naf
class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        action = (action + 1) / 2
        action *= self.action_space.high - self.action_space.low
        action += self.action_space.low
        return action

    def action(self, action_):
        return self._action(action_)

    def _reverse_action(self, action):
        action -= self.action_space.low
        action /= self.action_space.high - self.action_space.low
        action = action * 2 - 1
        return action


class MujocoMulti(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)
        self.scenario = kwargs["env_args"]["scenario"]  # e.g. Ant-v2
        self.agent_conf = kwargs["env_args"]["agent_conf"]  # e.g. '2x3'

        self.agent_partitions, self.mujoco_edges, self.mujoco_globals = (
            get_parts_and_edges(self.scenario, self.agent_conf)
        )

        self.n_agents = len(self.agent_partitions)
        self.n_actions = max([len(l) for l in self.agent_partitions])
        self.obs_add_global_pos = kwargs["env_args"].get("obs_add_global_pos", False)

        self.agent_obsk = kwargs["env_args"].get(
            "agent_obsk", None
        )  # if None, fully observable else k>=0 implies observe nearest k agents or joints
        self.agent_obsk_agents = kwargs["env_args"].get(
            "agent_obsk_agents", False
        )  # observe full k nearest agents (True) or just single joints (False)

        if self.agent_obsk is not None:
            self.k_categories_label = kwargs["env_args"].get("k_categories")
            if self.k_categories_label is None:
                if self.scenario in ["Ant-v2", "manyagent_ant"]:
                    self.k_categories_label = "qpos,qvel,cfrc_ext|qpos"
                elif self.scenario in ["Humanoid-v2", "HumanoidStandup-v2"]:
                    self.k_categories_label = (
                        "qpos,qvel,cfrc_ext,cvel,cinert,qfrc_actuator|qpos"
                    )
                elif self.scenario in ["Reacher-v2"]:
                    self.k_categories_label = "qpos,qvel,fingertip_dist|qpos"
                elif self.scenario in ["coupled_half_cheetah"]:
                    self.k_categories_label = "qpos,qvel,ten_J,ten_length,ten_velocity|"
                else:
                    self.k_categories_label = "qpos,qvel|qpos"

            k_split = self.k_categories_label.split("|")
            self.k_categories = [
                k_split[k if k < len(k_split) else -1].split(",")
                for k in range(self.agent_obsk + 1)
            ]

            self.global_categories_label = kwargs["env_args"].get("global_categories")
            self.global_categories = (
                self.global_categories_label.split(",")
                if self.global_categories_label is not None
                else []
            )

        if self.agent_obsk is not None:
            self.k_dicts = [
                get_joints_at_kdist(
                    agent_id,
                    self.agent_partitions,
                    self.mujoco_edges,
                    k=self.agent_obsk,
                    kagents=False,
                )
                for agent_id in range(self.n_agents)
            ]

        # load scenario from script
        self.episode_limit = self.args.episode_limit

        self.env_version = kwargs["env_args"].get("env_version", 2)
        if self.env_version == 2:
            try:
                self.wrapped_env = NormalizedActions(gym.make(self.scenario))
            except gym.error.Error:  # env not in gym
                raise NotImplementedError("Custom env not implemented!")
                # self.wrapped_env = NormalizedActions(
                #     TimeLimit(
                #         partial(env_REGISTRY[self.scenario], **kwargs["env_args"])(),
                #         max_episode_steps=self.episode_limit,
                #     )
                # )
        else:
            assert False, "not implemented!"
        self.timelimit_env = self.wrapped_env.env
        self.timelimit_env._max_episode_steps = self.episode_limit
        self.env = self.timelimit_env.env
        self.timelimit_env.reset()
        self.obs_size = self.get_obs_size()

        # COMPATIBILITY
        self.n = self.n_agents
        self.observation_space = [
            Box(low=-10, high=10, shape=(self.obs_size,)) for _ in range(self.n_agents)
        ]

        self.share_obs_size = self.get_state_size()
        self.share_observation_space = [
            Box(low=-10, high=10, shape=(self.share_obs_size,))
            for _ in range(self.n_agents)
        ]

        acdims = [len(ap) for ap in self.agent_partitions]
        self.action_space = tuple(
            [
                Box(
                    self.env.action_space.low[sum(acdims[:0]) : sum(acdims[:1])],
                    self.env.action_space.high[sum(acdims[:0]) : sum(acdims[:1])],
                )
                for a in range(self.n_agents)
            ]
        )
        self.true_action_space = tuple(
            [
                Box(
                    self.env.action_space.low[sum(acdims[:a]) : sum(acdims[: a + 1])],
                    self.env.action_space.high[sum(acdims[:a]) : sum(acdims[: a + 1])],
                )
                for a in range(self.n_agents)
            ]
        )

    def step(self, actions):
        env_actions = np.concatenate(
            [
                actions[i][: self.true_action_space[i].low.shape[0]]
                for i in range(self.n_agents)
            ]
        )
        obs_n, reward_n, done_n, info_n = self.wrapped_env.step(env_actions)
        self.steps += 1

        info = {}
        info.update(info_n)

        if done_n:
            if self.steps < self.episode_limit:
                # the next state will be masked out
                info["bad_transition"] = False
            else:
                # the next state will not be masked out
                info["bad_transition"] = True

        # return reward_n, done_n, info
        rewards = [[reward_n]] * self.n_agents
        dones = [done_n] * self.n_agents
        infos = [info for _ in range(self.n_agents)]

        # add absorbing state and normalize observation
        obss = self.get_obs()
        states = self.get_state() 

        # if done_n and self.steps < self.episode_limit:
        #     fake_obs = np.zeros(self.obs_size)
        #     fake_obs[-1] = 1.0
        #     fake_state = np.zeros(self.share_obs_size)
        #     fake_state[-1] = 1.0
        #     obss = [fake_obs for _ in range(self.n_agents)]
        #     states = [fake_state for _ in range(self.n_agents)]

        # norm
        # for i in range(self.n_agents):
        #     assert (obss[i][-1] == 0.0 and states[i][-1] == 0.0) or (obss[i][-1] == 1.0 and states[i][-1] == 1.0 and info["bad_transition"] == False)
        #     obss[i] = (obss[i] - np.mean(obss[i])) / np.std(obss[i])
        #     states[i] = (states[i] - np.mean(states[i])) / np.std(states[i])
            
        return (
            obss,
            states,
            rewards,
            dones,
            infos,
            self.get_avail_actions(),
        )

    def get_obs(self):
        """Returns all agent observat3ions in a list"""
        state = self.env.unwrapped._get_obs()
        obs_n = []
        for a in range(self.n_agents):
            agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
            agent_id_feats[a] = 1.0
            obs_i = np.concatenate([state, agent_id_feats])
            obs_i = (obs_i - np.mean(obs_i)) / np.std(obs_i)  # norm obs
            # obs_i = np.concatenate([obs_i, np.zeros(1)])
            obs_n.append(obs_i)
        return obs_n

    def get_obs_agent(self, agent_id):
        if self.agent_obsk is None:
            return self.env.unwrapped._get_obs()
        else:
            return build_obs(
                self.env,
                self.k_dicts[agent_id],
                self.k_categories,
                self.mujoco_globals,
                self.global_categories,
            )

    def get_obs_size(self):
        """Returns the shape of the observation"""
        if self.agent_obsk is None:
            raise NotImplementedError("Not implemented!")
            return self.get_obs_agent(0).size
        else:
            return len(self.get_obs()[0])

    def get_state(self, team=None):
        # TODO: May want global states for different teams (so cannot see what the other team is communicating e.g.)
        state = self.env.unwrapped._get_obs()
        state = (state - np.mean(state)) / np.std(state) # norm state
        # state = np.concatenate([state, np.zeros(1)])
        share_obs = []
        for a in range(self.n_agents):
            share_obs.append(state)
        return share_obs

    def get_state_size(self):
        """Returns the shape of the state"""
        return len(self.get_state()[0])

    def get_avail_actions(self):  # all actions are always available
        return None

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return None

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return (
            self.n_actions
        )  # CAREFUL! - for continuous dims, this is action space dim rather
        # return self.env.action_space.shape[0]

    def get_stats(self):
        return {}

    # TODO: Temp hack
    def get_agg_stats(self, stats):
        return {}

    def reset(self, **kwargs):
        """Returns initial observations and states"""
        self.steps = 0
        self.timelimit_env.reset()

        # normalize observation
        obss = self.get_obs()
        states = self.get_state()
        # norm
        # for i in range(self.n_agents):
        #     assert obss[i][-1] == 0.0 and states[i][-1] == 0.0
        #     obss[i] = (obss[i] - np.mean(obss[i])) / np.std(obss[i])
        #     states[i] = (states[i] - np.mean(states[i])) / np.std(states[i])
        return obss, states, self.get_avail_actions()

    def render(self, **kwargs):
        self.env.render(**kwargs)

    def close(self):
        pass

    def seed(self, seed):
        self.wrapped_env.seed(seed)

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
            "action_spaces": self.action_space,
            "actions_dtype": np.float32,
            "normalise_actions": False,
        }
        return env_info
