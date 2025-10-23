from .multiagentenv import MultiAgentEnv
from miner.MinerMultiAgentEnv import GameSocket
from miner.bots import BlackPantherBot, DeepMindBot, GreedyBot
import numpy as np
import math
import random


class MinerWrapper(MultiAgentEnv):
    def __init__(
        self,
        key,
        time_limit,
        **kwargs,
    ):
        self.mode = key.split("_")[1]
        self.n_agents = int(key.split("_")[2])
        self.n_enemies = int(key.split("_")[-1])
        self._env = GameSocket(self.n_agents + self.n_enemies)
        self._env.connect()
        self.mapIDs = list(self._env.maps.keys())
    
        self._obs = None
        self._info = None
        self.reset()
        self.avails = np.stack([self._env.get_avails(self._env.bots[id]) for id in range(self.n_agents)])
        self.episode_limit = self._env.maxStep


    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        actions = [int(a) for a in actions]
        self._env.stepCount += 1
        enemy_actions = [bot.get_action(self.n_agents + id) for id, bot in enumerate(self.bots)]
        self._env.step(actions + enemy_actions)

        rewards = []
        for agent in self._env.bots:
            score = agent.score - self.scores.get(agent.playerId, 0)
            self.scores[agent.playerId] = agent.score
            rewards.append(score)
        
        state = self._get_state()
        self._obs = np.stack([self._get_obs(state, id) for id in range(self.n_agents)])
        
        done = self._env.stepCount >= self._env.maxStep
        for user in self._env.bots:
            if user.status != 0:
                done = True
                break
        try:
            self.avails = np.stack([self._env.get_avails(self._env.bots[id]) for id in range(self.n_agents)])
        except:
            done = True
        my_info = {}
        if done:
            ally_score = np.mean([self.scores[id] for id in range(self.n_agents)])
            enemy_score = np.mean([self.scores[self.n_agents+id] for id in range(self.n_enemies)])
            my_info["go_count"] = self._env.stepCount
            my_info["battle_won"] = ally_score > enemy_score
        reward = np.sum(rewards[:self.n_agents])
        return reward, done, my_info

    def get_obs(self):
        return self._obs

    def get_obs_size(self):
        return self._obs.shape[1]

    def get_state(self):
        return np.concatenate(self._obs)

    def get_state_size(self):
        return self._obs.shape[0] * self._obs.shape[1]

    def get_avail_actions(self):
        return self.avails

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.avails.shape[-1]

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        try:
            for bot in self.bots:
                if isinstance(bot, DeepMindBot):
                    bot.heuristic.exit_game()
                    del bot.heuristic
                del bot
        except Exception as e:
            pass
        mapID = self.mapIDs[np.random.randint(0, len(self.mapIDs))]
        posID_x = np.random.randint(21)
        posID_y = np.random.randint(9)
        self._env.reset([mapID, posID_x, posID_y, 50, 100])
        if self.mode == "easy":
            bots = [GreedyBot(self._env), GreedyBot(self._env)]
        elif self.mode == "medium":
            bots = [GreedyBot(self._env), DeepMindBot(self._env)]
        elif self.mode == "hard":
            bots = [BlackPantherBot(self._env), DeepMindBot(self._env)]
        bots = bots * math.ceil(self.n_enemies / len(bots))
        random.shuffle(bots)
        self.bots = bots
        self.scores = {}
        self._env.stepCount = 0
        state = self._get_state()
        self._obs = np.stack([self._get_obs(state, id) for id in range(self.n_agents)])
        self.avails = np.stack([self._env.get_avails(self._env.bots[id]) for id in range(self.n_agents)])
        return self._obs, {}

    def _get_state(self):
        max_x = self._env.userMatch.gameinfo.width
        max_y = self._env.userMatch.gameinfo.height
        data = np.zeros([max_x, max_y])
        for cell in self._env.userMatch.gameinfo.obstacles:
            if cell.type > 0:
                data[cell.posx, cell.posy] = - 1 / cell.type
        for cell in self._env.userMatch.gameinfo.golds:
            data[cell.posx, cell.posy] = cell.amount / 1000
        data = data.flatten()
        return data
    
    def _get_obs(self, state, agent_id=0):
        agent = self._env.bots[agent_id]
        max_x = self._env.userMatch.gameinfo.width
        max_y = self._env.userMatch.gameinfo.height

        all_pos = [agent.posx/max_x, agent.posy/max_y]
        users = [user for user in self._env.bots if user.playerId != agent_id]
        for user in users:
            if user.playerId < self.n_agents:
                all_pos += [user.posx/max_x, user.posy/max_y]
        for user in users:
            if user.playerId >= self.n_agents:
                all_pos += [user.posx/max_x, user.posy/max_y]
        
        feats = np.concatenate((all_pos, state, [agent.energy/self._env.E, self._env.stepCount/self._env.maxStep]))
        self.s_dim = 2 * self.n_agents
        self.e_dim = len(all_pos)
        return feats
    
    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def save_replay(self):
        pass

    def get_stats(self):
        return {}