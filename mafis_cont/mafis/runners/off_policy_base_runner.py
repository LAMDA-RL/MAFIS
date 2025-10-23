"""Base runner for off-policy algorithms."""
import os
import time
import torch
import numpy as np
import setproctitle
from torch.distributions import Categorical
from mafis.utils.envs_tools import (
    make_eval_env,
    make_train_env,
    make_render_env,
    set_seed,
    get_num_agents,
)
from mafis.utils.models_tools import init_device
from mafis.utils.configs_tools import init_dir, save_config, get_task_name
from mafis.algorithms.critics import CRITIC_REGISTRY
from mafis.common.buffers.off_policy_buffer_ep import OffPolicyBufferEP
from mafis.algorithms.critics.soft_twin_continuous_q_critic import langevin_mcmc_s_a


class OffPolicyBaseRunner:
    """Base runner for off-policy algorithms."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the OffPolicyBaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        if "policy_freq" in self.algo_args["algo"]:
            self.policy_freq = self.algo_args["algo"]["policy_freq"]
        else:
            self.policy_freq = 1

        self.state_type = env_args.get("state_type", "EP")

        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        self.task_name = get_task_name(args["env"], env_args)
        if not self.algo_args["render"]["use_render"]:
            self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args["env"],
                env_args,
                args["algo"],
                args["exp_name"],
                algo_args["seed"]["seed"],
                logger_path=algo_args["logger"]["log_dir"],
            )
            save_config(args, algo_args, env_args, self.run_dir)
            self.log_file = open(
                os.path.join(self.run_dir, "progress.txt"), "w", encoding="utf-8"
            )
        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )

        # env
        if self.algo_args["render"]["use_render"]:  # make envs for rendering
            (
                self.envs,
                self.manual_render,
                self.manual_expand_dims,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(args["env"], algo_args["seed"]["seed"], env_args)
        else:  # make envs for training and evaluation
            self.envs = make_train_env(
                args["env"],
                algo_args["seed"]["seed"],
                algo_args["train"]["n_rollout_threads"],
                env_args,
            )
            self.eval_envs = (
                make_eval_env(
                    args["env"],
                    algo_args["seed"]["seed"],
                    algo_args["eval"]["n_eval_rollout_threads"],
                    env_args,
                )
                if algo_args["eval"]["use_eval"]
                else None
            )
        self.num_agents = get_num_agents(args["env"], env_args, self.envs)
        self.agent_deaths = np.zeros(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1)
        )

        self.action_spaces = self.envs.action_space
        for agent_id in range(self.num_agents):
            self.action_spaces[agent_id].seed(algo_args["seed"]["seed"] + agent_id + 1)

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)


        if not self.algo_args["render"]["use_render"]:
            self.critic = CRITIC_REGISTRY["hasac"](
                {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                self.envs.share_observation_space[0],
                self.envs.observation_space[0],
                self.envs.action_space,
                self.num_agents,
                self.state_type,
                device=self.device,
            )
            if self.state_type == "EP":
                self.buffer = OffPolicyBufferEP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    self.envs.share_observation_space[0],
                    self.num_agents,
                    self.envs.observation_space,
                    self.envs.action_space,
                )
                if self.args["algo"] == "mafis":
                    self.demo_buffer = OffPolicyBufferEP(
                        {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                        self.envs.share_observation_space[0],
                        self.num_agents,
                        self.envs.observation_space,
                        self.envs.action_space,
                    )
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        self.value_normalizer = None

        if self.algo_args["train"]["model_dir"] is not None:
            self.restore()

        self.total_it = 0  # total iteration


    def run(self):
        """Run the training (or rendering) pipeline."""
        self.train_episode_rewards = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_rewards = []
        # warmup
        print("start warmup")
        obs, share_obs, available_actions = self.warmup()
        print("finish warmup, start training")
        # train and eval
        steps = (
            self.algo_args["train"]["num_env_steps"]
            // self.algo_args["train"]["n_rollout_threads"]
        )
        update_num = int(  # update number per train
            self.algo_args["train"]["update_per_train"]
            * self.algo_args["train"]["train_interval"]
        )
        self.eval(1)
        for step in range(1, steps + 1):
            actions = self.get_actions(
                obs, share_obs, available_actions=available_actions, add_random=True
            )
            (
                new_obs,
                new_share_obs,
                rewards,
                dones,
                infos,
                new_available_actions,
            ) = self.envs.step(
                actions
            )  # rewards: (n_threads, n_agents, 1); dones: (n_threads, n_agents)
            # available_actions: (n_threads, ) of None or (n_threads, n_agents, action_number)
            next_obs = new_obs.copy()
            next_share_obs = new_share_obs.copy()
            next_available_actions = new_available_actions.copy()
            data = (
                share_obs,
                obs.transpose(1, 0, 2),
                actions.transpose(1, 0, 2),
                available_actions.transpose(1, 0, 2)
                if len(np.array(available_actions).shape) == 3
                else None,
                rewards,
                dones,
                infos,
                next_share_obs,
                next_obs,
                next_available_actions.transpose(1, 0, 2)
                if len(np.array(available_actions).shape) == 3
                else None,
            )
            self.insert(data)
            obs = new_obs
            share_obs = new_share_obs
            available_actions = new_available_actions
            if step % self.algo_args["train"]["train_interval"] == 0:
                if self.algo_args["train"]["use_linear_lr_decay"]:
                    self.critic.lr_decay(step, steps)
                for _ in range(update_num):
                    if self.args["algo"] == "mafis":
                        critic_loss = self.train_MAFIS()
                    else:
                        raise NotImplementedError
                if self.args["algo"] == "mafis":
                    self.writter.add_scalar("loss", critic_loss[0], step)
                    self.writter.add_scalar("critic_loss/loss1", critic_loss[1], step)
                    self.writter.add_scalar("critic_loss/loss2", critic_loss[2], step)
                    self.writter.add_scalar("grad_loss/loss1", critic_loss[3], step)
                    self.writter.add_scalar("grad_loss/loss2", critic_loss[4], step)
                    self.writter.add_scalar("l2_loss", critic_loss[5], step)
            if step % self.algo_args["train"]["eval_interval"] == 0:
                cur_step = (
                    self.algo_args["train"]["warmup_steps"]
                    + step * self.algo_args["train"]["n_rollout_threads"]
                )
                if self.algo_args["eval"]["use_eval"]:
                    print(
                        f"Env {self.args['env']} Task {self.task_name} Algo {self.args['algo']} Exp {self.args['exp_name']} Evaluation at step {cur_step} / {self.algo_args['train']['num_env_steps']}:"
                    )
                    self.eval(cur_step)
                else:
                    print(
                        f"Env {self.args['env']} Task {self.task_name} Algo {self.args['algo']} Exp {self.args['exp_name']} Step {cur_step} / {self.algo_args['train']['num_env_steps']}, average step reward in buffer: {self.buffer.get_mean_rewards()}.\n"
                    )
                    if len(self.done_episodes_rewards) > 0:
                        aver_episode_rewards = np.mean(self.done_episodes_rewards)
                        print(
                            "Some episodes done, average episode reward is {}.\n".format(
                                aver_episode_rewards
                            )
                        )
                        self.log_file.write(
                            ",".join(map(str, [cur_step, aver_episode_rewards])) + "\n"
                        )
                        self.log_file.flush()
                        self.done_episodes_rewards = []
                self.save()

    def warmup(self):
        """Warmup the replay buffer with random actions"""
        warmup_steps = (
            self.algo_args["train"]["warmup_steps"]
            // self.algo_args["train"]["n_rollout_threads"]
        )
        # obs: (n_threads, n_agents, dim)
        # share_obs: (n_threads, n_agents, dim)
        # available_actions: (threads, n_agents, dim)
        obs, share_obs, available_actions = self.envs.reset()
        for _ in range(warmup_steps):
            # action: (n_threads, n_agents, dim)
            actions = self.sample_actions(available_actions)
            (
                new_obs,
                new_share_obs,
                rewards,
                dones,
                infos,
                new_available_actions,
            ) = self.envs.step(actions)
            next_obs = new_obs.copy()
            next_share_obs = new_share_obs.copy()
            next_available_actions = new_available_actions.copy()
            data = (
                share_obs,
                obs.transpose(1, 0, 2),
                actions.transpose(1, 0, 2),
                available_actions.transpose(1, 0, 2)
                if len(np.array(available_actions).shape) == 3
                else None,
                rewards,
                dones,
                infos,
                next_share_obs,
                next_obs,
                next_available_actions.transpose(1, 0, 2)
                if len(np.array(available_actions).shape) == 3
                else None,
            )
            self.insert(data)
            obs = new_obs
            share_obs = new_share_obs
            available_actions = new_available_actions
            
        if self.args["algo"] == "mafis":
            import h5py
            env_name = self.env_args["scenario"]
            hdFile_r = h5py.File(f"{env_name}.h5", 'r')
            demo_size = 20000
            off_obs = np.array(hdFile_r.get("obs"))[:demo_size]
            off_actions = np.array(hdFile_r.get("actions"))[:demo_size]
            off_rewards = np.array(hdFile_r.get("rewards"))[:demo_size]
            off_share_obs = np.array(hdFile_r.get("share_obs"))[:demo_size]
            off_term = np.array(hdFile_r.get("terminated"))[:demo_size]
            off_next_obs = np.array(hdFile_r.get("next_obs"))[:demo_size]
            off_next_share_obs = np.array(hdFile_r.get("next_share_obs"))[:demo_size]
            
            data = (
                off_share_obs,
                off_obs.transpose(1, 0, 2),
                off_actions.transpose(1, 0, 2),
                None,
                off_rewards,
                off_term,
                None,
                off_next_share_obs,
                off_next_obs,
                None,
            )
            self.insert_demo(data)

        return obs, share_obs, available_actions

    def insert(self, data):
        (
            share_obs,  # (n_threads, n_agents, share_obs_dim)
            obs,  # (n_agents, n_threads, obs_dim)
            actions,  # (n_agents, n_threads, action_dim)
            available_actions,  # None or (n_agents, n_threads, action_number)
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            infos,  # type: list, shape: (n_threads, n_agents)
            next_share_obs,  # (n_threads, n_agents, next_share_obs_dim)
            next_obs,  # (n_threads, n_agents, next_obs_dim)
            next_available_actions,  # None or (n_agents, n_threads, next_action_number)
        ) = data

        dones_env = np.all(dones, axis=1)  # if all agents are done, then env is done
        reward_env = np.mean(rewards, axis=1).flatten()
        self.train_episode_rewards += reward_env

        # valid_transition denotes whether each transition is valid or not (invalid if corresponding agent is dead)
        # shape: (n_threads, n_agents, 1)
        valid_transitions = 1 - self.agent_deaths

        self.agent_deaths = np.expand_dims(dones, axis=-1)

        # terms use False to denote truncation and True to denote termination
        if self.state_type == "EP":
            terms = np.full((self.algo_args["train"]["n_rollout_threads"], 1), False)
            for i in range(self.algo_args["train"]["n_rollout_threads"]):
                if dones_env[i]:
                    if not (
                        "bad_transition" in infos[i][0].keys()
                        and infos[i][0]["bad_transition"] == True
                    ):
                        terms[i][0] = True

        for i in range(self.algo_args["train"]["n_rollout_threads"]):
            if dones_env[i]:
                self.done_episodes_rewards.append(self.train_episode_rewards[i])
                self.train_episode_rewards[i] = 0
                self.agent_deaths = np.zeros(
                    (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1)
                )
                if "original_obs" in infos[i][0]:
                    next_obs[i] = infos[i][0]["original_obs"].copy()
                if "original_state" in infos[i][0]:
                    next_share_obs[i] = infos[i][0]["original_state"].copy()
        

        # add absorbing state by setting zero state 
        for i in range(self.algo_args["train"]["n_rollout_threads"]):
            if self.args["algo"] == "mafis" and terms[i][0]:
                next_share_obs[i] = np.zeros_like(next_share_obs[i])
                next_obs[i] = np.zeros_like(next_obs[i])
                share_obs_dim = next_share_obs.shape[-1]
                obs_dim = next_obs.shape[-1]
                action_dim = actions.shape[-1]
                n_agents = next_obs.shape[1]
                
                absorb_num = 10

                fake_share_obs = np.zeros((absorb_num, share_obs_dim))
                fake_obs = np.zeros((n_agents, absorb_num, obs_dim))
                fake_actions = np.zeros((n_agents, absorb_num, action_dim))
                fake_rewards = np.zeros((absorb_num, 1))
                fake_data = (
                    fake_share_obs,
                    fake_obs,
                    fake_actions,
                    None,
                    fake_rewards,
                    np.zeros((absorb_num, 1)),
                    np.zeros((n_agents, absorb_num, 1)),
                    np.zeros((absorb_num, 1)),
                    fake_share_obs,
                    fake_obs,
                    None
                )
                self.buffer.insert(fake_data)

        if self.state_type == "EP":
            data = (
                share_obs[:, 0],  # (n_threads, share_obs_dim)
                obs,  # (n_agents, n_threads, obs_dim)
                actions,  # (n_agents, n_threads, action_dim)
                available_actions,  # None or (n_agents, n_threads, action_number)
                rewards[:, 0],  # (n_threads, 1)
                np.expand_dims(dones_env, axis=-1),  # (n_threads, 1)
                valid_transitions.transpose(1, 0, 2),  # (n_agents, n_threads, 1)
                terms,  # (n_threads, 1)
                next_share_obs[:, 0],  # (n_threads, next_share_obs_dim)
                next_obs.transpose(1, 0, 2),  # (n_agents, n_threads, next_obs_dim)
                next_available_actions,  # None or (n_agents, n_threads, next_action_number)
            )

        self.buffer.insert(data)

    def insert_demo(self, data):
        (
            share_obs,  # (n_threads, n_agents, share_obs_dim)
            obs,  # (n_agents, n_threads, obs_dim)
            actions,  # (n_agents, n_threads, action_dim)
            available_actions,  # None or (n_agents, n_threads, action_number)
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            infos,  # type: list, shape: (n_threads, n_agents)
            next_share_obs,  # (n_threads, n_agents, next_share_obs_dim)
            next_obs,  # (n_threads, n_agents, next_obs_dim)
            next_available_actions,  # None or (n_agents, n_threads, next_action_number)
        ) = data

        dones_env = np.all(dones, axis=1)  # if all agents are done, then env is done
        reward_env = np.mean(rewards, axis=1).flatten()
    
        terms = dones_env.reshape(-1, 1)
        valid_transitions = np.ones_like(rewards)
        data = (
                share_obs[:, 0],  # (n_threads, share_obs_dim)
                obs,  # (n_agents, n_threads, obs_dim)
                actions,  # (n_agents, n_threads, action_dim)
                available_actions,  # None or (n_agents, n_threads, action_number)
                rewards[:, 0],  # (n_threads, 1)
                np.expand_dims(dones_env, axis=-1),  # (n_threads, 1)
                valid_transitions.transpose(1, 0, 2),  # (n_agents, n_threads, 1)
                terms,  # (n_threads, 1)
                next_share_obs[:, 0],  # (n_threads, next_share_obs_dim)
                next_obs.transpose(1, 0, 2),  # (n_agents, n_threads, next_obs_dim)
                next_available_actions,  # None or (n_agents, n_threads, next_action_number)
            )
        self.demo_buffer.insert(data)

    def sample_actions(self, available_actions=None):
        """Sample random actions for warmup.
        Args:
            available_actions: (np.ndarray) denotes which actions are available to agent (if None, all actions available),
                                 shape is (n_threads, n_agents, action_number) or (n_threads, ) of None
        Returns:
            actions: (np.ndarray) sampled actions, shape is (n_threads, n_agents, dim)
        """
        actions = []
        for agent_id in range(self.num_agents):
            action = []
            for thread in range(self.algo_args["train"]["n_rollout_threads"]):
                if available_actions[thread] is None:
                    action.append(self.action_spaces[agent_id].sample())
                else:
                    action.append(
                        Categorical(
                            torch.tensor(available_actions[thread, agent_id, :])
                        ).sample()
                    )
            actions.append(action)
        if self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
            return np.expand_dims(np.array(actions).transpose(1, 0), axis=-1)

        return np.array(actions).transpose(1, 0, 2)

    def get_actions(self, obs, share_obs, available_actions=None, add_random=True):
        """Get actions for rollout.
        Args:
            obs: (np.ndarray) input observation, shape is (n_threads, n_agents, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent (if None, all actions available),
                                 shape is (n_threads, n_agents, action_number) or (n_threads, ) of None
            add_random: (bool) whether to add randomness
        Returns:
            actions: (np.ndarray) agent actions, shape is (n_threads, n_agents, dim)
        """
        if self.args["algo"] == "mafis":
            sample_num = self.algo_args["train"]["sample_num"]
            noise = self.algo_args["algo"]["noise"]
            langevin_steps = self.algo_args["algo"]["langevin_steps"]
            alpha = self.algo_args["algo"]["alpha"]
            epsilon = self.algo_args["algo"]["langevin_step_size"]
            self.critic.turn_on_grad()

            torch_obs = torch.from_numpy(obs.transpose(1, 0, 2)).to(torch.float32).cuda().unsqueeze(2)
            repeated_obs = torch.repeat_interleave(torch_obs, sample_num, dim=2)
            torch_share_obs = torch.from_numpy(share_obs.transpose(1, 0, 2)).to(torch.float32).cuda().unsqueeze(2)
            repeated_share_obs = torch.repeat_interleave(torch_share_obs, sample_num, dim=2)
            weight = self.critic.mixer(repeated_share_obs).sigmoid().squeeze(-1).transpose(1, 0).detach()
            random_actions = (2 * torch.rand(repeated_obs.shape[0], repeated_obs.shape[1], repeated_obs.shape[2], self.action_spaces[0].shape[0]) - 1).cuda()
            actions = langevin_mcmc_s_a(self.critic.critic, 
                                                    repeated_obs, 
                                                    random_actions, 
                                                    num_iterations = langevin_steps, 
                                                    noise_scale = noise, 
                                                    alpha = alpha,
                                                    epsilon = epsilon,
                                                    grad_clip = 10, 
                                                    delta_clip = 0.5, 
                                                    margin_clip = 1.0,
                                                    sampler_stepsize_init = 1,
                                                    sampler_stepsize_final = 1e-5,
                                                    weight=weight
                                                    )
            q = self.critic.critic(repeated_obs, actions)
            indices = q.max(2)[1].unsqueeze(-1).expand(-1, -1, -1, self.action_spaces[0].shape[0])
            actions = torch.gather(actions, 2, indices).cpu().squeeze(2)
            self.critic.turn_off_grad()
        else:
            raise NotImplementedError
        return np.array(actions).transpose(1, 0, 2)
    
    def get_dec_actions(self, obs, share_obs, available_actions=None, add_random=True):
        """Get decentralized actions for rollout."""
        assert self.args["algo"] == "mafis"

        sample_num = self.algo_args["train"]["sample_num"]
        noise = self.algo_args["algo"]["noise"]
        langevin_steps = self.algo_args["algo"]["langevin_steps"]
        alpha = self.algo_args["algo"]["alpha"]
        epsilon = self.algo_args["algo"]["langevin_step_size"]

        torch_obs = torch.from_numpy(obs.transpose(1, 0, 2)).to(torch.float32).cuda().unsqueeze(2)
        repeated_obs = torch.repeat_interleave(torch_obs, sample_num, dim=2)
        torch_share_obs = torch.from_numpy(share_obs.transpose(1, 0, 2)).to(torch.float32).cuda().unsqueeze(2)
        repeated_share_obs = torch.repeat_interleave(torch_share_obs, sample_num, dim=2)
        weight = self.critic.mixer(repeated_share_obs).sigmoid().squeeze(-1).transpose(1, 0).detach()
        random_actions = (2 * torch.rand(repeated_obs.shape[0], repeated_obs.shape[1], repeated_obs.shape[2], self.action_spaces[0].shape[0]) - 1).cuda()
        actions = langevin_mcmc_s_a(self.critic.critic, 
                                                repeated_obs, 
                                                random_actions, 
                                                num_iterations = langevin_steps, 
                                                noise_scale = noise, 
                                                alpha = alpha,
                                                epsilon = epsilon,
                                                grad_clip = 10, 
                                                delta_clip = 0.5, 
                                                margin_clip = 1.0,
                                                sampler_stepsize_init = 1,
                                                sampler_stepsize_final = 1e-5,
                                                weight=None
                                                )
        q = self.critic.critic(repeated_obs, actions)
        indices = q.max(2)[1].unsqueeze(-1).expand(-1, -1, -1, self.action_spaces[0].shape[0])
        actions = torch.gather(actions, 2, indices).cpu().squeeze(2)

        return np.array(actions).transpose(1, 0, 2)

    def train(self):
        """Train the model"""
        raise NotImplementedError

    def eval(self, step):
        """Evaluate the model"""
        eval_episode_rewards = []
        one_episode_rewards = []
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])
        eval_episode = 0
        episode_lens = []
        one_episode_len = np.zeros(
            self.algo_args["eval"]["n_eval_rollout_threads"], dtype=int
        )

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        while True:
            if self.args["algo"] == "mafis":
                eval_actions = self.get_dec_actions(
                    eval_obs, eval_share_obs, available_actions=eval_available_actions, add_random=False
                )
            else:
                eval_actions = self.get_actions(
                    eval_obs, eval_share_obs, available_actions=eval_available_actions, add_random=False
                )
            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])

            one_episode_len += 1

            eval_dones_env = np.all(eval_dones, axis=1)

            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards[eval_i].append(
                        np.sum(one_episode_rewards[eval_i], axis=0)
                    )
                    one_episode_rewards[eval_i] = []
                    episode_lens.append(one_episode_len[eval_i].copy())
                    one_episode_len[eval_i] = 0

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                # eval_log returns whether the current model should be saved
                eval_episode_rewards = np.concatenate(
                    [rewards for rewards in eval_episode_rewards if rewards]
                )
                eval_avg_rew = np.mean(eval_episode_rewards)
                eval_avg_len = np.mean(episode_lens)
                print(f"Eval average episode reward is {eval_avg_rew}, eval average episode length is {eval_avg_len}.\n")
                
                self.log_file.write(",".join(map(str, [step, eval_avg_rew, eval_avg_len])) + "\n")
                self.log_file.flush()
                self.writter.add_scalar(
                    "eval_average_episode_rewards", eval_avg_rew, step
                )
                self.writter.add_scalar(
                    "eval_average_episode_length", eval_avg_len, step
                )
                break

    def restore(self):
        """Restore the model"""
        if not self.algo_args["render"]["use_render"]:
            self.critic.restore(self.algo_args["train"]["model_dir"])
            if self.value_normalizer is not None:
                value_normalizer_state_dict = torch.load(
                    str(self.algo_args["train"]["model_dir"])
                    + "/value_normalizer"
                    + ".pt"
                )
                self.value_normalizer.load_state_dict(value_normalizer_state_dict)

    def save(self):
        """Save the model"""
        self.critic.save(self.save_dir)
        if self.value_normalizer is not None:
            torch.save(
                self.value_normalizer.state_dict(),
                str(self.save_dir) + "/value_normalizer" + ".pt",
            )

    def close(self):
        """Close environment, writter, and log file."""
        # post process
        if self.algo_args["render"]["use_render"]:
            self.envs.close()
        else:
            self.envs.close()
            if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
            self.writter.close()
            self.log_file.close()
