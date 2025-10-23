"""Soft Twin Continuous Q Critic."""
import numpy as np
import torch
import torch.nn.functional as F
from mafis.algorithms.critics.twin_continuous_q_critic import TwinContinuousQCritic
from mafis.utils.envs_tools import check
from abc import ABC, abstractmethod

class SoftTwinContinuousQCritic(TwinContinuousQCritic):
    """Soft Twin Continuous Q Critic.
    Critic that learns two soft Q-functions. The action space can be continuous and discrete.
    Note that the name SoftTwinContinuousQCritic emphasizes its structure that takes observations and actions as input
    and outputs the q values. Thus, it is commonly used to handle continuous action space; meanwhile, it can also be
    used in discrete action space.
    """

    def __init__(
        self,
        args,
        share_obs_space,
        obs_space,
        act_space,
        num_agents,
        state_type,
        device=torch.device("cpu"),
    ):
        """Initialize the critic."""
        super(SoftTwinContinuousQCritic, self).__init__(
            args, share_obs_space, obs_space, act_space, num_agents, state_type, device
        )
        self.tpdv_a = dict(dtype=torch.int64, device=device)
        self.args = args
        self.batch_size = self.args["batch_size"]
        self.noise = self.args["noise"]


    def get_values(self, share_obs, actions):
        """Get the soft Q values for the given observations and actions."""
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        return torch.min(
            self.critic(share_obs, actions), self.critic2(share_obs, actions)
        )

        
    def train_MAFIS(
        self,
        obs,
        share_obs,
        actions,
        done,
        valid_transition,
        term,
        next_obs,
        next_share_obs,
        gamma,
    ):
        ## NEW!: copy share_obs for each agent
        """Args:
            obs: (n_agents, batch_size, dim)
            share_obs: EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            actions: (n_agents, batch_size, dim)
            reward: EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            done: EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            valid_transition: (n_agents, batch_size, 1)
            term: EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            next_share_obs: EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            next_actions: (n_agents, batch_size, dim)
            next_logp_actions: (n_agents, batch_size, 1)
            gamma: EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
        """
        share_obs = np.expand_dims(share_obs, axis=0)
        share_obs = np.repeat(share_obs, self.num_agents, axis=0)
        next_share_obs = np.expand_dims(next_share_obs, axis=0)
        next_share_obs = np.repeat(next_share_obs, self.num_agents, axis=0)

        obs = check(obs).to(**self.tpdv)
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        done = check(done).to(**self.tpdv)
        term = check(term).to(**self.tpdv)
        gamma = 0.99
        next_obs = check(next_obs).to(**self.tpdv)
        next_share_obs = check(next_share_obs).to(**self.tpdv)
        
        sample_num = self.sample_num
        langevin_steps = self.args["langevin_steps"]
        alpha = self.args["alpha"]
        epsilon = self.args["langevin_step_size"]
        
        repeated_curr_obs = torch.repeat_interleave(obs, sample_num, dim=1)
        repeated_curr_share_obs = torch.repeat_interleave(share_obs, sample_num, dim=1)
        repeated_term = torch.repeat_interleave(term, sample_num, dim=0)
        random_delta_action_samples = (2 * torch.rand(repeated_curr_obs.shape[0], repeated_curr_obs.shape[1], actions.shape[-1]) - 1).to(**self.tpdv)
        
        curr_weight = self.mixer(repeated_curr_share_obs).sigmoid().squeeze(-1).transpose(1, 0)
        demo_weight = self.mixer(share_obs).sigmoid().squeeze(-1).transpose(1, 0)
        langevin_delta_actions = langevin_mcmc_s_a(self.target_critic, 
                                                    repeated_curr_obs, 
                                                    random_delta_action_samples, 
                                                    num_iterations = langevin_steps, 
                                                    noise_scale = self.noise, 
                                                    alpha=alpha,
                                                    epsilon=epsilon,
                                                    grad_clip = 10, 
                                                    delta_clip = 0.5, 
                                                    margin_clip = 1.0,
                                                    sampler_stepsize_init = 1,
                                                    sampler_stepsize_final = 1e-5,
                                                    weight=curr_weight
                                                    )
            
        repeated_next_obs = torch.repeat_interleave(next_obs, sample_num, dim=1)
        repeated_next_share_obs = torch.repeat_interleave(next_share_obs, sample_num, dim=1)
        random_delta_next_action_samples = (2 * torch.rand(repeated_next_obs.shape[0], repeated_next_obs.shape[1], actions.shape[-1]) - 1).to(**self.tpdv)
        next_weight = self.mixer(repeated_next_share_obs).sigmoid().squeeze(-1).transpose(1, 0)
        
        langevin_delta_next_actions = langevin_mcmc_s_a(self.target_critic, 
                                                    repeated_next_obs, 
                                                    random_delta_next_action_samples, 
                                                    num_iterations = langevin_steps, 
                                                    noise_scale = self.noise, 
                                                    alpha=alpha,
                                                    epsilon=epsilon,
                                                    grad_clip = 10, 
                                                    delta_clip = 0.5, 
                                                    margin_clip = 1.0,
                                                    sampler_stepsize_init = 1,
                                                    sampler_stepsize_final = 1e-5,
                                                    weight=next_weight
                                                    )
        
        next_q_values = self.critic(repeated_next_obs, langevin_delta_next_actions).transpose(1, 0).squeeze(2)    
        next_v = (next_weight * next_q_values).sum(-1, keepdim=True)
        y = gamma * next_v 

        demo_q_values = self.critic(obs, actions).transpose(1, 0).squeeze(2)    
        demo_q_tot = (demo_weight * demo_q_values).sum(-1, keepdim=True)

        curr_q_values = self.critic(repeated_curr_obs, langevin_delta_actions).transpose(1, 0).squeeze(2)
        curr_v = (curr_weight * curr_q_values).sum(-1, keepdim=True)
        
        critic_loss1 = -(demo_q_tot[:self.args["batch_size"]].mean() - y[:self.args["batch_size"] * sample_num].mean())
        critic_loss2 = (curr_v - y).mean()
        
        grad_loss1, grad_norm_mean = self.grad_penalty(self.args["batch_size"] * sample_num * 2, repeated_curr_obs, langevin_delta_actions, grad_margin=1.0, create_graph=True)
        grad_loss2, grad_norm_mean = self.grad_penalty(self.args["batch_size"] * sample_num * 2, repeated_next_obs, langevin_delta_next_actions, grad_margin=1.0, create_graph=True)
        
        l2_loss = (demo_q_tot[:self.args["batch_size"]] ** 2).mean() + (curr_v ** 2).mean() + (next_v ** 2).mean()
        critic_loss = critic_loss1 + critic_loss2 + 0.25 * grad_loss1 + 0.25 * grad_loss2 + 0.01 * l2_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss, critic_loss1, critic_loss2, grad_loss1, grad_loss2, l2_loss
    
    def grad_penalty(
        self, 
        batch_size, 
        obs,
        action,   
        grad_margin: float=1.0,
        square_grad_penalty: bool=True,
        create_graph: bool=True,
        ):
        grad, _ = grad_wrt_next_s(self.critic, obs, action.detach(), self.args["alpha"], create_graph=create_graph)
        grad_norms = torch.norm(grad, dim=-1)
        grad_norm_mean = grad_norms.mean().item()

        if grad_margin is not None:
            grad_norms = grad_norms - grad_margin

            grad_norms = torch.clamp(grad_norms, 0., 1e10)

        if square_grad_penalty:
            grad_norms = grad_norms ** 2

        grad_loss = torch.mean(grad_norms)

        return grad_loss, grad_norm_mean

def langevin_mcmc_s_a(
        network,
        obs,
        delta_action,
        num_iterations=25,
        sampler_stepsize_init=1e-1,
        sampler_stepsize_decay=0.8,
        noise_scale=1.0,
        alpha=1.0,
        epsilon=0.5,
        grad_clip=None,
        delta_clip=None,
        margin_clip=None,
        use_polynomial_rate=True,  # default is exponential
        sampler_stepsize_final=1e-5,  # if using polynomial langevin rate.
        sampler_stepsize_power=3,  # if using polynomial langevin rate.
        weight=None
):
    stepsize = sampler_stepsize_init

    if use_polynomial_rate:
        schedule = PolynomialSchedule(sampler_stepsize_init, sampler_stepsize_final,
                                    sampler_stepsize_power, num_iterations)
    else:  # default to exponential rate
        schedule = ExponentialSchedule(sampler_stepsize_init,
                                    sampler_stepsize_decay)

    for step in range(num_iterations):
        delta_action, _, _ = langevin_step(network, 
                                            obs,
                                            delta_action,
                                            noise_scale,
                                            grad_clip,
                                            delta_clip,
                                            margin_clip,
                                            stepsize,
                                            alpha,
                                            epsilon,
                                            weight=weight
                                            )
        delta_action = delta_action.detach()
        stepsize = schedule.get_rate(step + 1)
    return delta_action

def grad_wrt_next_s(
        network,
        obs,
        delta_action,
        alpha,
        create_graph=False,
        weight=None,
):
    delta_action.requires_grad = True
    energies = network(obs, delta_action) / alpha
    if weight is not None:
        w = weight.transpose(1, 0).unsqueeze(-1)
        energies *= w
    grad = torch.autograd.grad(energies.sum(), delta_action, create_graph=create_graph)[0]
    return grad, energies

def langevin_step(
    network,
    obs,
    delta_action,
    noise_scale,
    grad_clip,
    delta_clip,
    margin_clip,
    stepsize,
    alpha,
    epsilon,
    weight=None,
):
    l_lambda = epsilon
    grad, energy = grad_wrt_next_s(network, obs, delta_action, alpha, weight=weight)

    grad_norm = torch.norm(grad, dim=-1, keepdim=True)

    if grad_clip:
        grad = torch.clamp(grad, -grad_clip, grad_clip)

    delta_action_drift = stepsize * (l_lambda * grad + torch.randn(delta_action.shape, device=grad.device) * noise_scale)

    if delta_clip:
       delta_action_drift = torch.clamp(delta_action_drift, -delta_clip, delta_clip)

    delta_action = delta_action + delta_action_drift

    if margin_clip is not None:
        delta_action = torch.clamp(delta_action, -margin_clip, margin_clip)
    return delta_action, energy, grad_norm


class ExponentialSchedule:
    def __init__(self, init, decay):
        self._decay = decay
        self._latest_lr = init

    def get_rate(self, index):
        del index
        self._latest_lr *= self._decay
        return self._latest_lr

class PolynomialSchedule:
    def __init__(self, init, final, power, num_steps):
        self._init = init
        self._final = final
        self._power = power
        self._num_steps = num_steps

    def get_rate(self, index):
        return ((self._init - self._final) *
                ((1 - (float(index) / float(self._num_steps-1))) ** (self._power))
                ) + self._final
