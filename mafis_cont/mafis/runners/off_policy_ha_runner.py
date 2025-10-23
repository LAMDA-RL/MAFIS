"""Runner for off-policy HARL algorithms."""
import torch
import numpy as np
import torch.nn.functional as F
from mafis.runners.off_policy_base_runner import OffPolicyBaseRunner
from mafis.utils.envs_tools import check


class OffPolicyHARunner(OffPolicyBaseRunner):
    """Runner for off-policy HA algorithms."""


    def train_MAFIS(self):
        """Train the model"""
        self.total_it += 1
        data = self.buffer.sample()
        (
            sp_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_obs,  # (n_agents, batch_size, dim)
            sp_actions,  # (n_agents, batch_size, dim)
            sp_available_actions,  # (n_agents, batch_size, dim)
            sp_reward,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_done,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_valid_transition,  # (n_agents, batch_size, 1)
            sp_term,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_next_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_next_obs,  # (n_agents, batch_size, dim)
            sp_next_available_actions,  # (n_agents, batch_size, dim)
            sp_gamma,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
        ) = data
        
        demo_data = self.demo_buffer.sample()
        (
            demo_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            demo_obs,  # (n_agents, batch_size, dim)
            demo_actions,  # (n_agents, batch_size, dim)
            demo_available_actions,  # (n_agents, batch_size, dim)
            demo_reward,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            demo_done,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            demo_valid_transition,  # (n_agents, batch_size, 1)
            demo_term,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            demo_next_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            demo_next_obs,  # (n_agents, batch_size, dim)
            demo_next_available_actions,  # (n_agents, batch_size, dim)
            demo_gamma,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
        ) = demo_data
        
        # train critic
        all_obs = np.concatenate([demo_obs, sp_obs], axis=1)
        all_share_obs = np.concatenate([demo_share_obs, sp_share_obs], axis=0)
        all_next_obs = np.concatenate([demo_next_obs, sp_next_obs], axis=1)
        all_next_share_obs = np.concatenate([demo_next_share_obs, sp_next_share_obs], axis=0)
        all_actions = np.concatenate([demo_actions, sp_actions], axis=1)
        all_done = np.concatenate([demo_done, sp_done], axis=0)
        all_term = np.concatenate([demo_term, sp_term], axis=0)
        all_valid_transition = np.concatenate([demo_valid_transition, sp_valid_transition], axis=1)
        all_gamma = np.concatenate([demo_gamma, sp_gamma], axis=0)

        self.critic.turn_on_grad()
        critic_loss = self.critic.train_MAFIS(
            all_obs,
            all_share_obs,
            all_actions,
            all_done,
            all_valid_transition,
            all_term,
            all_next_obs,
            all_next_share_obs,
            all_gamma,
        )
        self.critic.turn_off_grad()
        
        self.critic.soft_update()
        return critic_loss
