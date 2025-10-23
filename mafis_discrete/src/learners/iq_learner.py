import random

import torch as th
from torch.optim import Adam
from torch.nn import functional as F
from modules.mixers.qmix import Mix


class IQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        if args.mixer is not None:
            if args.mixer == "mix":
                self.mixer = Mix(args)
                if args.name == "MAFIS_online":
                    self.mac.add_mixer(self.mixer)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())

        self.optimiser = Adam(params=self.params, lr=args.lr)

        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        

    def train(self, batch, t_env, episode_num):
        # Get the relevant quantities
        obss = batch["obs"]
        states = batch["state"]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)    
        mac_out = th.stack(mac_out, dim=1)
        
        individual_q = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(-1)
        current_q_tot = self.mixer(individual_q, states[:, :-1])
        next_v_tot = self.mixer.forward_V_star(mac_out[:, 1:], states[:, 1:])
        y = (1 - terminated) * self.args.gamma * next_v_tot
        r = (current_q_tot - y) * mask
        
        with th.no_grad():
            # Use different divergence functions (For χ2 divergence we instead add a third bellmann error-like term)
            if self.args.div == "hellinger":
                phi_grad = 1 / (1 + r) ** 2
            elif self.args.div == "kl":
                # original dual form for kl divergence (sub optimal)
                phi_grad = th.exp(-r - 1)
            elif self.args.div == "kl_fix":
                # our proposed unbiased form for fixing kl divergence
                phi_grad = th.exp(-r)
            elif self.args.div == "kl2":
                # biased dual form for kl divergence
                phi_grad = F.softmax(-r, dim=0) * r.shape[0]
            elif self.args.div == "js":
                # jensen–shannon
                phi_grad = th.exp(-r) / (2 - th.exp(-r))
            else:
                phi_grad = 1
        
        loss = - (phi_grad * r).mean()
        
        current_v_tot = self.mixer.forward_V_star(mac_out[:, :-1], states[:, :-1])
        value_loss = ((current_v_tot - y) * mask).mean()
        loss += value_loss

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("value_loss", value_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat("q", current_q_tot.mean().item(), t_env)
            self.logger.log_stat("v", current_v_tot.mean().item(), t_env)
            self.log_stats_t = t_env

    def train_online(self, batch, t_env, episode_num):
        # Get the relevant quantities
        for _ in range(self.args.epoch):
            obss = batch["obs"]
            states = batch["state"]
            actions = batch["actions"][:, :-1]
            terminated = batch["terminated"][:, :-1].float()
            mask = batch["filled"][:, :-1].float()
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
                
            mac_out = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.mac.forward(batch, t)
                mac_out.append(agent_outs)    
            mac_out = th.stack(mac_out, dim=1)
            individual_q = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(-1)
            current_q_tot = self.mixer(individual_q, states[:, :-1])
            next_v_tot = self.mixer.forward_V_star(mac_out[:, 1:], states[:, 1:])
            y = (1 - terminated) * self.args.gamma * next_v_tot
            r = (current_q_tot - y) * mask
            r = r[:self.args.batch_size//2]
            
            with th.no_grad():
                # Use different divergence functions (For χ2 divergence we instead add a third bellmann error-like term)
                if self.args.div == "hellinger":
                    phi_grad = 1 / (1 + r) ** 2
                elif self.args.div == "kl":
                    # original dual form for kl divergence (sub optimal)
                    phi_grad = th.exp(-r - 1)
                elif self.args.div == "kl_fix":
                    # our proposed unbiased form for fixing kl divergence
                    phi_grad = th.exp(-r)
                elif self.args.div == "kl2":
                    # biased dual form for kl divergence
                    phi_grad = F.softmax(-r, dim=0) * r.shape[0]
                elif self.args.div == "js":
                    # jensen–shannon
                    phi_grad = th.exp(-r) / (2 - th.exp(-r))
                else:
                    phi_grad = 1
            loss = -(phi_grad * r).mean()
            
            current_v_tot = self.mixer.forward_V_star(mac_out[:, :-1], states[:, :-1])
            value_loss = ((current_v_tot - y) * mask).mean()
            loss += value_loss
            
            self.optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(
                self.params, self.args.grad_norm_clip)
            self.optimiser.step()

            if t_env - self.log_stats_t >= self.args.learner_log_interval:
                self.logger.log_stat("loss", loss.item(), t_env)
                self.logger.log_stat("value_loss", value_loss.item(), t_env)
                self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
                self.logger.log_stat("q", current_q_tot.mean().item(), t_env)
                self.logger.log_stat("v", current_v_tot.mean().item(), t_env)
                self.log_stats_t = t_env

    def cuda(self):
        self.mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path),
                    map_location=lambda storage, loc: storage)
        )
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
