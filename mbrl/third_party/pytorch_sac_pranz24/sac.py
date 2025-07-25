import os

import torch
import torch.nn.functional as F
from torch.optim import Adam

from mbrl.third_party.pytorch_sac_pranz24.model import (
    DeterministicPolicy,
    GaussianPolicy,
    QNetwork,
    GaussianPolicyLayerNorm,
    QNetworkLayerNorm
)
from mbrl.third_party.pytorch_sac_pranz24.utils import hard_update, soft_update
from mbrl.util import ReplayBuffer, InfoReplayBuffer
import numpy as np


class SAC(object):
    def __init__(self, num_inputs, action_space, args):
        self.args = args
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = args.device
        layer_norm = args.get("layer_norm", False)
        print("SAC LayerNorm:", layer_norm)
        if not layer_norm:
            self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(
                device=self.device
            )
        else:
            self.critic = QNetworkLayerNorm(num_inputs, action_space.shape[0], args.hidden_size).to(
                device=self.device
            )
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr, )#weight_decay=1e-5)
        if not layer_norm:
            self.critic_target = QNetwork(
                num_inputs, action_space.shape[0], args.hidden_size
            ).to(self.device)
        else: 
            self.critic_target = QNetworkLayerNorm(
                num_inputs, action_space.shape[0], args.hidden_size
            ).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                if args.target_entropy is None:
                    self.target_entropy = -torch.prod(
                        torch.Tensor(action_space.shape).to(self.device)
                    ).item()
                else:
                    self.target_entropy = args.target_entropy
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            if not layer_norm:
                self.policy = GaussianPolicy(
                    num_inputs, action_space.shape[0], args.hidden_size, action_space
                ).to(self.device)
            else:
                self.policy = GaussianPolicyLayerNorm(
                    num_inputs, action_space.shape[0], args.hidden_size, action_space
                ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(
                num_inputs, action_space.shape[0], args.hidden_size, action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        self.normalizer = None

    def select_action(self, state, batched=False, evaluate=False):
        state = torch.FloatTensor(state)
        if not batched:
            state = state.unsqueeze(0)
        state = state.to(self.device)
        if self.normalizer:
            dtype = state.dtype
            state_mean = self.normalizer.mean[:,0:state.shape[-1]]
            state_std = self.normalizer.std[:,0:state.shape[-1]]
            state = ((state - state_mean)/state_std).to(dtype)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        if batched:
            return action.detach().cpu().numpy()
        return action.detach().cpu().numpy()[0]
    
    def select_action_eps(self, state, eps, batched=False, evaluate=False):
        state = torch.FloatTensor(state)
        eps = torch.FloatTensor(eps)
        if not batched:
            state = state.unsqueeze(0)
            eps = eps.unsqueeze(0)
        state = state.to(self.device)
        eps = eps.to(self.device)
        if self.normalizer:
            dtype = state.dtype
            state_mean = self.normalizer.mean[:,0:state.shape[-1]]
            state_std = self.normalizer.std[:,0:state.shape[-1]]
            state = ((state - state_mean)/state_std).to(dtype)
        if evaluate is False:
            action, _, _ = self.policy.sample_using_eps(state, eps)
        else:
            _, _, action = self.policy.sample_using_eps(state, eps)
        if batched:
            return action.detach().cpu().numpy()
        return action.detach().cpu().numpy()[0]
    
    # def weighted_mse(self, qf, next_q_value, weights):
    #     return torch.mean(weights*torch.square((qf - next_q_value)))

    def update_parameters(
        self, memory, batch_size, updates, logger=None, reverse_mask=False
    ):
        # Sample a batch from memory and ignore truncateds
        if type(memory) is InfoReplayBuffer:
            tr_batch, _, _, loss_rate = memory.info_sample(batch_size)
            (
                state_batch,
                action_batch,
                next_state_batch,
                reward_batch,
                mask_batch,
                _,
            ) = tr_batch.astuple()

            # weights = loss_rate.mean(axis=-1)
            weights = np.ones_like(reward_batch)

        elif type(memory) is ReplayBuffer:
            (
                state_batch,
                action_batch,
                next_state_batch,
                reward_batch,
                mask_batch,
                _,
            ) = memory.sample(batch_size).astuple()

            weights = np.ones_like(reward_batch)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        weights = torch.FloatTensor(weights).to(self.device).unsqueeze(1)

        if self.normalizer:
            dtype = state_batch.dtype
            state_mean = self.normalizer.mean[:,:state_batch.shape[-1]]
            state_std = self.normalizer.std[:,:state_batch.shape[-1]]
            state_batch = ((state_batch - state_mean) / state_std).to(dtype)
            next_state_batch = ((next_state_batch - state_mean) / state_std).to(dtype)

        if reverse_mask:
            mask_batch = mask_batch.logical_not()

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_state_batch
            )
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(
            state_batch, action_batch
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = (weights*torch.square((qf1 - next_q_value))).sum()/weights.sum()  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = (weights*torch.square((qf2 - next_q_value))).sum()/weights.sum()  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        # print(qf_loss.shape, qf1.shape, next_q_value.shape)
        # raise NotImplementedError

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (weights*(
            (self.alpha * log_pi) - min_qf_pi
        )).sum()/weights.sum()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(weights*(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            )).sum()/(weights.sum())

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        if logger is not None:
            logger.log("train/batch_reward", (weights*reward_batch).sum()/weights.sum(), updates)
            logger.log("train_critic/loss", qf_loss, updates)
            logger.log("train_actor/loss", policy_loss, updates)
            if self.automatic_entropy_tuning:
                logger.log("train_actor/target_entropy", self.target_entropy, updates)
            else:
                logger.log("train_actor/target_entropy", 0, updates)
            logger.log("train_actor/entropy", -log_pi.mean(), updates)
            logger.log("train_alpha/loss", alpha_loss, updates)
            logger.log("train_alpha/value", self.alpha, updates)

        return (
            qf1_loss.item(),
            qf2_loss.item(),
            policy_loss.item(),
            alpha_loss.item(),
            alpha_tlogs.item(),
        )

    # Save model parameters
    def save_checkpoint(self, env_name=None, suffix="", ckpt_path=None):
        if ckpt_path is None:
            assert env_name is not None
            if not os.path.exists("checkpoints/"):
                os.makedirs("checkpoints/")
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print("Saving models to {}".format(ckpt_path))
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "critic_optimizer_state_dict": self.critic_optim.state_dict(),
                "policy_optimizer_state_dict": self.policy_optim.state_dict(),
            },
            ckpt_path,
        )

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print("Loading models from {}".format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.policy_optim.load_state_dict(checkpoint["policy_optimizer_state_dict"])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
