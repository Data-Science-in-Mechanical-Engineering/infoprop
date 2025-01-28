# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional, Sequence, cast

import gymnasium as gym
import hydra.utils
import numpy as np
import omegaconf
import torch

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac_pranz24
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math
from mbrl.planning.sac_wrapper import SACAgent
from omegaconf import OmegaConf
import colorednoise as cn

MBPO_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT + [
    ("epoch", "E", "int"),
    ("rollout_length", "RL", "int"),
]

def rollout_model_and_populate_sac_buffer(
    model_env: mbrl.models.ModelEnv,
    replay_buffer: mbrl.util.ReplayBuffer,
    agent: SACAgent,
    sac_buffer: mbrl.util.ReplayBuffer,
    sac_samples_action: bool,
    rollout_horizon: int,
    batch_size: int,
    env_steps,
    wandb_log = False,
    wandb = None,
    np_rng = None,
    act_shape = (6,1)
):
    with torch.no_grad():
        num_added = 0
        sampling_round = 0
        transitions = replay_buffer.sample(batch_size=batch_size).astuple() #sample a batch of transitions
        obs_dataset = transitions[0]
        actions_dataset = transitions[1]
        model_state = {"obs": obs_dataset}
        pred_next_obs, pred_rewards, pred_dones, next_model_state = model_env.info_step(
            actions_dataset, model_state, sample=True
        )
        conditional_var = next_model_state["conditional_var"]   
        quantization_bits = np.log2(1e-6) 
        lost_info_dataset = np.clip(0.5*torch.log2(2*np.pi*np.exp(1)*conditional_var).cpu().numpy() - quantization_bits, 0, np.inf)#.sum(axis=-1, keepdims=True)
        # compute thresholds
        per_step_loss_threshold = np.quantile(lost_info_dataset, q=0.9999, axis=0, keepdims=True)
        max_loss_threshold = rollout_horizon*np.quantile(lost_info_dataset, q=0.01, axis=0, keepdims=True)

        total_initial_states = 0
        complete_rollouts = []
        # repeat till we have added enough transitions
        while num_added < batch_size:
            batch_size_ = batch_size - int(num_added)
            total_initial_states += batch_size_

            if sampling_round > 0: 
                batch = replay_buffer.sample(batch_size_) # resample if we have not added enough transitions
                initial_obs, *_ = cast(mbrl.types.TransitionBatch, batch).astuple()

                
                obs = initial_obs
            else: obs=obs_dataset
            obs_act = obs
            obs_dims = obs.shape[-1]
            accum_dones = np.zeros(obs.shape[0],dtype=bool)
            conditional_var = 1e-10*np.ones(obs.shape)
            
            lost_info = np.zeros((batch_size_, obs_dims+1)) # lost info for each obs dim and reward

            model_state = {}
            rollout_tracker = np.zeros(batch_size_)
            complete_rollouts.append(rollout_tracker)
            curr_idx = np.arange(batch_size_)
            for step in range(rollout_horizon):
                accum_dones = np.zeros(obs.shape[0],dtype=bool)
                action = agent.act(obs, sample=sac_samples_action, batched=True)
                model_state["obs"] = obs
                pred_next_obs, pred_rewards, pred_dones, next_model_state = model_env.info_step(
                    action, model_state, sample=True
                )
                truncateds = np.zeros_like(pred_dones, dtype=bool)
                # change accum_dones here
                conditional_var = next_model_state["conditional_var"].cpu().numpy()  

                # compute lost info
                lost_info_this_step = np.clip(0.5*np.log2(2*np.pi*np.exp(1)*conditional_var) - quantization_bits, 0, np.inf)#.sum(axis=-1, keepdims=True)
                lost_info += lost_info_this_step

                # apply thresholds
                accum_dones |= (lost_info_this_step > per_step_loss_threshold).any(axis=-1)
                accum_dones |= (lost_info > max_loss_threshold).any(axis=-1)

                sac_buffer.add_batch(
                    obs[~accum_dones],
                    action[~accum_dones],
                    pred_next_obs[~accum_dones],
                    pred_rewards[~accum_dones, 0],
                    pred_dones[~accum_dones, 0],
                    truncateds[~accum_dones, 0],
                    lost_info_this_step[~accum_dones],
                    lost_info[~accum_dones],
                    np.ones_like(lost_info[~accum_dones]), # this is an artifact from an older version of the code. IGNORE!
                )
                num_added += ((~accum_dones).sum())
                added_idx = curr_idx[~accum_dones]

                rollout_tracker[added_idx] += 1

                accum_dones |= pred_dones.squeeze()
                obs = pred_next_obs[~accum_dones] 
                if len(obs) == 0: break
                obs_act = next_model_state["obs_act"][~accum_dones].cpu().numpy()
                lost_info = lost_info[~accum_dones]
                curr_idx = curr_idx[~accum_dones]
                
                
            print(num_added, step)
            sampling_round += 1
            if sampling_round == 1: assert num_added>0
        rollout_tracker = np.concatenate(complete_rollouts)
        sac_buffer.re_compute_sampling_idxs()
        if wandb_log:
            # wandb logging for rollout metrics
            data = {
                "rollout/env_step": env_steps + 1,
                "rollout/average_length": (rollout_tracker.mean()),
                "rollout/minimum_length": np.min(rollout_tracker),
                "rollout/maximum_length": np.max(rollout_tracker),
                "rollout/added_transitions": num_added
            }
            for obs_dim in range(obs_dims):
                data["rollout/obs_dim_" + str(obs_dim) + "_quant_ent"] = quantization_bits
                data["rollout/loss_rate_limit_" + str(obs_dim)] = per_step_loss_threshold[0, obs_dim]
                data["rollout/loss_limit_" + str(obs_dim)] = max_loss_threshold[0, obs_dim]
            data["rollout/reward_quant_ent"] = quantization_bits
            data["rollout/loss_rate_limit_reward"] = per_step_loss_threshold[0, -1]
            data["rollout/loss_limit_reward"] = max_loss_threshold[0, -1]
            hist_data = rollout_tracker[rollout_tracker>0]
            hist = np.histogram(hist_data, bins=min(512, int(hist_data.max())), density=True,)
            data["rollout/rollout_length_histogram"] = wandb.Histogram(np_histogram=hist)
            wandb.log(
            data=data,
        )
        return num_added


def evaluate(
    env: gym.Env,
    agent: SACAgent,
    num_episodes: int,
) -> float:
    avg_episode_reward = 0.0
    for episode in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0
        while not terminated and not truncated:
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
        avg_episode_reward += episode_reward
    return avg_episode_reward / num_episodes


def maybe_replace_sac_buffer(
    sac_buffer: Optional[mbrl.util.InfoReplayBuffer],
    obs_shape: Sequence[int],
    act_shape: Sequence[int],
    new_capacity: int,
    seed: int,
) -> mbrl.util.ReplayBuffer:
    if sac_buffer is None or new_capacity != sac_buffer.capacity:
        if sac_buffer is None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = sac_buffer.rng
        new_buffer = mbrl.util.InfoReplayBuffer(new_capacity, obs_shape, act_shape, rng=rng)
        if sac_buffer is None:
            return new_buffer
        (
            obs,
            action,
            next_obs,
            reward,
            terminated,
            truncated,
        ) = sac_buffer.get_all().astuple()
        new_buffer.add_batch(obs, action, next_obs, reward, terminated, truncated)
        return new_buffer
    return sac_buffer

def remove_old_transitions(sac_buffer, sac_buffer_tracker):
    to_be_removed = sac_buffer_tracker.pop(0)
    num_stored = sac_buffer.num_stored

    np.copyto(sac_buffer.obs[:num_stored - to_be_removed], sac_buffer.obs[to_be_removed:num_stored])
    np.copyto(sac_buffer.action[:num_stored - to_be_removed], sac_buffer.action[to_be_removed:num_stored])
    np.copyto(sac_buffer.reward[:num_stored - to_be_removed], sac_buffer.reward[to_be_removed:num_stored])
    np.copyto(sac_buffer.next_obs[:num_stored - to_be_removed], sac_buffer.next_obs[to_be_removed:num_stored])
    np.copyto(sac_buffer.terminated[:num_stored - to_be_removed], sac_buffer.terminated[to_be_removed:num_stored])
    np.copyto(sac_buffer.truncated[:num_stored - to_be_removed], sac_buffer.truncated[to_be_removed:num_stored])
    np.copyto(sac_buffer.info_lost_current_step[:num_stored - to_be_removed], sac_buffer.info_lost_current_step[to_be_removed:num_stored])
    np.copyto(sac_buffer.info_lost_total[:num_stored - to_be_removed], sac_buffer.info_lost_total[to_be_removed:num_stored])
    np.copyto(sac_buffer.label[:num_stored - to_be_removed], sac_buffer.label[to_be_removed:num_stored])
    sac_buffer.cur_idx = (num_stored - to_be_removed) % sac_buffer.capacity
    sac_buffer.num_stored = num_stored - to_be_removed

    return sac_buffer, sac_buffer_tracker


def train(
    env: gym.Env,
    test_env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    # ------------------- Initialization -------------------
    wandb = None
    debug_mode = True#cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    if cfg.wandb_log:
        import wandb

        wnb_cfg = OmegaConf.to_container(
            cfg, resolve=True,
        )

        wandb.init(
            # set the wandb project where this run will be logged
            project=cfg.wandb_project,

            # track hyperparameters and run metadata
            config=wnb_cfg,
            job_type="train",
            group=cfg.algorithm.name,
            name=cfg.experiment
        )

        # define metrics to be logged
        # rollout metrics
        wandb.define_metric("rollout/env_step")
        wandb.define_metric("rollout/average_length", step_metric="rollout/env_step")
        wandb.define_metric("rollout/minimum_length", step_metric="rollout/env_step")
        wandb.define_metric("rollout/maximum_length", step_metric="rollout/env_step")
        wandb.define_metric("rollout/added_transitions", step_metric="rollout/env_step")
        for obs_dim in range(obs_shape[0]):
            wandb.define_metric("rollout/obs_dim_" + str(obs_dim) + "_quant_ent", step_metric="rollout/env_step")
            wandb.define_metric("rollout/loss_rate_limit_" + str(obs_dim), step_metric="rollout/env_step")
            wandb.define_metric("rollout/loss_limit_" + str(obs_dim), step_metric="rollout/env_step")
        wandb.define_metric("rollout/reward_quant_ent", step_metric="rollout/env_step")
        wandb.define_metric("rollout/loss_rate_limit_reward", step_metric="rollout/env_step")
        wandb.define_metric("rollout/loss_limit_reward", step_metric="rollout/env_step")
        wandb.define_metric("rollout/rollout_length_histogram", step_metric="rollout/env_step")

        # agent training metrics
        wandb.define_metric("agent_train/step")
        wandb.define_metric("agent_train/batch_reward", step_metric="agent_train/step")
        wandb.define_metric("agent_train/actor_loss", step_metric="agent_train/step")
        wandb.define_metric("agent_train/actor_target_entropy", step_metric="agent_train/step")
        wandb.define_metric("agent_train/critic_loss", step_metric="agent_train/step")
        wandb.define_metric("agent_train/alpha_loss", step_metric="agent_train/step")
        wandb.define_metric("agent_train/alpha_value", step_metric="agent_train/step")
        wandb.define_metric("agent_train/actor_entropy", step_metric="agent_train/step")

        # evaluation metrics
        wandb.define_metric("agent_eval/env_step")
        wandb.define_metric("agent_eval/episode_reward", step_metric="agent_eval/env_step")
        wandb.define_metric("agent_eval/avg_reward_over_training", step_metric="agent_eval/env_step")

        # model training metrics
        wandb.define_metric("model_train/epoch")
        wandb.define_metric("model_train/train_dataset_size", step_metric="model_train/epoch")
        wandb.define_metric("model_train/val_dataset_size", step_metric="model_train/epoch")
        wandb.define_metric("model_train/model_loss", step_metric="model_train/epoch")
        wandb.define_metric("model_train/model_val_score", step_metric="model_train/epoch")
        wandb.define_metric("model_train/model_best_val_score", step_metric="model_train/epoch")

        wandb.define_metric("model_train_round/env_step")
        wandb.define_metric("model_train_round/epochs_trained", step_metric="model_train_round/env_step")

        # environment buffer metrics
        wandb.define_metric("env_buffer/env_step")
        for obs_dim in range(obs_shape[0]):
            wandb.define_metric("env_buffer/obs_dim_" + str(obs_dim) + "_histogram", step_metric="env_buffer/env_step")
        wandb.define_metric("env_buffer/reward_histogram", step_metric="env_buffer/env_step")

        # SAC buffer metrics
        wandb.define_metric("sac_buffer/env_step")
        for obs_dim in range(obs_shape[0]):
            wandb.define_metric("sac_buffer/obs_dim_" + str(obs_dim) + "_histogram", step_metric="sac_buffer/env_step")
            wandb.define_metric("sac_buffer/info_lost_current_step_" + str(obs_dim) + "_histogram", step_metric="sac_buffer/env_step")
            wandb.define_metric("sac_buffer/info_lost_total_" + str(obs_dim) + "_histogram", step_metric="sac_buffer/env_step")
            wandb.define_metric("sac_buffer/label_" + str(obs_dim) + "_histogram", step_metric="sac_buffer/env_step")

        wandb.define_metric("sac_buffer/reward_histogram", step_metric="sac_buffer/env_step")
        wandb.define_metric("sac_buffer/info_lost_current_step_reward_histogram", step_metric="sac_buffer/env_step")
        wandb.define_metric("sac_buffer/info_lost_total_reward_histogram", step_metric="sac_buffer/env_step")
        wandb.define_metric("sac_buffer/label_reward_histogram", step_metric="sac_buffer/env_step")

    mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)
    agent = SACAgent(
        cast(pytorch_sac_pranz24.SAC, hydra.utils.instantiate(cfg.algorithm.agent))
    )

    work_dir = work_dir or os.getcwd()
    # enable_back_compatible to use pytorch_sac agent
    logger = mbrl.util.Logger(work_dir, enable_back_compatible=True)
    logger.register_group(
        mbrl.constants.RESULTS_LOG_NAME,
        MBPO_LOG_FORMAT,
        color="green",
        dump_frequency=1,
    )

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    np.random.seed(cfg.seed)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    # -------------- Create initial overrides. dataset --------------
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    random_explore = cfg.algorithm.random_initial_explore
    mbrl.util.common.rollout_agent_trajectories(
        env,
        cfg.algorithm.initial_exploration_steps,
        mbrl.planning.RandomAgent(env) if random_explore else agent,
        {} if random_explore else {"sample": True, "batched": False},
        replay_buffer=replay_buffer,
    )

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    rollout_batch_size = (
        cfg.overrides.effective_model_rollouts_per_step * cfg.algorithm.freq_train_model
    )
    trains_per_epoch = int(
        np.ceil(cfg.overrides.epoch_length / cfg.overrides.freq_train_model)
    )
    updates_made = 0
    env_steps = 0
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, None, generator=torch_generator
    )
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=None if silent else logger,
        wandb_log = cfg.wandb_log,
        wandb = wandb
    )
    best_eval_reward = -np.inf
    epoch = 0
    sac_buffer = None
    rollout_tracker = []
    normalize_agent = cfg.algorithm.get("normalize_agent", False)
    if normalize_agent: agent.sac_agent.normalizer = dynamics_model.input_normalizer
    mbrl.util.common.train_model_and_save_model_and_data(
                                                            dynamics_model,
                                                            model_trainer,
                                                            cfg.overrides,
                                                            replay_buffer,
                                                            work_dir=None,
                                                        )
    rollout_length = int(
        mbrl.util.math.truncated_linear(
            *(cfg.overrides.rollout_schedule + [epoch + 1])
        )
    )
    sac_buffer_capacity = rollout_length * rollout_batch_size * trains_per_epoch
    sac_buffer_capacity *= cfg.overrides.num_epochs_to_retain_sac_buffer
    sac_buffer = maybe_replace_sac_buffer(
        sac_buffer, obs_shape, act_shape, sac_buffer_capacity, cfg.seed
    )
    num_added = rollout_model_and_populate_sac_buffer(
                                                        model_env,
                                                        replay_buffer,
                                                        agent,
                                                        sac_buffer,
                                                        cfg.algorithm.sac_samples_action,
                                                        rollout_length,
                                                        rollout_batch_size,
                                                        env_steps,
                                                        cfg.wandb_log,
                                                        wandb,
                                                        rng,
                                                        act_shape
                                                    )
    rollout_tracker.append(num_added)
    n_evals = 0
    sum_rewards = 0
    while env_steps < cfg.overrides.num_steps:
        rollout_length = int(
            mbrl.util.math.truncated_linear(
                *(cfg.overrides.rollout_schedule + [epoch + 1])
            )
        )
        sac_buffer_capacity = rollout_length * rollout_batch_size * trains_per_epoch
        sac_buffer_capacity *= cfg.overrides.num_epochs_to_retain_sac_buffer
        sac_buffer = maybe_replace_sac_buffer(
            sac_buffer, obs_shape, act_shape, sac_buffer_capacity, cfg.seed
        )
        obs = None
        terminated = False
        truncated = False
        action_noise = cn.powerlaw_psd_gaussian(1, (env.action_space.shape[0], cfg.overrides.epoch_length), random_state=rng)
        for steps_epoch in range(cfg.overrides.epoch_length):
            if steps_epoch == 0 or terminated or truncated:
                steps_epoch = 0
                obs, _ = env.reset()
                terminated = False
                truncated = False

            (
                next_obs,
                reward,
                terminated,
                truncated,
                _,
            ) = mbrl.util.common.step_env_and_add_to_buffer_eps(
                env, obs, agent, {}, replay_buffer, eps=action_noise[:, steps_epoch]
            )

            # --------------- Model Training -----------------
            if (env_steps + 1) % cfg.overrides.freq_train_model == 0:
                mbrl.util.common.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    replay_buffer,
                    work_dir=None,
                )

                # --------- Rollout new model and store imagined trajectories --------
                # Batch all rollouts for the next freq_train_model steps together
                if len(rollout_tracker) >= trains_per_epoch * cfg.overrides.num_epochs_to_retain_sac_buffer:
                    sac_buffer, rollout_tracker = remove_old_transitions(sac_buffer, rollout_tracker)
                num_added = rollout_model_and_populate_sac_buffer(
                    model_env,
                    replay_buffer,
                    agent,
                    sac_buffer,
                    cfg.algorithm.sac_samples_action,
                    rollout_length,
                    rollout_batch_size,
                    env_steps,
                    cfg.wandb_log,
                    wandb,
                    rng,
                    act_shape
                )

                rollout_tracker.append(num_added)

                if debug_mode:
                    print(
                        f"Epoch: {epoch}. "
                        f"SAC buffer size: {len(sac_buffer)}. "
                        f"Rollout length: {rollout_length}. "
                        f"Steps: {env_steps}"
                    )

            # --------------- Agent Training -----------------
            num_to_be_trained = cfg.overrides.num_sac_updates_per_step
            if num_to_be_trained < 1: num_to_be_trained = 1
            for _ in range(num_to_be_trained):
                use_real_data = rng.random() < cfg.algorithm.real_data_ratio
                which_buffer = replay_buffer if use_real_data else sac_buffer
                if (env_steps + 1) % cfg.overrides.sac_updates_every_steps != 0 or len(
                    which_buffer
                ) < cfg.overrides.sac_batch_size:
                    break  # only update every once in a while

                agent.sac_agent.update_parameters(
                    which_buffer,
                    cfg.overrides.sac_batch_size,
                    updates_made,
                    logger,
                    reverse_mask=True,
                )
                updates_made += 1
                if not silent and updates_made % cfg.log_frequency_agent == 0:
                    if cfg.wandb_log:
                        # wandb logging for agent training metrics
                        wandb.log(
                            data = {
                                    "agent_train/step": updates_made,
                                    "agent_train/batch_reward": logger._groups["train"][0]._meters["batch_reward"].value(),
                                    "agent_train/actor_loss": logger._groups["train"][0]._meters["actor_loss"].value(),
                                    "agent_train/actor_target_entropy": logger._groups["train"][0]._meters["actor_target_entropy"].value(),
                                    "agent_train/critic_loss": logger._groups["train"][0]._meters["critic_loss"].value(),
                                    "agent_train/alpha_loss": logger._groups["train"][0]._meters["alpha_loss"].value(),
                                    "agent_train/alpha_value": logger._groups["train"][0]._meters["alpha_value"].value(),
                                    "agent_train/actor_entropy": logger._groups["train"][0]._meters["actor_entropy"].value()
                                    }
                        )
                    logger.dump(updates_made, save=True)

            # ------ Epoch ended (evaluate and save model) ------
            if (env_steps + 1) % cfg.overrides.epoch_length == 0:
                avg_reward = evaluate(
                    test_env, agent, cfg.algorithm.num_eval_episodes,
                )
                if (env_steps + 1) >= 100000:
                    sum_rewards += avg_reward
                    n_evals += 1
                    avg_reward_over_training = sum_rewards/n_evals
                else:
                    avg_reward_over_training = 0
                logger.log_data(
                    mbrl.constants.RESULTS_LOG_NAME,
                    {
                        "epoch": epoch,
                        "env_step": env_steps,
                        "episode_reward": avg_reward,
                        "rollout_length": rollout_length,
                    },
                )
                if cfg.wandb_log:
                    # wandb logging for evaluation and buffer metrics
                    data = {
                        "agent_eval/env_step": env_steps + 1,
                        "agent_eval/episode_reward": avg_reward,
                        "agent_eval/avg_reward_over_training": avg_reward_over_training
                    }
                    # obs, _, _, rewards, _, _ = sac_buffer.get_all().astuple()
                    tr_batch, info_lost_current_step, info_lost_total, label = sac_buffer.info_sample(1000000)
                    # label = label / label.sum(axis=0, keepdims=True)
                    obs, _, _, rewards, _, _ = tr_batch.astuple()
                    env_obs, _, _, env_rewards, _, _ = replay_buffer.get_all().astuple()
                    data["sac_buffer/env_step"] = env_steps + 1
                    data["env_buffer/env_step"] = env_steps + 1

                    for obs_dim in range(obs.shape[-1]):
                        var_hist_data = obs[:, obs_dim]
                        var_hist = np.histogram(var_hist_data, bins=512, density=True,)
                        data["sac_buffer/obs_dim_" + str(obs_dim) + "_histogram"] = wandb.Histogram(np_histogram=var_hist)

                        var_hist_data = info_lost_current_step[:, obs_dim]
                        var_hist = np.histogram(var_hist_data, bins=512, density=True, )
                        data["sac_buffer/info_lost_current_step_" + str(obs_dim) + "_histogram"] = wandb.Histogram(np_histogram=var_hist)

                        var_hist_data = info_lost_total[:, obs_dim]
                        var_hist = np.histogram(var_hist_data, bins=512, density=True, )
                        data["sac_buffer/info_lost_total_" + str(obs_dim) + "_histogram"] = wandb.Histogram(np_histogram=var_hist)

                        var_hist_data = label[:, obs_dim]
                        var_hist = np.histogram(var_hist_data, bins=512, density=True, )
                        data["sac_buffer/label_" + str(obs_dim) + "_histogram"] = wandb.Histogram(np_histogram=var_hist)
                
                        var_hist_data = env_obs[:, obs_dim]
                        var_hist = np.histogram(var_hist_data, bins=512, density=True)
                        data["env_buffer/obs_dim_" + str(obs_dim) + "_histogram"] = wandb.Histogram(np_histogram=var_hist)



                    var_hist_data = rewards[:]
                    var_hist = np.histogram(var_hist_data, bins=512, density=True, )
                    data["sac_buffer/reward_histogram"] = wandb.Histogram(np_histogram=var_hist)
                    
                    var_hist_data = info_lost_current_step[:,-1]
                    var_hist = np.histogram(var_hist_data, bins=512, density=True, )
                    data["sac_buffer/info_lost_current_step_reward_histogram"] = wandb.Histogram(np_histogram=var_hist)
                    var_hist_data = info_lost_total[:,-1]
                    var_hist = np.histogram(var_hist_data, bins=512, density=True, )
                    data["sac_buffer/info_lost_total_reward_histogram"] = wandb.Histogram(np_histogram=var_hist)
                    var_hist_data = label[:,-1]
                    var_hist = np.histogram(var_hist_data, bins=512, density=True, )
                    data["sac_buffer/label_reward_histogram"] = wandb.Histogram(np_histogram=var_hist)
                    var_hist_data = env_rewards[:]
                    var_hist = np.histogram(var_hist_data, bins=512, density=True)
                    data["env_buffer/reward_histogram"] = wandb.Histogram(np_histogram=var_hist)

                    wandb.log(
                        data=data
                    )
                if avg_reward > best_eval_reward:
                    best_eval_reward = avg_reward
                    agent.sac_agent.save_checkpoint(
                        ckpt_path=os.path.join(work_dir, "sac.pth")
                    )
                    
                epoch += 1

            env_steps += 1
            obs = next_obs
    if cfg.wandb_log:
        wandb.finish()
    return np.float32(best_eval_reward)