import copy
import functools
import logging
import math
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree

import ray
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.core.rl_module import RLModule
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import _directStepOptimizerSingleton
from ray.rllib.utils import NullContextManager, force_list
from ray.rllib.utils.annotations import (
    DeveloperAPI,
    OverrideToImplementCustomLogic,
    OverrideToImplementCustomLogic_CallToSuperRecommended,
    is_overridden,
    override,
)
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.utils.error import ERR_MSG_TORCH_POLICY_CANNOT_SAVE_MODEL
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import (
    DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY,
    NUM_AGENT_STEPS_TRAINED,
    NUM_GRAD_UPDATES_LIFETIME,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import normalize_action
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import (
    AlgorithmConfigDict,
    GradInfoDict,
    ModelGradients,
    ModelWeights,
    PolicyState,
    TensorStructType,
    TensorType,
)

if TYPE_CHECKING:
    from ray.rllib.evaluation import Episode  # noqa

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.numpy import SMALL_NUMBER
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.from_config import from_config

import torch.multiprocessing as multiprocessing
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass
import queue
import redis
import pickle


class AsyncLearner(multiprocessing.Process):
    def __init__(
        self,
        learner_idx,
        init_state_dict,
        config,
        distributed_world_size,
        observation_space,
        action_space,
        framework,
        is_recurrent,
        device_idx,
        device,
        shared_dict,
    ):
        multiprocessing.Process.__init__(self)
        self.daemon = True
        self.stop = False

        self.learner_idx = learner_idx
        self.config = config
        if self.config["algo_name"] in ["ppo", "ppo_serverless"]:
            self.loss = self.ppo_loss
        elif self.config["algo_name"] in ["appo", "appo_serverless"]:
            self.loss = self.appo_loss
        elif self.config["algo_name"] in ["impala", "impala_serverless"]:
            self.loss = self.impala_loss

        self.device_idx = device_idx
        self.device = device

        # Init model
        dist_class, logit_dim = ModelCatalog.get_action_dist(
            action_space, config["model"], framework=framework
        )
        model = ModelCatalog.get_model_v2(
            obs_space=observation_space,
            action_space=action_space,
            num_outputs=logit_dim,
            model_config=config["model"],
            framework=framework,
        )
        self.model = model.to(device)
        self.model.load_state_dict(init_state_dict)
        
        self.is_model = copy.deepcopy(self.model).to(device)
        self.is_model_weights = []

        self.target_model = copy.deepcopy(self.model).to(device)
        self.dist_class = dist_class
        self.distributed_world_size = distributed_world_size
        self.framework = framework
        self.observation_space = observation_space
        self.action_space = action_space
        self.is_recurrent = is_recurrent

        # Init optimizers
        optimizers = [
            torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        ]
        self.exploration = self._create_exploration()
        optimizers = self.exploration.get_exploration_optimizer(optimizers)
        self.optimizers = force_list(optimizers)

        self.shared_dict = shared_dict

        self.grad_clock = 0

    def set_weights(self, weights: ModelWeights) -> None:
        weights = convert_to_torch_tensor(weights, device=self.device)
        if self.config.get("_enable_new_api_stack", False):
            self.model.set_state(weights)
        else:
            self.model.load_state_dict(weights)
    
    def set_is_weights(self, weights: ModelWeights) -> None:
        weights = convert_to_torch_tensor(weights, device=self.device)
        if self.config.get("_enable_new_api_stack", False):
            self.is_model.set_state(weights)
        else:
            self.is_model.load_state_dict(weights)

    def get_weights(self) -> ModelWeights:
        return {k: v.cpu().detach().numpy() for k, v in self.model.state_dict().items()}

    def _create_exploration(self) -> Exploration:
        """Creates the Policy's Exploration object.

        This method only exists b/c some Algorithms do not use TfPolicy nor
        TorchPolicy, but inherit directly from Policy. Others inherit from
        TfPolicy w/o using DynamicTFPolicy.
        TODO(sven): unify these cases.

        Returns:
            Exploration: The Exploration object to be used by this Policy.
        """
        if getattr(self, "exploration", None) is not None:
            return self.exploration

        exploration = from_config(
            Exploration,
            self.config.get("exploration_config", {"type": "StochasticSampling"}),
            action_space=self.action_space,
            policy_config=self.config,
            model=getattr(self, "model", None),
            num_workers=self.config.get("num_workers", 0),
            worker_index=self.config.get("worker_index", 0),
            framework=getattr(self, "framework", self.config.get("framework", "tf")),
        )
        return exploration

    def apply_grad_clipping(
        self,
        grad_clip, 
        optimizer, 
        loss: TensorType
    ) -> Dict[str, TensorType]:
        """Applies gradient clipping to already computed grads inside `optimizer`.

        Note: This function does NOT perform an analogous operation as
        tf.clip_by_global_norm. It merely clips by norm (per gradient tensor) and
        then computes the global norm across all given tensors (but without clipping
        by that global norm).

        Args:
            policy: The TorchPolicy, which calculated `loss`.
            optimizer: A local torch optimizer object.
            loss: The torch loss tensor.

        Returns:
            An info dict containing the "grad_norm" key and the resulting clipped
            gradients.
        """
        grad_gnorm = 0
        if grad_clip is not None:
            clip_value = grad_clip
        else:
            clip_value = np.inf

        num_none_grads = 0
        for param_group in optimizer.param_groups:
            # Make sure we only pass params with grad != None into torch
            # clip_grad_norm_. Would fail otherwise.
            params = list(filter(lambda p: p.grad is not None, param_group["params"]))
            if params:
                # PyTorch clips gradients inplace and returns the norm before clipping
                # We therefore need to compute grad_gnorm further down (fixes #4965)
                global_norm = nn.utils.clip_grad_norm_(params, clip_value)

                if isinstance(global_norm, torch.Tensor):
                    global_norm = global_norm.cpu().numpy()

                grad_gnorm += min(global_norm, clip_value)
            else:
                num_none_grads += 1

        # Note (Kourosh): grads could indeed be zero. This method should still return
        # grad_gnorm in that case.
        if num_none_grads == len(optimizer.param_groups):
            # No grads available
            return {}
        return {"grad_gnorm": grad_gnorm}

    def extra_grad_process(
        self,
        grad_clip,
        local_optimizer,
        loss
    ):
        return self.apply_grad_clipping(
            grad_clip=grad_clip,
            optimizer=local_optimizer,
            loss=loss,
        )

    def explained_variance(
        self,
        y: TensorType, 
        pred: TensorType
    ) -> TensorType:
        """Computes the explained variance for a pair of labels and predictions.

        The formula used is:
        max(-1.0, 1.0 - (std(y - pred)^2 / std(y)^2))

        Args:
            y: The labels.
            pred: The predictions.

        Returns:
            The explained variance given a pair of labels and predictions.
        """
        y_var = torch.var(y, dim=[0])
        diff_var = torch.var(y - pred, dim=[0])
        min_ = torch.tensor([-1.0]).to(pred.device)
        return torch.max(min_, 1 - (diff_var / y_var + SMALL_NUMBER))[0]

    def make_time_major(
        self,
        rollout_fragment_length, 
        is_recurrent,
        seq_lens, 
        tensor
    ):
        """Swaps batch and trajectory axis.

        Args:
            policy: Policy reference
            seq_lens: Sequence lengths if recurrent or None
            tensor: A tensor or list of tensors to reshape.

        Returns:
            res: A tensor with swapped axes or a list of tensors with
            swapped axes.
        """
        if isinstance(tensor, (list, tuple)):
            return [self.make_time_major(rollout_fragment_length, is_recurrent, seq_lens, t) for t in tensor]

        if is_recurrent:
            B = seq_lens.shape[0]
            T = tensor.shape[0] // B
        else:
            # Important: chop the tensor into batches at known episode cut
            # boundaries.
            # TODO: (sven) this is kind of a hack and won't work for
            #  batch_mode=complete_episodes.
            T = rollout_fragment_length
            B = tensor.shape[0] // T
        rs = torch.reshape(tensor, [B, T] + list(tensor.shape[1:]))

        # Swap B and T axes.
        res = torch.transpose(rs, 1, 0)

        return res

    def _make_time_major(self, *args, **kwargs):
        return self.make_time_major(
            rollout_fragment_length=self.config["rollout_fragment_length"], 
            is_recurrent=self.is_recurrent,
            seq_lens=train_batch.get(SampleBatch.SEQ_LENS), 
            *args, 
            **kwargs
        )

    def update_target(self, tau=None):
        # Update_target_fn will be called periodically to copy Q network to
        # target Q network, using (soft) tau-synching.
        tau = tau or self.config.get("tau", 1.0)

        # Support partial (soft) synching.
        # If tau == 1.0: Full sync from Q-model to target Q-model.
        new_state_dict = {
            k: tau * self.model.state_dict()[k] + (1 - tau) * v
            for k, v in self.target_model.state_dict().items()
        }
        self.target_model.load_state_dict(new_state_dict)

    def ppo_loss(
        self,
        model: ModelV2,
        dist_class,
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Compute loss for Proximal Policy Objective.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The PPO loss tensor given the input batch.
        """

        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        if self.config["use_is_truncation"]:
            logp_ratio_list = [logp_ratio.clone().detach()]
            for is_weights in self.is_model_weights:
                self.set_is_weights(is_weights)
                self.is_model.eval()
                is_logits, _ = self.is_model(train_batch)
                is_curr_action_dist = dist_class(is_logits, self.is_model)

                is_logp_ratio = torch.exp(
                    is_curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
                    - train_batch[SampleBatch.ACTION_LOGP]
                )
                logp_ratio_list.append(is_logp_ratio.detach())
            
            stacked_logp_ratio = torch.stack(logp_ratio_list, dim=0)
            min_logp_ratio, _ = torch.min(stacked_logp_ratio, dim=0)
            min_logp_ratio = torch.clamp_min(min_logp_ratio, self.config["serverless_is_ratio"])
            logp_ratio = torch.minimum(logp_ratio, min_logp_ratio)

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            # TODO smorad: should we do anything besides warn? Could discard KL term
            # for this update
            # warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES]
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )

        # Compute a value function loss.
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            vf_loss_clipped = mean_vf_loss = torch.tensor(0.0).to(surrogate_loss.device)

        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.config["entropy_coeff"] * curr_entropy
        )

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.config["kl_coeff"] * mean_kl_loss

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        tower_stats = {}
        tower_stats["total_loss"] = total_loss.clone().detach()
        tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss).clone().detach()
        tower_stats["mean_vf_loss"] = mean_vf_loss.clone().detach()
        tower_stats["vf_explained_var"] = self.explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        ).clone().detach()
        tower_stats["mean_entropy"] = mean_entropy.clone().detach()
        tower_stats["mean_kl_loss"] = mean_kl_loss.clone().detach()

        return total_loss, tower_stats

    def appo_loss(
        self,
        model: ModelV2,
        dist_class,
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Constructs the loss for APPO.

        With IS modifications and V-trace for Advantage Estimation.

        Args:
            model (ModelV2): The Model to calculate the loss for.
            dist_class (Type[ActionDistribution]): The action distr. class.
            train_batch: The training data.

        Returns:
            Union[TensorType, List[TensorType]]: A single loss tensor or a list
                of loss tensors.
        """
        import ray.rllib.algorithms.impala.vtrace_torch as vtrace

        model_out, _ = model(train_batch)
        action_dist = dist_class(model_out, model)

        if isinstance(self.action_space, gym.spaces.Discrete):
            is_multidiscrete = False
            output_hidden_shape = [self.action_space.n]
        elif isinstance(self.action_space, gym.spaces.multi_discrete.MultiDiscrete):
            is_multidiscrete = True
            output_hidden_shape = self.action_space.nvec.astype(np.int32)
        else:
            is_multidiscrete = False
            output_hidden_shape = 1

        actions = train_batch[SampleBatch.ACTIONS]
        dones = train_batch[SampleBatch.TERMINATEDS]
        rewards = train_batch[SampleBatch.REWARDS]
        behaviour_logits = train_batch[SampleBatch.ACTION_DIST_INPUTS]

        self.target_model_out, _ = self.target_model(train_batch)

        prev_action_dist = dist_class(behaviour_logits, model)
        values = model.value_function()
        values_time_major = self._make_time_major(values)
        bootstrap_values_time_major = self._make_time_major(
            train_batch[SampleBatch.VALUES_BOOTSTRAPPED]
        )
        bootstrap_value = bootstrap_values_time_major[-1]

        if self.is_recurrent:
            max_seq_len = torch.max(train_batch[SampleBatch.SEQ_LENS])
            mask = sequence_mask(train_batch[SampleBatch.SEQ_LENS], max_seq_len)
            mask = torch.reshape(mask, [-1])
            mask = self._make_time_major(mask)
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        else:
            reduce_mean_valid = torch.mean

        if self.config["vtrace"]:
            logger.debug("Using V-Trace surrogate loss (vtrace=True)")

            old_policy_behaviour_logits = self.target_model_out.detach()
            old_policy_action_dist = dist_class(old_policy_behaviour_logits, model)

            if isinstance(output_hidden_shape, (list, tuple, np.ndarray)):
                unpacked_behaviour_logits = torch.split(
                    behaviour_logits, list(output_hidden_shape), dim=1
                )
                unpacked_old_policy_behaviour_logits = torch.split(
                    old_policy_behaviour_logits, list(output_hidden_shape), dim=1
                )
            else:
                unpacked_behaviour_logits = torch.chunk(
                    behaviour_logits, output_hidden_shape, dim=1
                )
                unpacked_old_policy_behaviour_logits = torch.chunk(
                    old_policy_behaviour_logits, output_hidden_shape, dim=1
                )

            # Prepare actions for loss.
            loss_actions = (
                actions if is_multidiscrete else torch.unsqueeze(actions, dim=1)
            )

            # Prepare KL for loss.
            action_kl = self._make_time_major(old_policy_action_dist.kl(action_dist))

            # Compute vtrace on the CPU for better perf.
            vtrace_returns = vtrace.multi_from_logits(
                behaviour_policy_logits=self._make_time_major(unpacked_behaviour_logits),
                target_policy_logits=self._make_time_major(
                    unpacked_old_policy_behaviour_logits
                ),
                actions=torch.unbind(self._make_time_major(loss_actions), dim=2),
                discounts=(1.0 - self._make_time_major(dones).float())
                * self.config["gamma"],
                rewards=self._make_time_major(rewards),
                values=values_time_major,
                bootstrap_value=bootstrap_value,
                dist_class=TorchCategorical if is_multidiscrete else dist_class,
                model=model,
                clip_rho_threshold=self.config["vtrace_clip_rho_threshold"],
                clip_pg_rho_threshold=self.config["vtrace_clip_pg_rho_threshold"],
            )

            actions_logp = self._make_time_major(action_dist.logp(actions))
            prev_actions_logp = self._make_time_major(prev_action_dist.logp(actions))
            old_policy_actions_logp = self._make_time_major(
                old_policy_action_dist.logp(actions)
            )
            is_ratio = torch.clamp(
                torch.exp(prev_actions_logp - old_policy_actions_logp), 0.0, 2.0
            )
            logp_ratio = is_ratio * torch.exp(actions_logp - prev_actions_logp)

            advantages = vtrace_returns.pg_advantages.to(logp_ratio.device)
            surrogate_loss = torch.min(
                advantages * logp_ratio,
                advantages
                * torch.clamp(
                    logp_ratio,
                    1 - self.config["clip_param"],
                    1 + self.config["clip_param"],
                ),
            )

            mean_kl_loss = reduce_mean_valid(action_kl)
            mean_policy_loss = -reduce_mean_valid(surrogate_loss)

            # The value function loss.
            value_targets = vtrace_returns.vs.to(values_time_major.device)
            delta = values_time_major - value_targets
            mean_vf_loss = 0.5 * reduce_mean_valid(torch.pow(delta, 2.0))

            # The entropy loss.
            mean_entropy = reduce_mean_valid(self._make_time_major(action_dist.entropy()))

        else:
            logger.debug("Using PPO surrogate loss (vtrace=False)")

            # Prepare KL for Loss
            action_kl = self._make_time_major(prev_action_dist.kl(action_dist))

            actions_logp = self._make_time_major(action_dist.logp(actions))
            prev_actions_logp = self._make_time_major(prev_action_dist.logp(actions))
            logp_ratio = torch.exp(actions_logp - prev_actions_logp)

            advantages = self._make_time_major(train_batch[Postprocessing.ADVANTAGES])
            surrogate_loss = torch.min(
                advantages * logp_ratio,
                advantages
                * torch.clamp(
                    logp_ratio,
                    1 - self.config["clip_param"],
                    1 + self.config["clip_param"],
                ),
            )

            mean_kl_loss = reduce_mean_valid(action_kl)
            mean_policy_loss = -reduce_mean_valid(surrogate_loss)

            # The value function loss.
            value_targets = self._make_time_major(train_batch[Postprocessing.VALUE_TARGETS])
            delta = values_time_major - value_targets
            mean_vf_loss = 0.5 * reduce_mean_valid(torch.pow(delta, 2.0))

            # The entropy loss.
            mean_entropy = reduce_mean_valid(self._make_time_major(action_dist.entropy()))

        # The summed weighted loss.
        total_loss = mean_policy_loss - mean_entropy * self.config["entropy_coeff"]
        # Optional additional KL Loss
        if self.config["use_kl_loss"]:
            total_loss += self.config["kl_coeff"] * mean_kl_loss

        # Optional vf loss (or in a separate term due to separate
        # optimizers/networks).
        loss_wo_vf = total_loss
        if not self.config["_separate_vf_optimizer"]:
            total_loss += mean_vf_loss * self.config["vf_loss_coeff"]

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        tower_stats = {}
        tower_stats["total_loss"] = total_loss.clone().detach()
        tower_stats["mean_policy_loss"] = mean_policy_loss.clone().detach()
        tower_stats["mean_kl_loss"] = mean_kl_loss.clone().detach()
        tower_stats["mean_vf_loss"] = mean_vf_loss.clone().detach()
        tower_stats["mean_entropy"] = mean_entropy.clone().detach()
        tower_stats["value_targets"] = value_targets.clone().detach()
        tower_stats["vf_explained_var"] = self.explained_variance(
            torch.reshape(value_targets, [-1]),
            torch.reshape(values_time_major, [-1]),
        ).clone().detach()

        # Return one total loss or two losses: vf vs rest (policy + kl).
        if self.config["_separate_vf_optimizer"]:
            return [loss_wo_vf, mean_vf_loss], tower_stats
        else:
            return total_loss, tower_stats
    
    def impala_loss(
        self,
        model: ModelV2,
        dist_class,
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        import ray.rllib.algorithms.impala.vtrace_torch as vtrace
        from ray.rllib.algorithms.impala.impala_torch_policy import VTraceLoss

        model_out, _ = model(train_batch)
        action_dist = dist_class(model_out, model)

        if isinstance(self.action_space, gym.spaces.Discrete):
            is_multidiscrete = False
            output_hidden_shape = [self.action_space.n]
        elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
            is_multidiscrete = True
            output_hidden_shape = self.action_space.nvec.astype(np.int32)
        else:
            is_multidiscrete = False
            output_hidden_shape = 1

        actions = train_batch[SampleBatch.ACTIONS]
        dones = train_batch[SampleBatch.TERMINATEDS]
        rewards = train_batch[SampleBatch.REWARDS]
        behaviour_action_logp = train_batch[SampleBatch.ACTION_LOGP]
        behaviour_logits = train_batch[SampleBatch.ACTION_DIST_INPUTS]
        if isinstance(output_hidden_shape, (list, tuple, np.ndarray)):
            unpacked_behaviour_logits = torch.split(
                behaviour_logits, list(output_hidden_shape), dim=1
            )
            unpacked_outputs = torch.split(model_out, list(output_hidden_shape), dim=1)
        else:
            unpacked_behaviour_logits = torch.chunk(
                behaviour_logits, output_hidden_shape, dim=1
            )
            unpacked_outputs = torch.chunk(model_out, output_hidden_shape, dim=1)
        values = model.value_function()
        values_time_major = self._make_time_major(values)
        bootstrap_values_time_major = self._make_time_major(
            train_batch[SampleBatch.VALUES_BOOTSTRAPPED]
        )
        bootstrap_value = bootstrap_values_time_major[-1]

        if self.is_recurrent:
            max_seq_len = torch.max(train_batch[SampleBatch.SEQ_LENS])
            mask_orig = sequence_mask(train_batch[SampleBatch.SEQ_LENS], max_seq_len)
            mask = torch.reshape(mask_orig, [-1])
        else:
            mask = torch.ones_like(rewards)

        # Prepare actions for loss.
        loss_actions = actions if is_multidiscrete else torch.unsqueeze(actions, dim=1)

        # Inputs are reshaped from [B * T] => [(T|T-1), B] for V-trace calc.
        loss = VTraceLoss(
            actions=self._make_time_major(loss_actions),
            actions_logp=self._make_time_major(action_dist.logp(actions)),
            actions_entropy=self._make_time_major(action_dist.entropy()),
            dones=self._make_time_major(dones),
            behaviour_action_logp=self._make_time_major(behaviour_action_logp),
            behaviour_logits=self._make_time_major(unpacked_behaviour_logits),
            target_logits=self._make_time_major(unpacked_outputs),
            discount=self.config["gamma"],
            rewards=self._make_time_major(rewards),
            values=values_time_major,
            bootstrap_value=bootstrap_value,
            dist_class=TorchCategorical if is_multidiscrete else dist_class,
            model=model,
            valid_mask=self._make_time_major(mask),
            config=self.config,
            vf_loss_coeff=self.config["vf_loss_coeff"],
            entropy_coeff=self.config["entropy_coeff"],
            clip_rho_threshold=self.config["vtrace_clip_rho_threshold"],
            clip_pg_rho_threshold=self.config["vtrace_clip_pg_rho_threshold"],
        )

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        tower_stats = {}
        tower_stats["pi_loss"] = loss.pi_loss.clone().detach()
        tower_stats["vf_loss"] = loss.vf_loss.clone().detach()
        tower_stats["entropy"] = loss.entropy.clone().detach()
        tower_stats["mean_entropy"] = loss.mean_entropy.clone().detach()
        tower_stats["total_loss"] = loss.total_loss.clone().detach()

        values_batched = make_time_major(
            rollout_fragment_length=self.config["rollout_fragment_length"], 
            is_recurrent=self.is_recurrent, 
            seq_lens=train_batch.get(SampleBatch.SEQ_LENS), 
            tensor=values,
        )
        tower_stats["vf_explained_var"] = self.explained_variance(
            torch.reshape(loss.value_targets, [-1]), torch.reshape(values_batched, [-1])
        ).clone().detach()

        if self.config.get("_separate_vf_optimizer"):
            return [loss.loss_wo_vf, loss.vf_loss], tower_stats
        else:
            return loss.total_loss, tower_stats

    def run(self):
        self.redis_pool = redis.ConnectionPool(
            host=self.config["redis_host"], 
            port=self.config["redis_port"], 
        )
        self.redis_client = redis.Redis(connection_pool=self.redis_pool)

        while not self.stop:
            self.step()

    def set_model(self):
        self.redis_client.set("learner_{}_weights".format(self.learner_idx), pickle.dumps(self.get_weights()))

    def get_model(self):
        mget_list = [
            "grad_clock",
            "aggregator_weights",
        ]

        [redis_grad_clock, aggregator_weights] = self.redis_client.mget(mget_list)
        if redis_grad_clock:
            redis_grad_clock = int(redis_grad_clock)
            if self.grad_clock < redis_grad_clock:
                self.grad_clock = redis_grad_clock
                self.set_weights(pickle.loads(aggregator_weights))
                self.set_model()

            if self.config["algo_name"] in ["appo", "appo_serverless"]:
                assert self.target_model
                self.update_target()

        if self.config["use_is_truncation"]:
            self.get_is_models()

    def get_is_models(self):
        mget_list = []
        for i in range(self.config["num_async_learners"]):
            if i != self.learner_idx:
                mget_list.append("learner_{}_weights".format(i))
        
        is_model_weights = self.redis_client.mget(mget_list)
        if None not in is_model_weights:
            self.is_model_weights = [pickle.loads(m) for m in is_model_weights]

    def compute_grad(self, sample_batch):
        torch.set_grad_enabled(torch.is_grad_enabled())
        
        try:
            with NullContextManager() if self.device.type == "cpu" else torch.cuda.device(  # noqa: E501
                self.device
            ):
                learner_time_start = time.time()

                loss_out, tower_stats = self.loss(self.model, self.dist_class, sample_batch)
                loss_out = force_list(loss_out)

                # Call Model's custom-loss with Policy loss outputs and
                # train_batch.
                if hasattr(self.model, "custom_loss"):
                    loss_out = self.model.custom_loss(loss_out, sample_batch)

                # Loop through all optimizers.
                grad_info = {"allreduce_latency": 0.0}

                parameters = list(self.model.parameters())
                all_grads = [None for _ in range(len(parameters))]
                for opt_idx, opt in enumerate(self.optimizers):
                    # Erase gradients in all vars of the tower that this
                    # optimizer would affect.
                    opt.zero_grad()
                    
                    # Recompute gradients of loss over all variables.
                    loss_out[opt_idx].backward(retain_graph=True)
                    grad_info.update(
                        self.extra_grad_process(self.config["grad_clip"], opt, loss_out[opt_idx])
                    )

                    grads = []
                    # Note that return values are just references;
                    # Calling zero_grad would modify the values.
                    for param_idx, param in enumerate(parameters):
                        if param.grad is not None:
                            grads.append(param.grad)
                        all_grads[param_idx] = param.grad.clone().detach()

                    if self.distributed_world_size:
                        start = time.time()
                        if torch.cuda.is_available():
                            # Sadly, allreduce_coalesced does not work with
                            # CUDA yet.
                            for g in grads:
                                torch.distributed.all_reduce(
                                    g, op=torch.distributed.ReduceOp.SUM
                                )
                        else:
                            torch.distributed.all_reduce_coalesced(
                                grads, op=torch.distributed.ReduceOp.SUM
                            )

                        for param_group in opt.param_groups:
                            for p in param_group["params"]:
                                if p.grad is not None:
                                    p.grad /= self.distributed_world_size

                        grad_info["allreduce_latency"] += time.time() - start

                learner_time = time.time() - learner_time_start

                return (
                    (all_grads, grad_info), 
                    (self.learner_idx, sample_batch.num_grad_updates, learner_time, self.grad_clock), 
                    tower_stats
                )
                 
        except Exception as e:
            import traceback

            return (
                ValueError(
                    e.args[0]
                    + "\n traceback"
                    + traceback.format_exc()
                    + "\n"
                    + "In model {} on device {}".format(self.model, self.device)
                ), 
                e
            )

    def step(self):
        sample_batch = self.redis_client.rpop("sample_queue_{}".format(self.device_idx))
        if sample_batch:
            sample_batch = pickle.loads(sample_batch)
            self.get_model()
            self.model.train()

            grad_result = self.compute_grad(sample_batch)
            self.redis_client.lpush("grad_queue", pickle.dumps(grad_result))


class AsyncAggregator(threading.Thread):
    def __init__(
        self,
        policy,
        model,
        device,
        shared_dict,
    ):
        threading.Thread.__init__(self)
        self.daemon = True
        self.stop = False

        self.policy = policy
        self.config = self.policy.config
        self.model = model
        self.device = device
        self.shared_dict = shared_dict
        self.learners = list(self.policy.async_learners.values())
        
        self.local_queue = []
        self.local_result_queue = []
        self.to_process_queue = []
        self.to_process_staleness_list = []
        
        self.grad_clock = 0
        self.epoch = 0

        self.redis_client = self.policy.redis_client

    def run(self):
        while not self.stop:
            self.step()

    def set_model(self):
        self.redis_client.mset(
            {
                "grad_clock": self.grad_clock,
                "aggregator_weights": pickle.dumps(self.policy.get_weights()),
            }
        )
    
    def get_grads(self):
        grad_list = self.redis_client.rpop("grad_queue", 114514)
        if grad_list:
            grad_list = [pickle.loads(g) for g in grad_list]
            self.local_queue.extend(grad_list)

    def aggregate_grads(self, grad_results):
        # Collect gradients from completed learners
        tower_outputs = []
        learner_idxs = []
        learner_times = []
        grad_clock_list = []
        grad_staleness_list = []
        ssp_clock_list = []
        ssp_staleness_list = []
        ready_models_and_batches = {}

        for result in grad_results:
            grads, async_info, tower_stats = result
            (learner_idx, num_grad_updates_per_batch, learner_time, grad_clock) = async_info
            learner_model = self.policy.model_gpu_towers[learner_idx]
            learner_idxs.append(learner_idx)

            # Update model stats
            if self.config["algo_name"] in ["ppo", "ppo_serverless"]:
                learner_model.tower_stats["total_loss"] = tower_stats["total_loss"]
                learner_model.tower_stats["mean_policy_loss"] = tower_stats["mean_policy_loss"]
                learner_model.tower_stats["mean_vf_loss"] = tower_stats["mean_vf_loss"]
                learner_model.tower_stats["vf_explained_var"] = tower_stats["vf_explained_var"]
                learner_model.tower_stats["mean_entropy"] = tower_stats["mean_entropy"]
                learner_model.tower_stats["mean_kl_loss"] = tower_stats["mean_kl_loss"]
            elif self.config["algo_name"] in ["appo", "appo_serverless"]:
                learner_model.tower_stats["total_loss"] = tower_stats["total_loss"]
                learner_model.tower_stats["mean_policy_loss"] = tower_stats["mean_policy_loss"]
                learner_model.tower_stats["mean_vf_loss"] = tower_stats["mean_vf_loss"]
                learner_model.tower_stats["vf_explained_var"] = tower_stats["vf_explained_var"]
                learner_model.tower_stats["mean_entropy"] = tower_stats["mean_entropy"]
                learner_model.tower_stats["mean_kl_loss"] = tower_stats["mean_kl_loss"]
                learner_model.tower_stats["value_targets"] = tower_stats["value_targets"]
            elif self.config["algo_name"] in ["impala", "impala_serverless"]:
                learner_model.tower_stats["total_loss"] = tower_stats["total_loss"]
                learner_model.tower_stats["vf_explained_var"] = tower_stats["vf_explained_var"]
                learner_model.tower_stats["mean_entropy"] = tower_stats["mean_entropy"]
                learner_model.tower_stats["pi_loss"] = tower_stats["pi_loss"]
                learner_model.tower_stats["vf_loss"] = tower_stats["vf_loss"]
                learner_model.tower_stats["entropy"] = tower_stats["entropy"]

            tower_outputs.append(grads)
            ready_models_and_batches[learner_model] = num_grad_updates_per_batch
            learner_times.append(learner_time)
            
            grad_clock_list.append(grad_clock)
            grad_staleness_list.append(self.grad_clock - grad_clock)

        return tower_outputs, learner_idxs, learner_times, grad_clock_list, grad_staleness_list, ssp_clock_list, ssp_staleness_list, ready_models_and_batches

    def step(self):
        # Try to get gradients
        self.get_grads()

        if len(self.local_queue) > 0:
            ps_start_time = time.time()
            
            # Batch tower results
            batch_fetches = {}
            for i, _ in enumerate(self.policy.model_gpu_towers):
                batch_fetches[f"tower_{i}"] = {}

            #
            # Compute gradients across multi GPUs
            #

            tower_outputs = []

            # Serverless reduce
            if self.config["use_serverless"]:
                if self.local_queue:
                    grad = self.local_queue.pop(0)
                    if len(grad) == 2:
                        raise ValueError(grad[0])
                    
                    (_, async_info, _) = grad
                    (_, _, _, grad_clock) = async_info

                    self.to_process_queue.append(grad)
                    self.to_process_staleness_list.append(self.grad_clock - grad_clock)

                epoch_staleness_mean = np.mean(self.to_process_staleness_list)
                
                if epoch_staleness_mean <= self.config["serverless_bound"] or len(self.to_process_queue) >= self.config["num_async_learners"]:
                    tower_outputs, learner_idxs, learner_times, grad_clock_list, grad_staleness_list, ssp_clock_list, ssp_staleness_list, ready_models_and_batches = self.aggregate_grads(self.to_process_queue)

                    self.to_process_queue = []
                    self.to_process_staleness_list = []

            if tower_outputs:
                if self.config["use_serverless"]:
                    for output, staleness in zip(tower_outputs, grad_staleness_list):
                        all_grads = output[0]
                        
                        # Set main model's grads
                        for i, p in enumerate(self.model.parameters()):
                            p.grad = all_grads[i]
                        
                        if self.config["use_weighted_grads"] and staleness > 0:
                            self.policy.apply_gradients(
                                gradients=_directStepOptimizerSingleton,
                                staleness=staleness
                            )
                        else:
                            self.policy.apply_gradients(
                                gradients=_directStepOptimizerSingleton,
                                staleness=None
                            )
                        
                        self.policy.num_grad_updates += 1
                else:
                    all_grads = []

                    # Mean-reduce gradients over GPU-towers (do this on CPU: self.device).
                    for i in range(len(tower_outputs[0][0])):
                        if tower_outputs[0][0][i] is not None:
                            all_grads.append(
                                torch.mean(
                                    torch.stack([t[0][i].to(self.device) for t in tower_outputs]),
                                    dim=0,
                                )
                            )
                        else:
                            all_grads.append(None)
                
                    # Set main model's grads
                    for i, p in enumerate(self.model.parameters()):
                        p.grad = all_grads[i]

                    self.policy.apply_gradients(_directStepOptimizerSingleton)
                    self.policy.num_grad_updates += 1

                ps_end_time = time.time()

                batch_fetches["async_info"] = {}
                batch_fetches["async_info"]["epoch"] = self.epoch
                batch_fetches["async_info"]["learner_idxs"] = learner_idxs
                batch_fetches["async_info"]["learner_times"] = learner_times
                batch_fetches["async_info"]["ps_time"] = ps_end_time - ps_start_time
                batch_fetches["async_info"]["grad_ps_clock"] = self.grad_clock
                batch_fetches["async_info"]["grad_clock_list"] = grad_clock_list
                batch_fetches["async_info"]["grad_staleness_list"] = grad_staleness_list
                batch_fetches["async_info"]["config_serverless_bound"] = self.config["serverless_bound"]

                if self.config["use_serverless"]:
                    batch_fetches["async_info"]["epoch_staleness_mean"] = epoch_staleness_mean

                for i, model in enumerate(self.policy.model_gpu_towers):
                    if model in ready_models_and_batches:
                        num_grad_updates_per_batch = ready_models_and_batches[model]
                        batch_fetches[f"tower_{i}"].update(
                            {
                                LEARNER_STATS_KEY: self.policy.stats_fn(),
                                "model": {}
                                if self.config.get("_enable_new_api_stack", False)
                                else model.metrics(),
                                NUM_GRAD_UPDATES_LIFETIME: self.policy.num_grad_updates,
                                # -1, b/c we have to measure this diff before we do the update
                                # above.
                                DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY: (
                                    self.policy.num_grad_updates - 1 - (num_grad_updates_per_batch or 0)
                                ),
                            }
                        )
                    else:
                        batch_fetches[f"tower_{i}"].update(
                            {
                                LEARNER_STATS_KEY: self.policy.stats_fn(),
                                "model": {}
                                if self.config.get("_enable_new_api_stack", False)
                                else model.metrics(),
                                NUM_GRAD_UPDATES_LIFETIME: self.policy.num_grad_updates,
                                # -1, b/c we have to measure this diff before we do the update
                                # above.
                                DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY: (
                                    self.policy.num_grad_updates - 1 - 0
                                ),
                            }
                        )
                batch_fetches.update(self.policy.extra_compute_grad_fetches())

                # Update the result and latest model weights
                self.local_result_queue.append(batch_fetches)
                
                self.grad_clock = self.grad_clock + 1
                self.epoch = self.epoch + 1
                self.set_model()


@DeveloperAPI
class TorchPolicyV2(Policy):
    """PyTorch specific Policy class to use with RLlib."""

    @DeveloperAPI
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config: AlgorithmConfigDict,
        *,
        max_seq_len: int = 20,
    ):
        """Initializes a TorchPolicy instance.

        Args:
            observation_space: Observation space of the policy.
            action_space: Action space of the policy.
            config: The Policy's config dict.
            max_seq_len: Max sequence length for LSTM training.
        """
        self.framework = config["framework"] = "torch"

        self._loss_initialized = False
        super().__init__(observation_space, action_space, config)

        # Create model.
        if self.config.get("_enable_new_api_stack", False):
            model = self.make_rl_module()

            dist_class = None
        else:
            model, dist_class = self._init_model_and_dist_class()

        # Create multi-GPU model towers, if necessary.
        # - The central main model will be stored under self.model, residing
        #   on self.device (normally, a CPU).
        # - Each GPU will have a copy of that model under
        #   self.model_gpu_towers, matching the devices in self.devices.
        # - Parallelization is done by splitting the train batch and passing
        #   it through the model copies in parallel, then averaging over the
        #   resulting gradients, applying these averages on the main model and
        #   updating all towers' weights from the main model.
        # - In case of just one device (1 (fake or real) GPU or 1 CPU), no
        #   parallelization will be done.

        # Get devices to build the graph on.
        num_gpus = self._get_num_gpus_for_policy()
        gpu_ids = list(range(torch.cuda.device_count()))
        logger.info(f"Found {len(gpu_ids)} visible cuda devices.")

        # Place on one or more CPU(s) when either:
        # - Fake GPU mode.
        # - num_gpus=0 (either set by user or we are in local_mode=True).
        # - No GPUs available.
        if config["_fake_gpus"] or num_gpus == 0 or not gpu_ids:
            self.device = torch.device("cpu")
            self.devices = [self.device for _ in range(int(math.ceil(num_gpus)) or 1)]
            self.model_gpu_towers = [
                model if i == 0 else copy.deepcopy(model)
                for i in range(int(math.ceil(num_gpus)) or 1)
            ]
            if hasattr(self, "target_model"):
                self.target_models = {
                    m: self.target_model for m in self.model_gpu_towers
                }
            self.model = model
        # Place on one or more actual GPU(s), when:
        # - num_gpus > 0 (set by user) AND
        # - local_mode=False AND
        # - actual GPUs available AND
        # - non-fake GPU mode.
        else:
            # We are a remote worker (WORKER_MODE=1):
            # GPUs should be assigned to us by ray.
            if ray._private.worker._mode() == ray._private.worker.WORKER_MODE:
                gpu_ids = ray.get_gpu_ids()

            if len(gpu_ids) < num_gpus:
                raise ValueError(
                    "TorchPolicy was not able to find enough GPU IDs! Found "
                    f"{gpu_ids}, but num_gpus={num_gpus}."
                )

            self.devices = [
                torch.device("cuda:{}".format(i))
                for i, id_ in enumerate(gpu_ids)
                if i < num_gpus
            ]
            self.device = self.devices[0]
            ids = [id_ for i, id_ in enumerate(gpu_ids) if i < num_gpus]

            # Check if async learning
            self.model_gpu_towers = []
            if self.config["async_learn"]:
                self.redis_pool = redis.ConnectionPool(
                    host=self.config["redis_host"], 
                    port=self.config["redis_port"], 
                )
                self.redis_client = redis.Redis(connection_pool=self.redis_pool)
                
                self.model_gpu_towers_per_device = {}
                self.async_learners = {}

                self.manager = multiprocessing.Manager()
                self.async_shared_dict = self.manager.dict()
                self.async_shared_dict["ssp_table"] = self.manager.list()

                if hasattr(self, "target_model"):
                    self.target_models = {}

                num_learners_per_device = int(self.config["num_async_learners"] / len(self.devices))
                for i, _ in enumerate(ids):
                    device = self.devices[i]
                    self.model_gpu_towers_per_device[device] = []
                    for _ in range(num_learners_per_device):
                        model_copy = copy.deepcopy(model)
                        # self.model_gpu_towers.append((model_copy.to(device)))
                        self.model_gpu_towers.append(model_copy)
                        self.model_gpu_towers_per_device[device].append(model_copy)
                        self.async_learners[model_copy] = None
                        self.async_shared_dict["ssp_table"].append(0)

                        if hasattr(self, "target_model"):
                            self.target_models[model_copy] = copy.deepcopy(self.target_model).to(device)

                # Parameter server
                self.model = copy.deepcopy(model).to(self.device)
                self.async_learners[self.model] = None

                if hasattr(self, "target_model"):
                    self.target_models[self.model] = copy.deepcopy(self.target_model).to(self.device)
            else:
                for i, _ in enumerate(ids):
                    model_copy = copy.deepcopy(model)
                    self.model_gpu_towers.append(model_copy.to(self.devices[i]))
                if hasattr(self, "target_model"):
                    self.target_models = {
                        m: copy.deepcopy(self.target_model).to(self.devices[i])
                        for i, m in enumerate(self.model_gpu_towers)
                    }
                self.model = self.model_gpu_towers[0]

        self.dist_class = dist_class
        self.unwrapped_model = model  # used to support DistributedDataParallel

        # Lock used for locking some methods on the object-level.
        # This prevents possible race conditions when calling the model
        # first, then its value function (e.g. in a loss function), in
        # between of which another model call is made (e.g. to compute an
        # action).
        self._lock = threading.RLock()

        self._state_inputs = self.model.get_initial_state()
        self._is_recurrent = len(tree.flatten(self._state_inputs)) > 0
        if self.config.get("_enable_new_api_stack", False):
            # Maybe update view_requirements, e.g. for recurrent case.
            self.view_requirements = self.model.update_default_view_requirements(
                self.view_requirements
            )
        else:
            # Auto-update model's inference view requirements, if recurrent.
            self._update_model_view_requirements_from_init_state()
            # Combine view_requirements for Model and Policy.
            self.view_requirements.update(self.model.view_requirements)

        if self.config.get("_enable_new_api_stack", False):
            # We don't need an exploration object with RLModules
            self.exploration = None
        else:
            self.exploration = self._create_exploration()

        if not self.config.get("_enable_new_api_stack", False):
            self.optimizers = force_list(self.optimizer())

            def weight_grad(s):
                if s > 0:
                    weight_factor = 1 / (s ** (1/self.config["serverless_lr_smooth"]))
                    if weight_factor < self.config["serverless_decay"]:
                        weight_factor = 1
                else:
                    weight_factor = 1
                return weight_factor

            if self.config["use_weighted_grads"]:
                self.schedulers = []
                for opt in self.optimizers:
                    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, weight_grad)
                    self.schedulers.append(lr_scheduler)

            # Backward compatibility workaround so Policy will call self.loss()
            # directly.
            # TODO (jungong): clean up after all policies are migrated to new sub-class
            #  implementation.
            self._loss = None

            # Store, which params (by index within the model's list of
            # parameters) should be updated per optimizer.
            # Maps optimizer idx to set or param indices.
            self.multi_gpu_param_groups: List[Set[int]] = []
            main_params = {p: i for i, p in enumerate(self.model.parameters())}
            for o in self.optimizers:
                param_indices = []
                for pg_idx, pg in enumerate(o.param_groups):
                    for p in pg["params"]:
                        param_indices.append(main_params[p])
                self.multi_gpu_param_groups.append(set(param_indices))

            # Create n sample-batch buffers (num_multi_gpu_tower_stacks), each
            # one with m towers (num_gpus).
            num_buffers = self.config.get("num_multi_gpu_tower_stacks", 1)
            self._loaded_batches = [[] for _ in range(num_buffers)]

        # If set, means we are using distributed allreduce during learning.
        self.distributed_world_size = None

        self.batch_divisibility_req = self.get_batch_divisibility_req()
        self.max_seq_len = max_seq_len

        # If model is an RLModule it won't have tower_stats instead there will be a
        # self.tower_state[model] -> dict for each tower.
        self.tower_stats = {}
        if not hasattr(self.model, "tower_stats"):
            for model in self.model_gpu_towers:
                self.tower_stats[model] = {}
            if config["async_learn"]: # Add parameter learner
                self.tower_stats[self.model] = {}

    def loss_initialized(self):
        return self._loss_initialized

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    @override(Policy)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Constructs the loss function.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            Loss tensor given the input batch.
        """
        # Under the new _enable_new_api_stack the loss function still gets called in
        # order to initialize the view requirements of the sample batches that are
        # returned by
        # the sampler. In this case, we don't actually want to compute any loss, however
        # if we access the keys that are needed for a forward_train pass, then the
        # sampler will include those keys in the sample batches it returns. This means
        # that the correct sample batch keys will be available when using the learner
        # group API.
        if self.config._enable_new_api_stack:
            for k in model.input_specs_train():
                train_batch[k]
            return None
        else:
            raise NotImplementedError

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def action_sampler_fn(
        self,
        model: ModelV2,
        *,
        obs_batch: TensorType,
        state_batches: TensorType,
        **kwargs,
    ) -> Tuple[TensorType, TensorType, TensorType, List[TensorType]]:
        """Custom function for sampling new actions given policy.

        Args:
            model: Underlying model.
            obs_batch: Observation tensor batch.
            state_batches: Action sampling state batch.

        Returns:
            Sampled action
            Log-likelihood
            Action distribution inputs
            Updated state
        """
        return None, None, None, None

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def action_distribution_fn(
        self,
        model: ModelV2,
        *,
        obs_batch: TensorType,
        state_batches: TensorType,
        **kwargs,
    ) -> Tuple[TensorType, type, List[TensorType]]:
        """Action distribution function for this Policy.

        Args:
            model: Underlying model.
            obs_batch: Observation tensor batch.
            state_batches: Action sampling state batch.

        Returns:
            Distribution input.
            ActionDistribution class.
            State outs.
        """
        return None, None, None

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def make_model(self) -> ModelV2:
        """Create model.

        Note: only one of make_model or make_model_and_action_dist
        can be overridden.

        Returns:
            ModelV2 model.
        """
        return None

    @ExperimentalAPI
    @override(Policy)
    def maybe_remove_time_dimension(self, input_dict: Dict[str, TensorType]):
        assert self.config.get(
            "_enable_new_api_stack", False
        ), "This is a helper method for the new learner API."

        if self.config.get("_enable_new_api_stack", False) and self.model.is_stateful():
            # Note that this is a temporary workaround to fit the old sampling stack
            # to RL Modules.
            ret = {}

            def fold_mapping(item):
                item = torch.as_tensor(item)
                size = item.size()
                b_dim, t_dim = list(size[:2])
                other_dims = list(size[2:])
                return item.reshape([b_dim * t_dim] + other_dims)

            for k, v in input_dict.items():
                if k not in (STATE_IN, STATE_OUT):
                    ret[k] = tree.map_structure(fold_mapping, v)
                else:
                    # state in already has time dimension.
                    ret[k] = v

            return ret
        else:
            return input_dict

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def make_model_and_action_dist(
        self,
    ) -> Tuple[ModelV2, Type[TorchDistributionWrapper]]:
        """Create model and action distribution function.

        Returns:
            ModelV2 model.
            ActionDistribution class.
        """
        return None, None

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def get_batch_divisibility_req(self) -> int:
        """Get batch divisibility request.

        Returns:
            Size N. A sample batch must be of size K*N.
        """
        # By default, any sized batch is ok, so simply return 1.
        return 1

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        """Stats function. Returns a dict of statistics.

        Args:
            train_batch: The SampleBatch (already) used for training.

        Returns:
            The stats dict.
        """
        return {}

    @DeveloperAPI
    @OverrideToImplementCustomLogic_CallToSuperRecommended
    def extra_grad_process(
        self, optimizer: "torch.optim.Optimizer", loss: TensorType
    ) -> Dict[str, TensorType]:
        """Called after each optimizer.zero_grad() + loss.backward() call.

        Called for each self.optimizers/loss-value pair.
        Allows for gradient processing before optimizer.step() is called.
        E.g. for gradient clipping.

        Args:
            optimizer: A torch optimizer object.
            loss: The loss tensor associated with the optimizer.

        Returns:
            An dict with information on the gradient processing step.
        """
        return {}

    @DeveloperAPI
    @OverrideToImplementCustomLogic_CallToSuperRecommended
    def extra_compute_grad_fetches(self) -> Dict[str, Any]:
        """Extra values to fetch and return from compute_gradients().

        Returns:
            Extra fetch dict to be added to the fetch dict of the
            `compute_gradients` call.
        """
        return {LEARNER_STATS_KEY: {}}  # e.g, stats, td error, etc.

    @DeveloperAPI
    @OverrideToImplementCustomLogic_CallToSuperRecommended
    def extra_action_out(
        self,
        input_dict: Dict[str, TensorType],
        state_batches: List[TensorType],
        model: TorchModelV2,
        action_dist: TorchDistributionWrapper,
    ) -> Dict[str, TensorType]:
        """Returns dict of extra info to include in experience batch.

        Args:
            input_dict: Dict of model input tensors.
            state_batches: List of state tensors.
            model: Reference to the model object.
            action_dist: Torch action dist object
                to get log-probs (e.g. for already sampled actions).

        Returns:
            Extra outputs to return in a `compute_actions_from_input_dict()`
            call (3rd return value).
        """
        return {}

    @override(Policy)
    @DeveloperAPI
    @OverrideToImplementCustomLogic_CallToSuperRecommended
    def postprocess_trajectory(
        self,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[Dict[Any, SampleBatch]] = None,
        episode: Optional["Episode"] = None,
    ) -> SampleBatch:
        """Postprocesses a trajectory and returns the processed trajectory.

        The trajectory contains only data from one episode and from one agent.
        - If  `config.batch_mode=truncate_episodes` (default), sample_batch may
        contain a truncated (at-the-end) episode, in case the
        `config.rollout_fragment_length` was reached by the sampler.
        - If `config.batch_mode=complete_episodes`, sample_batch will contain
        exactly one episode (no matter how long).
        New columns can be added to sample_batch and existing ones may be altered.

        Args:
            sample_batch: The SampleBatch to postprocess.
            other_agent_batches (Optional[Dict[PolicyID, SampleBatch]]): Optional
                dict of AgentIDs mapping to other agents' trajectory data (from the
                same episode). NOTE: The other agents use the same policy.
            episode (Optional[Episode]): Optional multi-agent episode
                object in which the agents operated.

        Returns:
            SampleBatch: The postprocessed, modified SampleBatch (or a new one).
        """
        return sample_batch

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def optimizer(
        self,
    ) -> Union[List["torch.optim.Optimizer"], "torch.optim.Optimizer"]:
        """Custom the local PyTorch optimizer(s) to use.

        Returns:
            The local PyTorch optimizer(s) to use for this Policy.
        """
        if hasattr(self, "config"):
            optimizers = [
                torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
            ]
        else:
            optimizers = [torch.optim.Adam(self.model.parameters())]
        if self.exploration:
            optimizers = self.exploration.get_exploration_optimizer(optimizers)
        return optimizers

    def _init_model_and_dist_class(self):
        if is_overridden(self.make_model) and is_overridden(
            self.make_model_and_action_dist
        ):
            raise ValueError(
                "Only one of make_model or make_model_and_action_dist "
                "can be overridden."
            )

        if is_overridden(self.make_model):
            model = self.make_model()
            dist_class, _ = ModelCatalog.get_action_dist(
                self.action_space, self.config["model"], framework=self.framework
            )
        elif is_overridden(self.make_model_and_action_dist):
            model, dist_class = self.make_model_and_action_dist()
        else:
            dist_class, logit_dim = ModelCatalog.get_action_dist(
                self.action_space, self.config["model"], framework=self.framework
            )
            model = ModelCatalog.get_model_v2(
                obs_space=self.observation_space,
                action_space=self.action_space,
                num_outputs=logit_dim,
                model_config=self.config["model"],
                framework=self.framework,
            )

        return model, dist_class

    @override(Policy)
    def compute_actions_from_input_dict(
        self,
        input_dict: Dict[str, TensorType],
        explore: bool = None,
        timestep: Optional[int] = None,
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:

        seq_lens = None
        with torch.no_grad():
            # Pass lazy (torch) tensor dict to Model as `input_dict`.
            input_dict = self._lazy_tensor_dict(input_dict)
            input_dict.set_training(True)
            if self.config.get("_enable_new_api_stack", False):
                return self._compute_action_helper(
                    input_dict,
                    state_batches=None,
                    seq_lens=None,
                    explore=explore,
                    timestep=timestep,
                )
            else:
                # Pack internal state inputs into (separate) list.
                state_batches = [
                    input_dict[k] for k in input_dict.keys() if "state_in" in k[:8]
                ]
                # Calculate RNN sequence lengths.
                if state_batches:
                    seq_lens = torch.tensor(
                        [1] * len(state_batches[0]),
                        dtype=torch.long,
                        device=state_batches[0].device,
                    )

                return self._compute_action_helper(
                    input_dict, state_batches, seq_lens, explore, timestep
                )

    @override(Policy)
    @DeveloperAPI
    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
        info_batch: Optional[Dict[str, list]] = None,
        episodes: Optional[List["Episode"]] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        **kwargs,
    ) -> Tuple[TensorStructType, List[TensorType], Dict[str, TensorType]]:

        with torch.no_grad():
            seq_lens = torch.ones(len(obs_batch), dtype=torch.int32)
            input_dict = self._lazy_tensor_dict(
                {
                    SampleBatch.CUR_OBS: obs_batch,
                    "is_training": False,
                }
            )
            if prev_action_batch is not None:
                input_dict[SampleBatch.PREV_ACTIONS] = np.asarray(prev_action_batch)
            if prev_reward_batch is not None:
                input_dict[SampleBatch.PREV_REWARDS] = np.asarray(prev_reward_batch)
            state_batches = [
                convert_to_torch_tensor(s, self.device) for s in (state_batches or [])
            ]
            return self._compute_action_helper(
                input_dict, state_batches, seq_lens, explore, timestep
            )

    @with_lock
    @override(Policy)
    @DeveloperAPI
    def compute_log_likelihoods(
        self,
        actions: Union[List[TensorStructType], TensorStructType],
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Optional[
            Union[List[TensorStructType], TensorStructType]
        ] = None,
        prev_reward_batch: Optional[
            Union[List[TensorStructType], TensorStructType]
        ] = None,
        actions_normalized: bool = True,
        in_training: bool = True,
    ) -> TensorType:

        if is_overridden(self.action_sampler_fn) and not is_overridden(
            self.action_distribution_fn
        ):
            raise ValueError(
                "Cannot compute log-prob/likelihood w/o an "
                "`action_distribution_fn` and a provided "
                "`action_sampler_fn`!"
            )

        with torch.no_grad():
            input_dict = self._lazy_tensor_dict(
                {SampleBatch.CUR_OBS: obs_batch, SampleBatch.ACTIONS: actions}
            )
            if prev_action_batch is not None:
                input_dict[SampleBatch.PREV_ACTIONS] = prev_action_batch
            if prev_reward_batch is not None:
                input_dict[SampleBatch.PREV_REWARDS] = prev_reward_batch
            seq_lens = torch.ones(len(obs_batch), dtype=torch.int32)
            state_batches = [
                convert_to_torch_tensor(s, self.device) for s in (state_batches or [])
            ]

            if self.exploration:
                # Exploration hook before each forward pass.
                self.exploration.before_compute_actions(explore=False)

            # Action dist class and inputs are generated via custom function.
            if is_overridden(self.action_distribution_fn):
                dist_inputs, dist_class, state_out = self.action_distribution_fn(
                    self.model,
                    obs_batch=input_dict,
                    state_batches=state_batches,
                    seq_lens=seq_lens,
                    explore=False,
                    is_training=False,
                )
                action_dist = dist_class(dist_inputs, self.model)
            # Default action-dist inputs calculation.
            else:
                if self.config.get("_enable_new_api_stack", False):
                    if in_training:
                        output = self.model.forward_train(input_dict)
                        action_dist_cls = self.model.get_train_action_dist_cls()
                        if action_dist_cls is None:
                            raise ValueError(
                                "The RLModules must provide an appropriate action "
                                "distribution class for training if is_eval_mode is "
                                "False."
                            )
                    else:
                        output = self.model.forward_exploration(input_dict)
                        action_dist_cls = self.model.get_exploration_action_dist_cls()
                        if action_dist_cls is None:
                            raise ValueError(
                                "The RLModules must provide an appropriate action "
                                "distribution class for exploration if is_eval_mode is "
                                "True."
                            )

                    action_dist_inputs = output.get(
                        SampleBatch.ACTION_DIST_INPUTS, None
                    )
                    if action_dist_inputs is None:
                        raise ValueError(
                            "The RLModules must provide inputs to create the action "
                            "distribution. These should be part of the output of the "
                            "appropriate forward method under the key "
                            "SampleBatch.ACTION_DIST_INPUTS."
                        )

                    action_dist = action_dist_cls.from_logits(action_dist_inputs)
                else:
                    dist_class = self.dist_class
                    dist_inputs, _ = self.model(input_dict, state_batches, seq_lens)

                    action_dist = dist_class(dist_inputs, self.model)

            # Normalize actions if necessary.
            actions = input_dict[SampleBatch.ACTIONS]
            if not actions_normalized and self.config["normalize_actions"]:
                actions = normalize_action(actions, self.action_space_struct)

            log_likelihoods = action_dist.logp(actions)

            return log_likelihoods

    @with_lock
    @override(Policy)
    @DeveloperAPI
    def learn_on_batch(self, postprocessed_batch: SampleBatch) -> Dict[str, TensorType]:
        # Set Model to train mode.
        if self.model:
            self.model.train()
        # Callback handling.
        learn_stats = {}
        self.callbacks.on_learn_on_batch(
            policy=self, train_batch=postprocessed_batch, result=learn_stats
        )

        # Compute gradients (will calculate all losses and `backward()`
        # them to get the grads).
        grads, fetches = self.compute_gradients(postprocessed_batch)

        # Step the optimizers.
        self.apply_gradients(_directStepOptimizerSingleton)

        self.num_grad_updates += 1
        if self.model and hasattr(self.model, "metrics"):
            fetches["model"] = self.model.metrics()
        else:
            fetches["model"] = {}

        fetches.update(
            {
                "custom_metrics": learn_stats,
                NUM_AGENT_STEPS_TRAINED: postprocessed_batch.count,
                NUM_GRAD_UPDATES_LIFETIME: self.num_grad_updates,
                # -1, b/c we have to measure this diff before we do the update above.
                DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY: (
                    self.num_grad_updates
                    - 1
                    - (postprocessed_batch.num_grad_updates or 0)
                ),
            }
        )

        return fetches

    @override(Policy)
    @DeveloperAPI
    def load_batch_into_buffer(
        self,
        batch: SampleBatch,
        buffer_index: int = 0,
    ) -> int:
        # Set the is_training flag of the batch.
        batch.set_training(True)

        # Shortcut for 1 CPU only: Store batch in `self._loaded_batches`.
        if len(self.devices) == 1 and self.devices[0].type == "cpu":
            assert buffer_index == 0
            pad_batch_to_sequences_of_same_size(
                batch=batch,
                max_seq_len=self.max_seq_len,
                shuffle=False,
                batch_divisibility_req=self.batch_divisibility_req,
                view_requirements=self.view_requirements,
                _enable_new_api_stack=self.config.get("_enable_new_api_stack", False),
                padding="last"
                if self.config.get("_enable_new_api_stack", False)
                else "zero",
            )
            self._lazy_tensor_dict(batch)
            self._loaded_batches[0] = [batch]
            return len(batch)

        # Batch (len=28, seq-lens=[4, 7, 4, 10, 3]):
        # 0123 0123456 0123 0123456789ABC

        # 1) split into n per-GPU sub batches (n=2).
        # [0123 0123456] [012] [3 0123456789 ABC]
        # (len=14, 14 seq-lens=[4, 7, 3] [1, 10, 3])
        if self.config["async_learn"]:
            slices = batch.timeslices(num_slices=len(self.model_gpu_towers))
        else:
            slices = batch.timeslices(num_slices=len(self.devices))

        # 2) zero-padding (max-seq-len=10).
        # - [0123000000 0123456000 0120000000]
        # - [3000000000 0123456789 ABC0000000]
        for slice in slices:
            pad_batch_to_sequences_of_same_size(
                batch=slice,
                max_seq_len=self.max_seq_len,
                shuffle=False,
                batch_divisibility_req=self.batch_divisibility_req,
                view_requirements=self.view_requirements,
                _enable_new_api_stack=self.config.get("_enable_new_api_stack", False),
                padding="last"
                if self.config.get("_enable_new_api_stack", False)
                else "zero",
            )

        # 3) Load splits into the given buffer (consisting of n GPUs).
        if self.config["async_learn"]:
            num_slices_per_device = int(len(slices) / len(self.devices))
            slices_per_device = {}
            for device in self.devices:
                slices_per_device[device] = []
                for i in range(num_slices_per_device):
                    slice = slices.pop(0)
                    slices_per_device[device].append(slice.to_device(device))
            self._loaded_batches[buffer_index] = slices_per_device

            # # Get the correct slice of the already loaded batch to use,
            # # based on offset and batch size
            # learner_batch_size = self.config.get(
            #     "sgd_minibatch_size", self.config["train_batch_size"]
            # ) // len(self.model_gpu_towers_per_device[self.devices[0]])

            # # Inqueue the batch pointers for async learn
            # samples_per_device = len(slices_per_device[self.devices[0]])
            # num_batches = max(1, samples_per_device // learner_batch_size)
            # for _ in range(self.config["num_sgd_iter"]):
            #     permutation = np.random.permutation(num_batches)
            #     for device in self.devices:
            #         for batch_index in range(num_batches):
            #             offset = permutation[batch_index] * learner_batch_size
            #             if learner_batch_size >= len(self._loaded_batches[buffer_index][device]):
            #                 for b in self._loaded_batches[buffer_index][device]:
            #                     self.async_sample_queue[device].put(b)
            #             else:
            #                 for b in self._loaded_batches[buffer_index][device][offset : offset + learner_batch_size]:
            #                     self.async_sample_queue[device].put(b)
            
            return len(slices_per_device[self.devices[0]])
        else:
            slices = [slice.to_device(self.devices[i]) for i, slice in enumerate(slices)]
            self._loaded_batches[buffer_index] = slices

            # Return loaded samples per-device.
            return len(slices[0])

    @override(Policy)
    @DeveloperAPI
    def get_num_samples_loaded_into_buffer(self, buffer_index: int = 0) -> int:
        if len(self.devices) == 1 and self.devices[0] == "/cpu:0":
            assert buffer_index == 0
        return sum(len(b) for b in self._loaded_batches[buffer_index])

    @override(Policy)
    @DeveloperAPI
    def learn_on_loaded_batch(
        self, 
        offset: int = 0, 
        buffer_index: int = 0,
    ):
        if not self._loaded_batches[buffer_index]:
            raise ValueError(
                "Must call Policy.load_batch_into_buffer() before "
                "Policy.learn_on_loaded_batch()!"
            )

        # Async learn
        if self.config["async_learn"]:
            exclude_time_start = time.time()

            self.start_async_learners()
            self.start_async_aggregator()

            # Sync model weights at beginning
            self.sync_learner_weights()

            # Load batches into sample queue
            # Get the correct slice of the already loaded batch to use,
            # based on offset and batch size
            slices_per_device = self._loaded_batches[buffer_index]
            learner_batch_size = self.config.get(
                "sgd_minibatch_size", self.config["train_batch_size"]
            ) // len(self.model_gpu_towers_per_device[self.devices[0]])

            # Inqueue the batch pointers for async learn
            samples_per_device = len(slices_per_device[self.devices[0]])
            num_batches = max(1, samples_per_device // learner_batch_size)
            permutation = np.random.permutation(num_batches)
            
            for device_idx, device in enumerate(self.devices):
                for batch_index in range(num_batches):
                    offset = permutation[batch_index] * learner_batch_size
                    if learner_batch_size >= len(self._loaded_batches[buffer_index][device]):
                        data_list = [pickle.dumps(b) for b in self._loaded_batches[buffer_index][device]]
                        self.redis_client.lpush("sample_queue_{}".format(device_idx), *data_list)
                    else:
                        data_list = [pickle.dumps(b) for b in self._loaded_batches[buffer_index][device][offset : offset + learner_batch_size]]
                        self.redis_client.lpush("sample_queue_{}".format(device_idx), *data_list)
            
            exclude_time_end = time.time()

            def all_queue_empty():
                for device_idx, _ in enumerate(self.devices):
                    if self.redis_client.llen("sample_queue_{}") > 0:
                        return False
                if self.redis_client.llen("grad_queue") > 0:
                    return False
                return True

            # Wait for all queues to be empty
            while not all_queue_empty():
                pass

            results = self.async_learners[self.model].local_result_queue
            self.async_learners[self.model].local_result_queue = []

            if results:
                results[0]["async_info"]["exclude_time"] = exclude_time_end - exclude_time_start

            return results
        else: 
            # Get the correct slice of the already loaded batch to use,
            # based on offset and batch size.
            device_batch_size = self.config.get(
                "sgd_minibatch_size", self.config["train_batch_size"]
            ) // len(self.devices)

            # Set Model to train mode.
            if self.model_gpu_towers:
                for t in self.model_gpu_towers:
                    t.train()

            # Shortcut for 1 CPU only: Batch should already be stored in
            # `self._loaded_batches`.
            if len(self.devices) == 1 and self.devices[0].type == "cpu":
                assert buffer_index == 0
                if device_batch_size >= len(self._loaded_batches[0][0]):
                    batch = self._loaded_batches[0][0]
                else:
                    batch = self._loaded_batches[0][0][offset : offset + device_batch_size]

                return self.learn_on_batch(batch)

            if len(self.devices) > 1:
                # Copy weights of main model (tower-0) to all other towers.
                state_dict = self.model.state_dict()
                # Just making sure tower-0 is really the same as self.model.
                assert self.model_gpu_towers[0] is self.model
                for tower in self.model_gpu_towers[1:]:
                    tower.load_state_dict(state_dict)

            if device_batch_size >= sum(len(s) for s in self._loaded_batches[buffer_index]):
                device_batches = self._loaded_batches[buffer_index]
            else:
                device_batches = [
                    b[offset : offset + device_batch_size]
                    for b in self._loaded_batches[buffer_index]
                ]

            # Callback handling.
            batch_fetches = {}
            for i, batch in enumerate(device_batches):
                custom_metrics = {}
                self.callbacks.on_learn_on_batch(
                    policy=self, train_batch=batch, result=custom_metrics
                )
                batch_fetches[f"tower_{i}"] = {"custom_metrics": custom_metrics}

            # Do the (maybe parallelized) gradient calculation step.
            tower_outputs = self._multi_gpu_parallel_grad_calc(device_batches)

            # Mean-reduce gradients over GPU-towers (do this on CPU: self.device).
            all_grads = []
            for i in range(len(tower_outputs[0][0])):
                if tower_outputs[0][0][i] is not None:
                    all_grads.append(
                        torch.mean(
                            torch.stack([t[0][i].to(self.device) for t in tower_outputs]),
                            dim=0,
                        )
                    )
                else:
                    all_grads.append(None)
            # Set main model's grads to mean-reduced values.
            for i, p in enumerate(self.model.parameters()):
                p.grad = all_grads[i]

            self.apply_gradients(_directStepOptimizerSingleton)

            self.num_grad_updates += 1

            for i, (model, batch) in enumerate(zip(self.model_gpu_towers, device_batches)):
                batch_fetches[f"tower_{i}"].update(
                    {
                        LEARNER_STATS_KEY: self.stats_fn(batch),
                        "model": {}
                        if self.config.get("_enable_new_api_stack", False)
                        else model.metrics(),
                        NUM_GRAD_UPDATES_LIFETIME: self.num_grad_updates,
                        # -1, b/c we have to measure this diff before we do the update
                        # above.
                        DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY: (
                            self.num_grad_updates - 1 - (batch.num_grad_updates or 0)
                        ),
                    }
                )
            batch_fetches.update(self.extra_compute_grad_fetches())

        return batch_fetches

    def sync_learner_weights(self):
        for t in self.model_gpu_towers:
            learner = self.async_learners[t]
            assert learner and learner.is_alive()
            learner.set_weights(self.get_weights())
            learner.grad_clock = self.async_learners[self.model].grad_clock

    def reset_async_epoch(self):
        if self.async_learners[self.model]:
            self.async_learners[self.model].epoch = 0

    def start_async_learners(self):
        # Init async learners if not started yet
        for device_idx, device in enumerate(self.devices):
            for model in self.model_gpu_towers_per_device[device]:
                if not self.async_learners[model]:
                    self.async_learners[model] = AsyncLearner(
                        learner_idx=self.model_gpu_towers.index(model),
                        init_state_dict=self.model.state_dict(),
                        config=self.config,
                        distributed_world_size=self.distributed_world_size,
                        observation_space=self.observation_space,
                        action_space=self.action_space,
                        framework=self.framework,
                        is_recurrent=self.is_recurrent(),
                        device_idx=device_idx,
                        device=device,
                        shared_dict=self.async_shared_dict,
                    )
                    self.async_learners[model].start()
        
        # Set models to train mode
        for t in self.model_gpu_towers:
            assert self.async_learners[t] and self.async_learners[t].is_alive()
            # t.train()

    def start_async_aggregator(self):
        if not self.async_learners[self.model]:
            self.async_learners[self.model] = AsyncAggregator(
                policy=self,
                model=self.model,
                device=self.device,
                shared_dict=self.async_shared_dict,
            )
            self.async_learners[self.model].start()

        assert self.async_learners[self.model] and self.async_learners[self.model].is_alive()
        self.model.train()

    def end_async_learners(self):
        for device in self.model_gpu_towers_per_device.keys():
            for model in self.model_gpu_towers_per_device[device]:
                if not self.async_learners[model]:
                    self.async_learners[model].stop = True

        for t in self.model_gpu_towers:
            assert not self.async_learners[t].is_alive()
            
    def end_async_ps(self):
        self.async_learners[self.model].stop = True
        assert not self.async_learners[self.model].is_alive()

    @with_lock
    @override(Policy)
    @DeveloperAPI
    def compute_gradients(self, postprocessed_batch: SampleBatch) -> ModelGradients:

        assert len(self.devices) == 1

        # If not done yet, see whether we have to zero-pad this batch.
        if not postprocessed_batch.zero_padded:
            pad_batch_to_sequences_of_same_size(
                batch=postprocessed_batch,
                max_seq_len=self.max_seq_len,
                shuffle=False,
                batch_divisibility_req=self.batch_divisibility_req,
                view_requirements=self.view_requirements,
                _enable_new_api_stack=self.config.get("_enable_new_api_stack", False),
                padding="last"
                if self.config.get("_enable_new_api_stack", False)
                else "zero",
            )

        postprocessed_batch.set_training(True)
        self._lazy_tensor_dict(postprocessed_batch, device=self.devices[0])

        # Do the (maybe parallelized) gradient calculation step.
        tower_outputs = self._multi_gpu_parallel_grad_calc([postprocessed_batch])

        all_grads, grad_info = tower_outputs[0]

        grad_info["allreduce_latency"] /= len(self.optimizers)
        grad_info.update(self.stats_fn(postprocessed_batch))

        fetches = self.extra_compute_grad_fetches()

        return all_grads, dict(fetches, **{LEARNER_STATS_KEY: grad_info})

    @override(Policy)
    @DeveloperAPI
    def apply_gradients(
        self, 
        gradients: ModelGradients, 
        staleness = None, 
    ) -> None:
        if gradients == _directStepOptimizerSingleton:
            if self.config["use_weighted_grads"] and staleness:
                for opt, scheduler in zip(self.optimizers, self.schedulers):
                    opt.step()
                    scheduler.step()
            else:
                for i, opt in enumerate(self.optimizers):
                    opt.step()
        else:
            # TODO(sven): Not supported for multiple optimizers yet.
            assert len(self.optimizers) == 1
            for g, p in zip(gradients, self.model.parameters()):
                if g is not None:
                    if torch.is_tensor(g):
                        p.grad = g.to(self.device)
                    else:
                        p.grad = torch.from_numpy(g).to(self.device)

            self.optimizers[0].step()

    @DeveloperAPI
    def get_tower_stats(self, stats_name: str) -> List[TensorStructType]:
        """Returns list of per-tower stats, copied to this Policy's device.

        Args:
            stats_name: The name of the stats to average over (this str
                must exist as a key inside each tower's `tower_stats` dict).

        Returns:
            The list of stats tensor (structs) of all towers, copied to this
            Policy's device.

        Raises:
            AssertionError: If the `stats_name` cannot be found in any one
            of the tower's `tower_stats` dicts.
        """
        data = []
        for model in self.model_gpu_towers:
            if self.tower_stats:
                tower_stats = self.tower_stats[model]
            else:
                tower_stats = model.tower_stats

            if stats_name in tower_stats:
                data.append(
                    tree.map_structure(
                        lambda s: s.to(self.device), tower_stats[stats_name]
                    )
                )
        
        if self.config["async_learn"]:
            if self.tower_stats:
                tower_stats = self.tower_stats[self.model]
            else:
                tower_stats = self.model.tower_stats
            
            if stats_name in tower_stats:
                data.append(
                    tree.map_structure(
                        lambda s: s.to(self.device), tower_stats[stats_name]
                    )
                )

        assert len(data) > 0, (
            f"Stats `{stats_name}` not found in any of the towers (you have "
            f"{len(self.model_gpu_towers)} towers in total)! Make "
            "sure you call the loss function on at least one of the towers."
        )
        return data

    @override(Policy)
    @DeveloperAPI
    def get_weights(self) -> ModelWeights:
        return {k: v.cpu().detach().numpy() for k, v in self.model.state_dict().items()}

    @override(Policy)
    @DeveloperAPI
    def set_weights(self, weights: ModelWeights) -> None:
        weights = convert_to_torch_tensor(weights, device=self.device)
        if self.config.get("_enable_new_api_stack", False):
            self.model.set_state(weights)
        else:
            self.model.load_state_dict(weights)

    @override(Policy)
    @DeveloperAPI
    def is_recurrent(self) -> bool:
        return self._is_recurrent

    @override(Policy)
    @DeveloperAPI
    def num_state_tensors(self) -> int:
        return len(self.model.get_initial_state())

    @override(Policy)
    @DeveloperAPI
    def get_initial_state(self) -> List[TensorType]:
        if self.config.get("_enable_new_api_stack", False):
            # convert the tree of tensors to a tree to numpy arrays
            return tree.map_structure(
                lambda s: convert_to_numpy(s), self.model.get_initial_state()
            )

        return [s.detach().cpu().numpy() for s in self.model.get_initial_state()]

    @override(Policy)
    @DeveloperAPI
    @OverrideToImplementCustomLogic_CallToSuperRecommended
    def get_state(self) -> PolicyState:
        # Legacy Policy state (w/o torch.nn.Module and w/o PolicySpec).
        state = super().get_state()

        state["_optimizer_variables"] = []
        # In the new Learner API stack, the optimizers live in the learner.
        if not self.config.get("_enable_new_api_stack", False):
            for i, o in enumerate(self.optimizers):
                optim_state_dict = convert_to_numpy(o.state_dict())
                state["_optimizer_variables"].append(optim_state_dict)
        # Add exploration state.
        if not self.config.get("_enable_new_api_stack", False) and self.exploration:
            # This is not compatible with RLModules, which have a method
            # `forward_exploration` to specify custom exploration behavior.
            state["_exploration_state"] = self.exploration.get_state()
        return state

    @override(Policy)
    @DeveloperAPI
    @OverrideToImplementCustomLogic_CallToSuperRecommended
    def set_state(self, state: PolicyState) -> None:
        # Set optimizer vars first.
        optimizer_vars = state.get("_optimizer_variables", None)
        if optimizer_vars:
            assert len(optimizer_vars) == len(self.optimizers)
            for o, s in zip(self.optimizers, optimizer_vars):
                # Torch optimizer param_groups include things like beta, etc. These
                # parameters should be left as scalar and not converted to tensors.
                # otherwise, torch.optim.step() will start to complain.
                optim_state_dict = {"param_groups": s["param_groups"]}
                optim_state_dict["state"] = convert_to_torch_tensor(
                    s["state"], device=self.device
                )
                o.load_state_dict(optim_state_dict)
        # Set exploration's state.
        if hasattr(self, "exploration") and "_exploration_state" in state:
            self.exploration.set_state(state=state["_exploration_state"])

        # Restore glbal timestep.
        self.global_timestep = state["global_timestep"]

        # Then the Policy's (NN) weights and connectors.
        super().set_state(state)

    @override(Policy)
    @DeveloperAPI
    def export_model(self, export_dir: str, onnx: Optional[int] = None) -> None:
        """Exports the Policy's Model to local directory for serving.

        Creates a TorchScript model and saves it.

        Args:
            export_dir: Local writable directory or filename.
            onnx: If given, will export model in ONNX format. The
                value of this parameter set the ONNX OpSet version to use.
        """

        os.makedirs(export_dir, exist_ok=True)

        enable_rl_module = self.config.get("_enable_new_api_stack", False)
        if enable_rl_module and onnx:
            raise ValueError("ONNX export not supported for RLModule API.")

        if onnx:
            self._lazy_tensor_dict(self._dummy_batch)
            # Provide dummy state inputs if not an RNN (torch cannot jit with
            # returned empty internal states list).
            if "state_in_0" not in self._dummy_batch:
                self._dummy_batch["state_in_0"] = self._dummy_batch[
                    SampleBatch.SEQ_LENS
                ] = np.array([1.0])
            seq_lens = self._dummy_batch[SampleBatch.SEQ_LENS]

            state_ins = []
            i = 0
            while "state_in_{}".format(i) in self._dummy_batch:
                state_ins.append(self._dummy_batch["state_in_{}".format(i)])
                i += 1
            dummy_inputs = {
                k: self._dummy_batch[k]
                for k in self._dummy_batch.keys()
                if k != "is_training"
            }

            file_name = os.path.join(export_dir, "model.onnx")
            torch.onnx.export(
                self.model,
                (dummy_inputs, state_ins, seq_lens),
                file_name,
                export_params=True,
                opset_version=onnx,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys())
                + ["state_ins", SampleBatch.SEQ_LENS],
                output_names=["output", "state_outs"],
                dynamic_axes={
                    k: {0: "batch_size"}
                    for k in list(dummy_inputs.keys())
                    + ["state_ins", SampleBatch.SEQ_LENS]
                },
            )
        # Save the torch.Model (architecture and weights, so it can be retrieved
        # w/o access to the original (custom) Model or Policy code).
        else:
            filename = os.path.join(export_dir, "model.pt")
            try:
                torch.save(self.model, f=filename)
            except Exception:
                if os.path.exists(filename):
                    os.remove(filename)
                logger.warning(ERR_MSG_TORCH_POLICY_CANNOT_SAVE_MODEL)

    @override(Policy)
    @DeveloperAPI
    def import_model_from_h5(self, import_file: str) -> None:
        """Imports weights into torch model."""
        return self.model.import_from_h5(import_file)

    @with_lock
    def _compute_action_helper(
        self, input_dict, state_batches, seq_lens, explore, timestep
    ):
        """Shared forward pass logic (w/ and w/o trajectory view API).

        Returns:
            A tuple consisting of a) actions, b) state_out, c) extra_fetches.
            The input_dict is modified in-place to include a numpy copy of the computed
            actions under `SampleBatch.ACTIONS`.
        """
        explore = explore if explore is not None else self.config["explore"]
        timestep = timestep if timestep is not None else self.global_timestep

        # Switch to eval mode.
        if self.model:
            self.model.eval()

        extra_fetches = dist_inputs = logp = None

        # New API stack: `self.model` is-a RLModule.
        if isinstance(self.model, RLModule):
            if self.model.is_stateful():
                # For recurrent models, we need to add a time dimension.
                if not seq_lens:
                    # In order to calculate the batch size ad hoc, we need a sample
                    # batch.
                    if not isinstance(input_dict, SampleBatch):
                        input_dict = SampleBatch(input_dict)
                    seq_lens = np.array([1] * len(input_dict))
                input_dict = self.maybe_add_time_dimension(
                    input_dict, seq_lens=seq_lens
                )
            input_dict = convert_to_torch_tensor(input_dict, device=self.device)

            # Batches going into the RL Module should not have seq_lens.
            if SampleBatch.SEQ_LENS in input_dict:
                del input_dict[SampleBatch.SEQ_LENS]

            if explore:
                fwd_out = self.model.forward_exploration(input_dict)
                # For recurrent models, we need to remove the time dimension.
                fwd_out = self.maybe_remove_time_dimension(fwd_out)

                # ACTION_DIST_INPUTS field returned by `forward_exploration()` ->
                # Create a distribution object.
                action_dist = None
                # Maybe the RLModule has already computed actions.
                if SampleBatch.ACTION_DIST_INPUTS in fwd_out:
                    dist_inputs = fwd_out[SampleBatch.ACTION_DIST_INPUTS]
                    action_dist_class = self.model.get_exploration_action_dist_cls()
                    action_dist = action_dist_class.from_logits(dist_inputs)

                # If `forward_exploration()` returned actions, use them here as-is.
                if SampleBatch.ACTIONS in fwd_out:
                    actions = fwd_out[SampleBatch.ACTIONS]
                # Otherwise, sample actions from the distribution.
                else:
                    if action_dist is None:
                        raise KeyError(
                            "Your RLModule's `forward_exploration()` method must return"
                            f" a dict with either the {SampleBatch.ACTIONS} key or the "
                            f"{SampleBatch.ACTION_DIST_INPUTS} key in it (or both)!"
                        )
                    actions = action_dist.sample()

                # Compute action-logp and action-prob from distribution and add to
                # `extra_fetches`, if possible.
                if action_dist is not None:
                    logp = action_dist.logp(actions)
            else:
                fwd_out = self.model.forward_inference(input_dict)
                # For recurrent models, we need to remove the time dimension.
                fwd_out = self.maybe_remove_time_dimension(fwd_out)

                # ACTION_DIST_INPUTS field returned by `forward_exploration()` ->
                # Create a distribution object.
                action_dist = None
                if SampleBatch.ACTION_DIST_INPUTS in fwd_out:
                    dist_inputs = fwd_out[SampleBatch.ACTION_DIST_INPUTS]
                    action_dist_class = self.model.get_inference_action_dist_cls()
                    action_dist = action_dist_class.from_logits(dist_inputs)
                    action_dist = action_dist.to_deterministic()

                # If `forward_inference()` returned actions, use them here as-is.
                if SampleBatch.ACTIONS in fwd_out:
                    actions = fwd_out[SampleBatch.ACTIONS]
                # Otherwise, sample actions from the distribution.
                else:
                    if action_dist is None:
                        raise KeyError(
                            "Your RLModule's `forward_inference()` method must return"
                            f" a dict with either the {SampleBatch.ACTIONS} key or the "
                            f"{SampleBatch.ACTION_DIST_INPUTS} key in it (or both)!"
                        )
                    actions = action_dist.sample()

            # Anything but actions and state_out is an extra fetch.
            state_out = fwd_out.pop(STATE_OUT, {})
            extra_fetches = fwd_out

        elif is_overridden(self.action_sampler_fn):
            action_dist = None
            actions, logp, dist_inputs, state_out = self.action_sampler_fn(
                self.model,
                obs_batch=input_dict,
                state_batches=state_batches,
                explore=explore,
                timestep=timestep,
            )
        else:
            # Call the exploration before_compute_actions hook.
            self.exploration.before_compute_actions(explore=explore, timestep=timestep)
            if is_overridden(self.action_distribution_fn):
                dist_inputs, dist_class, state_out = self.action_distribution_fn(
                    self.model,
                    obs_batch=input_dict,
                    state_batches=state_batches,
                    seq_lens=seq_lens,
                    explore=explore,
                    timestep=timestep,
                    is_training=False,
                )
            else:
                dist_class = self.dist_class
                dist_inputs, state_out = self.model(input_dict, state_batches, seq_lens)

            if not (
                isinstance(dist_class, functools.partial)
                or issubclass(dist_class, TorchDistributionWrapper)
            ):
                raise ValueError(
                    "`dist_class` ({}) not a TorchDistributionWrapper "
                    "subclass! Make sure your `action_distribution_fn` or "
                    "`make_model_and_action_dist` return a correct "
                    "distribution class.".format(dist_class.__name__)
                )
            action_dist = dist_class(dist_inputs, self.model)

            # Get the exploration action from the forward results.
            actions, logp = self.exploration.get_exploration_action(
                action_distribution=action_dist, timestep=timestep, explore=explore
            )

        # Add default and custom fetches.
        if extra_fetches is None:
            extra_fetches = self.extra_action_out(
                input_dict, state_batches, self.model, action_dist
            )

        # Action-dist inputs.
        if dist_inputs is not None:
            extra_fetches[SampleBatch.ACTION_DIST_INPUTS] = dist_inputs

        # Action-logp and action-prob.
        if logp is not None:
            extra_fetches[SampleBatch.ACTION_PROB] = torch.exp(logp.float())
            extra_fetches[SampleBatch.ACTION_LOGP] = logp

        # Update our global timestep by the batch size.
        self.global_timestep += len(input_dict[SampleBatch.CUR_OBS])
        return convert_to_numpy((actions, state_out, extra_fetches))

    def _lazy_tensor_dict(self, postprocessed_batch: SampleBatch, device=None):
        if not isinstance(postprocessed_batch, SampleBatch):
            postprocessed_batch = SampleBatch(postprocessed_batch)
        postprocessed_batch.set_get_interceptor(
            functools.partial(convert_to_torch_tensor, device=device or self.device)
        )
        return postprocessed_batch

    def _multi_gpu_parallel_grad_calc(
        self, sample_batches: List[SampleBatch]
    ) -> List[Tuple[List[TensorType], GradInfoDict]]:
        """Performs a parallelized loss and gradient calculation over the batch.

        Splits up the given train batch into n shards (n=number of this
        Policy's devices) and passes each data shard (in parallel) through
        the loss function using the individual devices' models
        (self.model_gpu_towers). Then returns each tower's outputs.

        Args:
            sample_batches: A list of SampleBatch shards to
                calculate loss and gradients for.

        Returns:
            A list (one item per device) of 2-tuples, each with 1) gradient
            list and 2) grad info dict.
        """
        assert len(self.model_gpu_towers) == len(sample_batches)
        lock = threading.Lock()
        results = {}
        grad_enabled = torch.is_grad_enabled()

        def _worker(shard_idx, model, sample_batch, device):
            torch.set_grad_enabled(grad_enabled)
            try:
                with NullContextManager() if device.type == "cpu" else torch.cuda.device(  # noqa: E501
                    device
                ):
                    loss_out = force_list(
                        self.loss(model, self.dist_class, sample_batch)
                    )

                    # Call Model's custom-loss with Policy loss outputs and
                    # train_batch.
                    if hasattr(model, "custom_loss"):
                        loss_out = model.custom_loss(loss_out, sample_batch)

                    assert len(loss_out) == len(self.optimizers)

                    # Loop through all optimizers.
                    grad_info = {"allreduce_latency": 0.0}

                    parameters = list(model.parameters())
                    all_grads = [None for _ in range(len(parameters))]
                    for opt_idx, opt in enumerate(self.optimizers):
                        # Erase gradients in all vars of the tower that this
                        # optimizer would affect.
                        param_indices = self.multi_gpu_param_groups[opt_idx]
                        for param_idx, param in enumerate(parameters):
                            if param_idx in param_indices and param.grad is not None:
                                param.grad.data.zero_()
                        # Recompute gradients of loss over all variables.
                        loss_out[opt_idx].backward(retain_graph=True)
                        grad_info.update(
                            self.extra_grad_process(opt, loss_out[opt_idx])
                        )

                        grads = []
                        # Note that return values are just references;
                        # Calling zero_grad would modify the values.
                        for param_idx, param in enumerate(parameters):
                            if param_idx in param_indices:
                                if param.grad is not None:
                                    grads.append(param.grad)
                                all_grads[param_idx] = param.grad

                        if self.distributed_world_size:
                            start = time.time()
                            if torch.cuda.is_available():
                                # Sadly, allreduce_coalesced does not work with
                                # CUDA yet.
                                for g in grads:
                                    torch.distributed.all_reduce(
                                        g, op=torch.distributed.ReduceOp.SUM
                                    )
                            else:
                                torch.distributed.all_reduce_coalesced(
                                    grads, op=torch.distributed.ReduceOp.SUM
                                )

                            for param_group in opt.param_groups:
                                for p in param_group["params"]:
                                    if p.grad is not None:
                                        p.grad /= self.distributed_world_size

                            grad_info["allreduce_latency"] += time.time() - start

                with lock:
                    results[shard_idx] = (all_grads, grad_info)
            except Exception as e:
                import traceback

                with lock:
                    results[shard_idx] = (
                        ValueError(
                            e.args[0]
                            + "\n traceback"
                            + traceback.format_exc()
                            + "\n"
                            + "In tower {} on device {}".format(shard_idx, device)
                        ),
                        e,
                    )

        # Single device (GPU) or fake-GPU case (serialize for better
        # debugging).
        if len(self.devices) == 1 or self.config["_fake_gpus"]:
            for shard_idx, (model, sample_batch, device) in enumerate(
                zip(self.model_gpu_towers, sample_batches, self.devices)
            ):
                _worker(shard_idx, model, sample_batch, device)
                # Raise errors right away for better debugging.
                last_result = results[len(results) - 1]
                if isinstance(last_result[0], ValueError):
                    raise last_result[0] from last_result[1]
        # Multi device (GPU) case: Parallelize via threads.
        else:
            threads = [
                threading.Thread(
                    target=_worker, args=(shard_idx, model, sample_batch, device)
                )
                for shard_idx, (model, sample_batch, device) in enumerate(
                    zip(self.model_gpu_towers, sample_batches, self.devices)
                )
            ]

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        # Gather all threads' outputs and return.
        outputs = []
        for shard_idx in range(len(sample_batches)):
            output = results[shard_idx]
            if isinstance(output[0], Exception):
                raise output[0] from output[1]
            outputs.append(results[shard_idx])
        return outputs
