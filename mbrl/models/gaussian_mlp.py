# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F

import mbrl.util.math

from .model import Ensemble
from .util import EnsembleLinearLayer, truncated_normal_init
import numpy as np


class GaussianMLP(Ensemble):
    """Implements an ensemble of multi-layer perceptrons each modeling a Gaussian distribution.

    This model corresponds to a Probabilistic Ensemble in the Chua et al.,
    NeurIPS 2018 paper (PETS) https://arxiv.org/pdf/1805.12114.pdf

    It predicts per output mean and log variance, and its weights are updated using a Gaussian
    negative log likelihood loss. The log variance is bounded between learned ``min_log_var``
    and ``max_log_var`` parameters, trained as explained in Appendix A.1 of the paper.

    This class can also be used to build an ensemble of GaussianMLP models, by setting
    ``ensemble_size > 1`` in the constructor. Then, a single forward pass can be used to evaluate
    multiple independent MLPs at the same time. When this mode is active, the constructor will
    set ``self.num_members = ensemble_size``.

    For the ensemble variant, uncertainty propagation methods are available that can be used
    to aggregate the outputs of the different models in the ensemble.
    Valid propagation options are:

            - "random_model": for each output in the batch a model will be chosen at random.
              This corresponds to TS1 propagation in the PETS paper.
            - "fixed_model": for output j-th in the batch, the model will be chosen according to
              the model index in `propagation_indices[j]`. This can be used to implement TSinf
              propagation, described in the PETS paper.
            - "expectation": the output for each element in the batch will be the mean across
              models.

    The default value of ``None`` indicates that no uncertainty propagation, and the forward
    method returns all outputs of all models.

    Args:
        in_size (int): size of model input.
        out_size (int): size of model output.
        device (str or torch.device): the device to use for the model.
        num_layers (int): the number of layers in the model
                          (e.g., if ``num_layers == 3``, then model graph looks like
                          input -h1-> -h2-> -l3-> output).
        ensemble_size (int): the number of members in the ensemble. Defaults to 1.
        hid_size (int): the size of the hidden layers (e.g., size of h1 and h2 in the graph above).
        deterministic (bool): if ``True``, the model will be trained using MSE loss and no
            logvar prediction will be done. Defaults to ``False``.
        propagation_method (str, optional): the uncertainty propagation method to use (see
            above). Defaults to ``None``.
        learn_logvar_bounds (bool): if ``True``, the logvar bounds will be learned, otherwise
            they will be constant. Defaults to ``False``.
        activation_fn_cfg (dict or omegaconf.DictConfig, optional): configuration of the
            desired activation function. Defaults to torch.nn.ReLU when ``None``.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        num_layers: int = 4,
        ensemble_size: int = 1,
        hid_size: int = 200,
        deterministic: bool = False,
        propagation_method: Optional[str] = None,
        learn_logvar_bounds: bool = False,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
    ):
        super().__init__(
            ensemble_size, device, propagation_method, deterministic=deterministic
        )

        self.in_size = in_size
        self.out_size = out_size

        def create_activation():
            if activation_fn_cfg is None:
                activation_func = nn.ReLU()
            else:
                # Handle the case where activation_fn_cfg is a dict
                cfg = omegaconf.OmegaConf.create(activation_fn_cfg)
                activation_func = hydra.utils.instantiate(cfg)
            return activation_func

        def create_linear_layer(l_in, l_out):
            return EnsembleLinearLayer(ensemble_size, l_in, l_out)

        hidden_layers = [
            nn.Sequential(create_linear_layer(in_size, hid_size), create_activation())
        ]
        for i in range(num_layers - 1):
            hidden_layers.append(
                nn.Sequential(
                    create_linear_layer(hid_size, hid_size),
                    create_activation(),
                )
            )
        self.hidden_layers = nn.Sequential(*hidden_layers)

        if deterministic:
            self.mean_and_logvar = create_linear_layer(hid_size, out_size)
        else:
            self.mean_and_logvar = create_linear_layer(hid_size, 2 * out_size)
            self.min_logvar = nn.Parameter(
                -10 * torch.ones(1, out_size), requires_grad=learn_logvar_bounds
            )
            self.max_logvar = nn.Parameter(
                0.5 * torch.ones(1, out_size), requires_grad=learn_logvar_bounds
            )

        self.apply(truncated_normal_init)
        self.to(self.device)

        self.elite_models: List[int] = None

    def _maybe_toggle_layers_use_only_elite(self, only_elite: bool):
        if self.elite_models is None:
            return
        if self.num_members > 1 and only_elite:
            for layer in self.hidden_layers:
                # each layer is (linear layer, activation_func)
                layer[0].set_elite(self.elite_models)
                layer[0].toggle_use_only_elite()
            self.mean_and_logvar.set_elite(self.elite_models)
            self.mean_and_logvar.toggle_use_only_elite()

    def _default_forward(
        self, x: torch.Tensor, only_elite: bool = False, **_kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._maybe_toggle_layers_use_only_elite(only_elite)
        x = self.hidden_layers(x)
        mean_and_logvar = self.mean_and_logvar(x)
        self._maybe_toggle_layers_use_only_elite(only_elite)
        if self.deterministic:
            return mean_and_logvar, None
        else:
            mean = mean_and_logvar[..., : self.out_size]
            logvar = mean_and_logvar[..., self.out_size :]
            logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
            return mean, logvar

    def _forward_from_indices(
        self, x: torch.Tensor, model_shuffle_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, batch_size, _ = x.shape

        num_models = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        shuffled_x = x[:, model_shuffle_indices, ...].view(
            num_models, batch_size // num_models, -1
        )

        mean, logvar = self._default_forward(shuffled_x, only_elite=True)
        # note that mean and logvar are shuffled
        mean = mean.view(batch_size, -1)
        mean[model_shuffle_indices] = mean.clone()  # invert the shuffle

        if logvar is not None:
            logvar = logvar.view(batch_size, -1)
            logvar[model_shuffle_indices] = logvar.clone()  # invert the shuffle

        return mean, logvar

    def _forward_ensemble(
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.propagation_method is None:
            mean, logvar = self._default_forward(x, only_elite=False)
            if self.num_members == 1:
                mean = mean[0]
                logvar = logvar[0] if logvar is not None else None
            return mean, logvar
        assert x.ndim == 2
        model_len = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        if x.shape[0] % model_len != 0:
            raise ValueError(
                f"GaussianMLP ensemble requires batch size to be a multiple of the "
                f"number of models. Current batch size is {x.shape[0]} for "
                f"{model_len} models."
            )
        x = x.unsqueeze(0)
        if self.propagation_method == "random_model":
            # passing generator causes segmentation fault
            # see https://github.com/pytorch/pytorch/issues/44714
            model_indices = torch.randperm(x.shape[1], device=self.device)
            return self._forward_from_indices(x, model_indices)
        if self.propagation_method == "fixed_model":
            if propagation_indices is None:
                raise ValueError(
                    "When using propagation='fixed_model', `propagation_indices` must be provided."
                )
            return self._forward_from_indices(x, propagation_indices)
        if self.propagation_method == "expectation":
            mean, logvar = self._default_forward(x, only_elite=True)
            return mean.mean(dim=0), logvar.mean(dim=0)
        raise ValueError(f"Invalid propagation method {self.propagation_method}.")

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
        use_propagation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes mean and logvar predictions for the given input.

        When ``self.num_members > 1``, the model supports uncertainty propagation options
        that can be used to aggregate the outputs of the different models in the ensemble.
        Valid propagation options are:

            - "random_model": for each output in the batch a model will be chosen at random.
              This corresponds to TS1 propagation in the PETS paper.
            - "fixed_model": for output j-th in the batch, the model will be chosen according to
              the model index in `propagation_indices[j]`. This can be used to implement TSinf
              propagation, described in the PETS paper.
            - "expectation": the output for each element in the batch will be the mean across
              models.

        If a set of elite models has been indicated (via :meth:`set_elite()`), then all
        propagation methods will operate with only on the elite set. This has no effect when
        ``propagation is None``, in which case the forward pass will return one output for
        each model.

        Args:
            x (tensor): the input to the model. When ``self.propagation is None``,
                the shape must be ``E x B x Id`` or ``B x Id``, where ``E``, ``B``
                and ``Id`` represent ensemble size, batch size, and input dimension,
                respectively. In this case, each model in the ensemble will get one slice
                from the first dimension (e.g., the i-th ensemble member gets ``x[i]``).

                For other values of ``self.propagation`` (and ``use_propagation=True``),
                the shape must be ``B x Id``.
            rng (torch.Generator, optional): random number generator to use for "random_model"
                propagation.
            propagation_indices (tensor, optional): propagation indices to use,
                as generated by :meth:`sample_propagation_indices`. Ignore if
                `use_propagation == False` or `self.propagation_method != "fixed_model".
            use_propagation (bool): if ``False``, the propagation method will be ignored
                and the method will return outputs for all models. Defaults to ``True``.

        Returns:
            (tuple of two tensors): the predicted mean and log variance of the output. If
            ``propagation is not None``, the output will be 2-D (batch size, and output dimension).
            Otherwise, the outputs will have shape ``E x B x Od``, where ``Od`` represents
            output dimension.

        Note:
            For efficiency considerations, the propagation method used by this class is an
            approximate version of that described by Chua et al. In particular, instead of
            sampling models independently for each input in the batch, we ensure that each
            model gets exactly the same number of samples (which are assigned randomly
            with equal probability), resulting in a smaller batch size which we use for the forward
            pass. If this is a concern, consider using ``propagation=None``, and passing
            the output to :func:`mbrl.util.math.propagate`.

        """
        if use_propagation:
            return self._forward_ensemble(
                x, rng=rng, propagation_indices=propagation_indices
            )
        return self._default_forward(x)

    def _mse_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add model dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_mean, _ = self.forward(model_in, use_propagation=False)
        return F.mse_loss(pred_mean, target, reduction="none").sum((1, 2)).sum()

    def _nll_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add ensemble dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_mean, pred_logvar = self.forward(model_in, use_propagation=False)
        if target.shape[0] != self.num_members:
            target = target.repeat(self.num_members, 1, 1)
        nll = (
            mbrl.util.math.gaussian_nll(pred_mean, pred_logvar, target, reduce=False)
            .mean((1, 2))  # average over batch and target dimension
            .sum()
        )  # sum over ensemble dimension
        nll += 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        return nll

    def loss(
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes Gaussian NLL loss.

        It also includes terms for ``max_logvar`` and ``min_logvar`` with small weights,
        with positive and negative signs, respectively.

        This function returns no metadata, so the second output is set to an empty dict.

        Args:
            model_in (tensor): input tensor. The shape must be ``E x B x Id``, or ``B x Id``
                where ``E``, ``B`` and ``Id`` represent ensemble size, batch size, and input
                dimension, respectively.
            target (tensor): target tensor. The shape must be ``E x B x Id``, or ``B x Od``
                where ``E``, ``B`` and ``Od`` represent ensemble size, batch size, and output
                dimension, respectively.

        Returns:
            (tensor): a loss tensor representing the Gaussian negative log-likelihood of
            the model over the given input/target. If the model is an ensemble, returns
            the average over all models.
        """
        if self.deterministic:
            return self._mse_loss(model_in, target), {}
        else:
            return self._nll_loss(model_in, target), {}

    def eval_score(  # type: ignore
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the squared error for the model over the given input/target.

        When model is not an ensemble, this is equivalent to
        `F.mse_loss(model(model_in, target), reduction="none")`. If the model is ensemble,
        then return is batched over the model dimension.

        This function returns no metadata, so the second output is set to an empty dict.

        Args:
            model_in (tensor): input tensor. The shape must be ``B x Id``, where `B`` and ``Id``
                batch size, and input dimension, respectively.
            target (tensor): target tensor. The shape must be ``B x Od``, where ``B`` and ``Od``
                represent batch size, and output dimension, respectively.

        Returns:
            (tensor): a tensor with the squared error per output dimension, batched over model.
        """
        assert model_in.ndim == 2 and target.ndim == 2
        with torch.no_grad():
            pred_mean, _ = self.forward(model_in, use_propagation=False)
            target = target.repeat((self.num_members, 1, 1))
            return F.mse_loss(pred_mean, target, reduction="none"), {}

    def sample_propagation_indices(
        self, batch_size: int, _rng: torch.Generator
    ) -> torch.Tensor:
        model_len = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        if batch_size % model_len != 0:
            raise ValueError(
                "To use GaussianMLP's ensemble propagation, the batch size must "
                "be a multiple of the number of models in the ensemble."
            )
        # rng causes segmentation fault, see https://github.com/pytorch/pytorch/issues/44714
        return torch.randperm(batch_size, device=self.device)

    def set_elite(self, elite_indices: Sequence[int]):
        if len(elite_indices) != self.num_members:
            self.elite_models = list(elite_indices)

    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves the model to the given directory."""
        model_dict = {
            "state_dict": self.state_dict(),
            "elite_models": self.elite_models,
        }
        torch.save(model_dict, pathlib.Path(save_dir) / self._MODEL_FNAME)

    def load(self, load_dir: Union[str, pathlib.Path]):
        """Loads the model from the given path."""
        model_dict = torch.load(pathlib.Path(load_dir) / self._MODEL_FNAME)
        self.load_state_dict(model_dict["state_dict"])
        self.elite_models = model_dict["elite_models"]
    
    def info_sample_1d(
        self,
        model_input: torch.Tensor,
        model_state: Dict[str, torch.Tensor],
        deterministic: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Samples an output from the model using .

        This method will be used by :class:`ModelEnv` to simulate a transition of the form.
            outputs_t+1, s_t+1 = sample(model_input_t, s_t), where

            - model_input_t: observation and action at time t, concatenated across axis=1.
            - s_t: model state at time t (as returned by :meth:`reset()` or :meth:`sample()`.
            - outputs_t+1: observation and reward at time t+1, concatenated across axis=1.

        The default implementation returns `s_t+1=s_t`.

        Args:
            model_input (tensor): the observation and action at.
            model_state (tensor): the model state st. Must contain a key
                "propagation_indices" to use for uncertainty propagation.
            deterministic (bool): if ``True``, the model returns a deterministic
                "sample" (e.g., the mean prediction). Defaults to ``False``.
            rng (`torch.Generator`, optional): an optional random number generator
                to use.

        Returns:
            (tuple): predicted observation, rewards, terminal indicator and model
                state dictionary. Everything but the observation is optional, and can
                be returned with value ``None``.
        """
        if deterministic or self.deterministic:
            raise NotImplementedError
        assert rng is not None
        means, logvars = self.forward(
            model_input, use_propagation=False
        )
        variances = logvars.exp()
        fused_var = 1/((1/variances).mean(dim=0))
        fused_mean = fused_var * ((1/variances)*means).mean(dim=0)
        
        deltas = (means - fused_mean.unsqueeze(0))
        epist_var = (torch.square(deltas)).mean(dim=0) 
        
        K = fused_var / (fused_var + epist_var)

        chosen_model = torch.randint(low=0,high=means.shape[0], size=(means.shape[1],), generator=rng, device=self.device)
        batch_idx =torch.arange(means.shape[1])
        sample = torch.normal(means[chosen_model, batch_idx, :], torch.sqrt(variances[chosen_model, batch_idx, :]), generator=rng)

        conditional_mean = fused_mean + K * (sample-fused_mean)

        conditional_var = (1-K)*fused_var
        
        model_state["fused_var"] = fused_var#[:,dim_idx,dim_idx]
        model_state["conditional_var"] = conditional_var#[:,dim_idx,dim_idx]
        model_state["obs_act"] = torch.normal(conditional_mean, torch.sqrt(conditional_var), generator=rng)

        return torch.normal(conditional_mean, torch.sqrt(conditional_var), generator=rng), model_state

        # conditional_sample, lost_ent = full_matrix_prop(means, variances, rng)

        # model_state["lost_ent"] = lost_ent

        # return conditional_sample, model_state
    
def full_matrix_prop(means, variances, rng=None):
    E, B, n = means.shape
    means_ = means.cpu()
    variances_ = variances.cpu()
    fused_var = 1/((1/variances_).mean(dim=0))
    fused_mean = fused_var * ((1/variances_)*means_).mean(dim=0)
    deltas = means_ - fused_var.unsqueeze(0)
    epist_var = (deltas.unsqueeze(-1) @ deltas.unsqueeze(-2)).mean(dim=0)
    fused_var_extended = torch.diag_embed(fused_var)
    total_var = fused_var_extended + epist_var
    K = torch.linalg.solve(total_var, fused_var_extended, left=False)
    conditional_var = (torch.diag_embed(torch.ones_like(fused_var)) - K) @ fused_var_extended
    L, Q = torch.linalg.eigh(conditional_var)
    L[L<1e-12] = 1e-12
    lost_ent = (0.5*(n*np.log2(2*np.pi*np.exp(1)) + torch.log2(L).sum(axis=-1))-n*np.log2(1e-6))
    chosen_model = torch.randint(low=0,high=E, size=(B,))
    batch_idx =torch.arange(B)
    sample = torch.normal(means[chosen_model, batch_idx, :], torch.sqrt(variances[chosen_model, batch_idx, :]), generator = rng)

    fused_mean = fused_mean.to(sample.device)
    K = K.to(sample.device)
    lost_ent = lost_ent.to(sample.device)
    conditional_mean = fused_mean + (K @ (sample-fused_mean).unsqueeze(-1)).squeeze(-1)

    eps = torch.randn(*(conditional_mean.shape), generator=rng, device=sample.device)

    conditional_sample = conditional_mean + (Q.cuda()@torch.diag_embed(L.cuda().sqrt())@ eps.unsqueeze(-1)).squeeze(-1)
    return conditional_sample, lost_ent
