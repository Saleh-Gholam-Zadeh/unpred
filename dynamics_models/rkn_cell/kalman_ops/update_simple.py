import torch
import numpy as np
from utils.ConfigDict import ConfigDict
from typing import Iterable, Tuple, List
nn = torch.nn

class Update(nn.Module):


    def __init__(self, latent_obs_dim: int, config: ConfigDict, dtype: torch.dtype = torch.float32):
        """
        RKN Cell (mostly) as described in the original RKN paper
        :param latent_obs_dim: latent observation dimension
        :param config: config dict object, for configuring the cell
        :param dtype: datatype
        """
        super(Update, self).__init__()
        self._lod = latent_obs_dim
        self._lsd = 2 * self._lod

        self.c = config
        self._dtype = dtype


    @property
    def _device(self):
        return self._tm_11_full.device

 #   @torch.jit.script_method
    def forward(self, prior_mean: torch.Tensor, prior_cov: Iterable[torch.Tensor], obs: torch.Tensor, obs_var: torch.Tensor, obs_valid: torch.Tensor = None) -> \
            Tuple[torch.Tensor, Iterable[torch.Tensor], torch.Tensor, Iterable[torch.Tensor]]:
        """
        forward pass trough the cell. For proper recurrent model feed back outputs 3 and 4 (next prior belief at next
        time step

        :param prior_mean: prior mean at time t
        :param prior_cov: prior covariance at time t
        :param obs: observation at time t
        :param obs_var: observation variance at time t
        :param obs_valid: flag indicating whether observation at time t valid
        :return: posterior mean at time t, posterior covariance at time t
                 prior mean at time t + 1, prior covariance time t + 1
        """

        if self.c.never_invalid:
            post_mean, post_cov = self._update(prior_mean, prior_cov, obs, obs_var)
        else:
            assert obs_valid is not None
            post_mean, post_cov = self._masked_update(prior_mean, prior_cov, obs, obs_var, obs_valid)

        return  post_mean, post_cov

    def _update(self, prior_mean: torch.Tensor, prior_cov: Iterable[torch.Tensor],
                obs_mean: torch.Tensor, obs_var: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Performs update step
        :param prior_mean: current prior state mean
        :param prior_cov: current prior state covariance
        :param obs_mean: current observation mean
        :param obs_var: current covariance mean
        :return: current posterior state and covariance
        """

        # compute kalman gain (eq 2 and 3 in paper)
        denominator = prior_cov + obs_var
        q = prior_cov / denominator

        # update mean (eq 4 in paper)
        residual = obs_mean - prior_mean[:, :self._lod]
        new_mean = prior_mean + q * residual

        # update covariance (eq 5 -7 in paper)
        covar_factor = 1 - q
        new_covar = covar_factor * prior_cov

        return new_mean, new_covar


    def _masked_update(self,
                       prior_mean: torch.Tensor, prior_covar: Iterable[torch.Tensor],
                       obs_mean: torch.Tensor, obs_var: torch.Tensor, obs_valid: torch.Tensor) \
            -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """ Ensures update only happens if observation is valid
        :param prior_mean: current prior state mean
        :param prior_covar: current prior state covariance
        :param obs_mean: current observation mean
        :param obs_var: current observation covariance
        :param obs_valid: indicating if observation is valid
        :return: current posterior state mean and covariance
        """
        # obs_valid = obs_valid[..., None]
        update_mean, update_covar = self._update(prior_mean, prior_covar, obs_mean, obs_var)

        masked_mean = update_mean.where(obs_valid, prior_mean)
        masked_covar = update_covar.where(obs_valid, prior_covar)
        return masked_mean, masked_covar