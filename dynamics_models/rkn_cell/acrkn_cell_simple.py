import torch
import numpy as np
from utils.ConfigDict import ConfigDict
from typing import Iterable, Tuple, List
nn = torch.nn


def bmv(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Batched Matrix Vector Product"""
    return torch.bmm(mat, vec[..., None])[..., 0]


def dadat(a: torch.Tensor, diag_mat: torch.Tensor) -> torch.Tensor:
    """Batched computation of diagonal entries of (A * diag_mat * A^T) where A is a batch of square matrices and
    diag_mat is a batch of diagonal matrices (represented as vectors containing diagonal entries)
    :param a: batch of square matrices,
    :param diag_mat: batch of diagonal matrices (represented as vecotrs containing diagonal entries
    :returns diagonal entries of  A * diag_mat * A^T"""
    return bmv(a**2, diag_mat)


def dadbt(a: torch.Tensor, diag_mat: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched computation of diagonal entries of (A * diag_mat * B^T) where A and B are batches of square matrices and
     diag_mat is a batch of diagonal matrices (represented as vectors containing diagonal entries)
    :param a: batch square matrices
    :param diag_mat: batch of diagonal matrices (represented as vecotrs containing diagonal entries
    :param b: batch of square matrices
    :returns diagonal entries of  A * diag_mat * B^T"""
    return bmv(a * b, diag_mat)


def elup1(x: torch.Tensor) -> torch.Tensor:
    """
    elu + 1 activation faction to ensure positive covariances
    :param x: input
    :return: exp(x) if x < 0 else x + 1
    """
    return torch.exp(x).where(x < 0.0, x + 1.0)


def elup1_inv(x: torch.Tensor) -> torch.Tensor:
    """
    inverse of elu+1, numpy only, for initialization
    :param x: input
    :return:
    """
    return np.log(x) if x < 1.0 else (x - 1.0)


class AcRKNCell(nn.Module):

    @staticmethod
    def get_default_config() -> ConfigDict:
        config = ConfigDict(
            num_basis=15,
            bandwidth=3,
            trans_net_hidden_units=[],
            control_net_hidden_units=[60],
            trans_net_hidden_activation="Tanh",
            control_net_hidden_activation='ReLU',
            learn_trans_covar=True,
            trans_covar=0.1,
            learn_initial_state_covar=True,
            initial_state_covar=10,
            learning_rate=7e-3,
            enc_out_norm='post',
            clip_gradients=True,
            never_invalid=True
        )
        config.finalize_adding()
        return config

    def __init__(self, latent_obs_dim: int, act_dim: int, config: ConfigDict = None, dtype: torch.dtype = torch.float32):
        """
        RKN Cell (mostly) as described in the original RKN paper
        :param latent_obs_dim: latent observation dimension
        :param config: config dict object, for configuring the cell
        :param dtype: datatype
        """
        super(AcRKNCell, self).__init__()
        self._lod = latent_obs_dim
        self._lsd = self._lod
        self._lad = act_dim

        if config==None:
            self.c = AcRKNCell.get_default_config()
        else:
            self.c = config
        self._dtype = dtype



        self._build_transition_model()

    @property
    def _device(self):
        return self._tm_11_full.device

#    @torch.jit.script_method
    def forward(self, prior_mean: torch.Tensor, prior_cov: Iterable[torch.Tensor], action: torch.Tensor,
                obs: torch.Tensor, obs_var: torch.Tensor, obs_valid: torch.Tensor = None) -> \
            Tuple[torch.Tensor, Iterable[torch.Tensor], torch.Tensor, Iterable[torch.Tensor]]:
        """
        forward pass trough the cell. For proper recurrent model feed back outputs 3 and 4 (next prior belief at next
        time step

        :param prior_mean: prior mean at time t
        :param prior_cov: prior covariance at time t
        :param action: action at time t
        :param obs: observation at time t
        :param obs_var: observation variance at time t
        :param obs_valid: flag indicating whether observation at time t valid
        :return: posterior mean at time t, posterior covariance at time t
                 prior mean at time t + 1, prior covariance time t + 1
        """

        #print(prior_mean.shape, prior_cov[0].shape, prior_cov[1].shape, prior_cov[2].shape)

        if self.c.never_invalid:
            post_mean, post_cov = self._update(prior_mean, prior_cov, obs, obs_var)
        else:
            assert obs_valid is not None
            post_mean, post_cov = self._masked_update(prior_mean, prior_cov, obs, obs_var, obs_valid)

        #print(post_mean.shape, post_cov[0].shape,post_cov[1].shape, post_cov[2].shape)

        next_prior_mean, next_prior_cov = self._predict(post_mean, post_cov, action)


        return  post_mean, post_cov, next_prior_mean, next_prior_cov

    def _build_coefficient_net(self, num_hidden: Iterable[int], activation: str) -> torch.nn.Sequential:
        """
        builds the network computing the coefficients from the posterior mean. Currently only fully connected
        neural networks with same activation across all hidden layers supported
        TODO: Allow more flexible architectures
        :param num_hidden: number of hidden uints per layer
        :param activation: hidden activation
        :return: coefficient network
        """
        layers = []
        prev_dim = self._lsd
        for n in num_hidden:
            layers.append(nn.Linear(prev_dim, n))
            layers.append(getattr(nn, activation)())
            prev_dim = n
        layers.append(nn.Linear(prev_dim, self.c.num_basis))
        layers.append(nn.Softmax(dim=-1))
        return nn.Sequential(*layers).to(dtype=self._dtype)

    def _build_control_net(self, num_hidden: Iterable[int], activation: str) -> torch.nn.Sequential:
        """
        builds the control network computing the contribution of the actions towards state transitions.
        Currently only fully connected neural networks with same activation across all hidden layers supported
        TODO: Allow more flexible architectures
        :param num_hidden: number of hidden units per layer
        :param activation: hidden activation
        :return: coefficient network
        """
        layers = []
        prev_dim = self._lad
        for n in num_hidden:
            layers.append(nn.Linear(prev_dim, n))
            layers.append(getattr(nn, activation)())
            prev_dim = n
        layers.append(nn.Linear(prev_dim, self._lsd))
        return nn.Sequential(*layers).to(dtype=self._dtype)


    def _build_transition_model(self) -> None:
        """
        Builds the basis functions for transition model and the nosie
        :return:
        """
        # build state independent basis matrices
        self._transition_matrices = nn.Parameter(0.8*torch.ones(self._lsd, dtype=self._dtype)[None, :].repeat(self.c.num_basis, 1))
        self._control_net = self._build_control_net(self.c.control_net_hidden_units,
                                                    self.c.control_net_hidden_activation)
        self._coefficient_net = self._build_coefficient_net(self.c.trans_net_hidden_units,
                                                           self.c.trans_net_hidden_activation)

        init_trans_cov = elup1_inv(self.c.trans_covar)
        # TODO: This is currently a different noise for each dim, not like in original paper (and acrkn)
        self._log_transition_noise = \
            nn.Parameter(nn.init.constant_(torch.empty(1, self._lsd, dtype=self._dtype), init_trans_cov))

    def get_transition_model(self, post_mean: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Compute the locally-linear transition model given the current posterior mean
        :param post_mean: current posterior mean
        :return: transition matrices (4 Blocks), transition covariance (vector of size lsd)
        """
        # prepare transition model
        coefficients = torch.reshape(self._coefficient_net(post_mean), [-1, self.c.num_basis, 1])

        tm_full =  self._transition_matrices[None,: ,:]
        tm_full = (coefficients * tm_full).sum(dim=1)

        trans_cov = elup1(self._log_transition_noise)

        return tm_full, trans_cov

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

    def _predict(self, post_mean: torch.Tensor, post_covar: List[torch.Tensor], action: torch.Tensor) \
            -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """ Performs prediction step
        :param post_mean: last posterior mean
        :param post_covar: last posterior covariance
        :return: current prior state mean and covariance
        """
        # compute state dependent transition matrix
        tm, trans_covar = self.get_transition_model(post_mean)
        control_factor = self._control_net(action)

        mu_prior = tm * post_mean + control_factor

        # predict next prior covariance (eq 10 - 12 in paper supplement)
        cov_prior = tm ** 2 * post_covar + trans_covar

        return mu_prior, cov_prior


