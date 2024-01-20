import torch
from dynamics_models.rkn_cell.acrkn_cell_simple import AcRKNCell
nn = torch.nn


class AcRKNLayer(nn.Module):

    def __init__(self, latent_obs_dim, act_dim, cell_config, dtype=torch.float32):
        super().__init__()
        self._lod = latent_obs_dim
        self._lsd = latent_obs_dim
        self._cell = AcRKNCell(latent_obs_dim, act_dim, cell_config, dtype)

    def forward(self, latent_obs, obs_vars, actions, initial_mean, initial_cov, obs_valid=None):
        """
        This currently only returns the posteriors. If you also need the priors uncomment the corresponding parts

        :param latent_obs: latent observations
        :param obs_vars: uncertainty estimate in latent observations
        :param initial_mean: mean of initial belief
        :param initial_cov: covariance of initial belief (as 3 vectors)
        :param obs_valid: flags indicating which observations are valid, which are not
        """

        # tif you need a version that also returns the prior uncomment the respective parts below
        # prepare list for return

        prior_mean_list = []
        prior_cov_list = []

        post_mean_list = []
        post_cov_list = []


        # initialize prior
        prior_mean, prior_cov = initial_mean, initial_cov

        # actual computation
        for i in range(latent_obs.shape[1]):


            cur_obs_valid = obs_valid[:, i] if obs_valid is not None else None
            post_mean, post_cov, next_prior_mean, next_prior_cov = \
                self._cell(prior_mean, prior_cov, actions[:,i], latent_obs[:, i], obs_vars[:, i], cur_obs_valid)

            post_mean_list.append(post_mean)
            post_cov_list.append(post_cov)
            prior_mean_list.append(next_prior_mean)
            prior_cov_list.append(post_cov)

            prior_mean = next_prior_mean
            prior_cov = next_prior_cov

        # stack results
        prior_means = torch.stack(prior_mean_list, 1)
        prior_covs = torch.stack(prior_cov_list, 1)

        post_means = torch.stack(post_mean_list, 1)
        post_covs = torch.stack(post_cov_list, 1)

        return post_means, post_covs, prior_means, prior_covs
