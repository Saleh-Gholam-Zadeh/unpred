import torch
nn = torch.nn
from dynamics_models.rkn.acrkn.Decoder import SplitDiagGaussianDecoder
from utils.ConfigDict import ConfigDict
from typing import Tuple
import numpy as np


class FfnnDecoder(nn.Module):
    @staticmethod
    def get_default_config() -> ConfigDict:
        config = ConfigDict(
            decoder_hidden_units=[60],
            dec_out_norm='post',
            clip_gradients=True
        )
        config.finalize_adding()
        return config

    def __init__(self, ltd: int, target_dim: int, obs_dim: int, act_dim: int, config: ConfigDict = None, use_cuda_if_available: bool = True):
        super(FfnnDecoder, self).__init__()
        """
               :param target_dim:
               :param lod:
               :param config:
               :param use_cuda_if_available:
               """

        self._device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        if config == None:
            self.c = self.get_default_config()
        else:
            self.c = config
        self._ltd = ltd
        self._num_hidden_list = self.c.decoder_hidden_units
        self._target_dim = target_dim
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        if config == None:
            self.c = self.get_default_config()
        else:
            self.c = config

        ###### ACRKN DECODER OBJECT DEFINED
        SplitDiagGaussianDecoder._build_hidden_layers_mean = self._build_dec_hidden_layers_mean
        SplitDiagGaussianDecoder._build_hidden_layers_var = self._build_dec_hidden_layers_var
        self._dec = SplitDiagGaussianDecoder(out_dim=self._target_dim).to(self._device)




        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches

    def _flatten(self, x, y, z):
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)
        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        """
        self.batch_size, self.num_points, self.x_dim = x.size()
        self.y_dim = y.size()[-1]
        self.z_dim = z.size()[-1]
        # Flatten tensors, as encoder expects one dimensional inputs
        x_flat = x.contiguous().view(self.batch_size * self.num_points, self.x_dim)
        y_flat = y.contiguous().view(self.batch_size * self.num_points, self.y_dim)
        z_flat = z.contiguous().view(self.batch_size * self.num_points, self.z_dim)
        return x_flat, y_flat, z_flat

    def _build_dec_hidden_layers_mean(self):
        layers = []
        if self.c.latent_task_conditioning:
            last_hidden = self._ltd + self._obs_dim + self._act_dim
        else:
            last_hidden =  self._obs_dim + self._act_dim

        # hidden layer 1
        for hidden_dim in self._num_hidden_list:
            layers.append(nn.Linear(in_features=last_hidden, out_features=hidden_dim))
            layers.append(nn.ReLU())
            last_hidden = hidden_dim
        return nn.ModuleList(layers), last_hidden

    def _build_dec_hidden_layers_var(self):
        layers = []
        if self.c.latent_task_conditioning:
            last_hidden = self._ltd + self._obs_dim + self._act_dim
        else:
            last_hidden = self._obs_dim + self._act_dim
        # hidden layer 1
        for hidden_dim in self._num_hidden_list:
            layers.append(nn.Linear(in_features=last_hidden, out_features=hidden_dim))
            layers.append(nn.ReLU())
            last_hidden = hidden_dim
        return nn.ModuleList(layers), last_hidden

    def forward(self, obs_batch: torch.Tensor, act_batch: torch.Tensor, latent_task: torch.Tensor, obs_valid_batch: torch.Tensor) -> Tuple[float, float]:
        """Single update step on a batch
        :param obs_batch: batch of observation sequences
        :param act_batch: batch of action sequences
        :param obs_valid_batch: batch of observation valid flag sequences / incase of multistep predictions
        :param target_batch: batch of target sequences
        :param decode: whether to decode next_prior
        """
        latent_task = torch.unsqueeze(latent_task,1).repeat((1,obs_batch.shape[1],1))
        obs_flat, act_flat, latent_task_flat = self._flatten(obs_batch, act_batch, latent_task)
        if self.c.latent_task_conditioning:
            mean_inp = torch.cat((obs_flat,act_flat,latent_task_flat), dim=-1); var_inp = torch.cat((obs_flat,act_flat,latent_task_flat), dim=-1);
        else:
            mean_inp = torch.cat((obs_flat, act_flat), dim=-1);
            var_inp = torch.cat((obs_flat, act_flat), dim=-1);

        mean_flat, var_flat = self._dec(mean_inp, var_inp)
        mean = mean_flat.view(self.batch_size, self.num_points, self._target_dim)
        var = var_flat.view(self.batch_size, self.num_points, self._target_dim)
        return mean, var

