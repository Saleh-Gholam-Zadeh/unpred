import torch
from typing import Tuple, Iterable

nn = torch.nn


def elup1(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x).where(x < 0.0, x + 1.0)


class SplitDiagGaussianDecoder(nn.Module):

    def __init__(self, out_dim: int, activation: str = 'softplus'):
        """ Decoder for low dimensional outputs as described in the paper. This one is "split", i.e., there are
        completely separate networks mapping from latent mean to output mean and from latent cov to output var
        :param lod: latent observation dim (used to compute input sizes)
        :param out_dim: dimensionality of target data (assumed to be a vector, images not supported by this decoder)
        """
        super(SplitDiagGaussianDecoder, self).__init__()
        self._out_dim = out_dim
        self._activation = activation

        self._hidden_layers_mean, num_last_hidden_mean = self._build_hidden_layers_mean()
        assert isinstance(self._hidden_layers_mean, nn.ModuleList), "_build_hidden_layers_means needs to return a " \
                                                                    "torch.nn.ModuleList or else the hidden weights " \
                                                                    "are not found by the optimizer"

        self._hidden_layers_var, num_last_hidden_var = self._build_hidden_layers_var()
        assert isinstance(self._hidden_layers_var, nn.ModuleList), "_build_hidden_layers_var needs to return a " \
                                                                   "torch.nn.ModuleList or else the hidden weights " \
                                                                   "are not found by the optimizer"

        self._out_layer_mean = nn.Linear(in_features=num_last_hidden_mean, out_features=out_dim)
        self._out_layer_var = nn.Linear(in_features=num_last_hidden_var, out_features=out_dim)
        self._softplus = nn.Softplus()

    def _build_hidden_layers_mean(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for mean decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def _build_hidden_layers_var(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for variance decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def forward(self, latent_mean: torch.Tensor, latent_cov: Iterable[torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """ forward pass of decoder
        :param latent_mean:
        :param latent_cov:
        :return: output mean and variance
        """
        h_mean = latent_mean
        for layer in self._hidden_layers_mean:
            h_mean = layer(h_mean)
        mean = self._out_layer_mean(h_mean)

        h_var = latent_cov
        for layer in self._hidden_layers_var:
            h_var = layer(h_var)
        log_var = self._out_layer_var(h_var)
        if self._activation == 'softplus':
            var = self._softplus(log_var) + 0.0001
        else:
            var = elup1(log_var)
        return mean, var

class SplitDiagCondGaussianDecoder(nn.Module):

    def __init__(self, out_dim: int, activation: str = 'softplus'):
        """ Decoder for low dimensional outputs as described in the paper. This one is "split", i.e., there are
        completely separate networks mapping from latent mean to output mean and from latent cov to output var
        :param lod: latent observation dim (used to compute input sizes)
        :param out_dim: dimensionality of target data (assumed to be a vector, images not supported by this decoder)
        """
        super(SplitDiagCondGaussianDecoder, self).__init__()
        self._out_dim = out_dim
        self._activation = activation

        self._hidden_layers_mean, num_last_hidden_mean = self._build_hidden_layers_mean()
        assert isinstance(self._hidden_layers_mean, nn.ModuleList), "_build_hidden_layers_means needs to return a " \
                                                                    "torch.nn.ModuleList or else the hidden weights " \
                                                                    "are not found by the optimizer"

        self._hidden_layers_var, num_last_hidden_var = self._build_hidden_layers_var()
        assert isinstance(self._hidden_layers_var, nn.ModuleList), "_build_hidden_layers_var needs to return a " \
                                                                   "torch.nn.ModuleList or else the hidden weights " \
                                                                   "are not found by the optimizer"

        self._out_layer_mean = nn.Linear(in_features=num_last_hidden_mean, out_features=out_dim)
        self._out_layer_var = nn.Linear(in_features=num_last_hidden_var, out_features=out_dim)
        self._softplus = nn.Softplus()

    def _build_hidden_layers_mean(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for mean decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def _build_hidden_layers_var(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for variance decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def forward(self, latent_mean: torch.Tensor, latent_cov: Iterable[torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """ forward pass of decoder
        :param latent_mean:
        :param latent_cov:
        :return: output mean and variance
        """
        h_mean = latent_mean
        for layer in self._hidden_layers_mean:
            h_mean = layer(h_mean)
        mean = self._out_layer_mean(h_mean)

        h_var = latent_cov
        for layer in self._hidden_layers_var:
            h_var = layer(h_var)
        log_var = self._out_layer_var(h_var)
        if self._activation == 'softplus':
            var = self._softplus(log_var) + 0.0001
        else:
            var = elup1(log_var)
        return mean, var

class SimpleDecoder(nn.Module):

    def __init__(self, out_dim: int, output_normalization: str = "post", activation='softplus'):
        """ Decoder for low dimensional outputs as described in the paper. This one is "split", i.e., there are
        completely separate networks mapping from latent mean to output mean and from latent cov to output var
        :param lod: latent observation dim (used to compute input sizes)
        :param out_dim: dimensionality of target data (assumed to be a vector, images not supported by this decoder)
        """
        super(SimpleDecoder, self).__init__()

        self._hidden_layers, size_last_hidden = self._build_hidden_layers()
        assert isinstance(self._hidden_layers, nn.ModuleList), "_build_hidden_layers_means needs to return a " \
                                                                    "torch.nn.ModuleList or else the hidden weights " \
                                                                    "are not found by the optimizer"

        self._mean_layer = nn.Linear(in_features=size_last_hidden, out_features=out_dim)
        self._log_var_layer = nn.Linear(in_features=size_last_hidden, out_features=out_dim)
        self._softplus = nn.Softplus()

        self._output_normalization = output_normalization
        self._activation = activation

    def _build_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for mean decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError


    def forward(self, input: torch.Tensor) \
            -> Tuple[torch.Tensor]:
        """ forward pass of decoder
        :param input:
        :return: output mean
        """
        h = input
        for layer in self._hidden_layers:
            h = layer(h)

        mean = self._mean_layer(h)

        log_var = self._log_var_layer(h)
        if self._activation == 'softplus':
            var = self._softplus(log_var) + 0.0001
        else:
            var = elup1(log_var)
        return mean, var