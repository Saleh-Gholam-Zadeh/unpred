### Simpler observation model = I version of v3
import torch
import numpy as np
from utils.ConfigDict import ConfigDict
from typing import Iterable, Tuple, List
import tsensor

nn = torch.nn


def bmv(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Batched Matrix Vector Product"""
    return torch.bmm(mat, vec[..., None])[..., 0]

def gaussian_linear_transform(tm: torch.Tensor, mu: torch.Tensor, covar:torch.Tensor, control_factor:torch.Tensor
                              , process_covar: torch.Tensor):
    print("contextual_prediction line 17")
    print("control_factor:",control_factor)
    control_factor = torch.zeros_like(control_factor) #saleh_added

    mu_prior = tm*mu #+ control_factor #saleh_changed

    # predict next prior covariance (eq 10 - 12 in paper supplement)
    cov_prior = tm**2 * covar + process_covar
    return mu_prior, cov_prior

def gaussian_multi_linear_transform(tm: torch.Tensor, lm: torch.Tensor, mu: torch.Tensor, covar:torch.Tensor, mu_l: torch.Tensor,
                              covar_l: torch.Tensor, control_factor:torch.Tensor, process_covar: torch.Tensor):


    print("contextual_prediction_simple.py line 31")
    print("control_factor:",control_factor)
    control_factor = torch.zeros_like(control_factor) #saleh_added

    lm_batched = lm.repeat((mu_l.shape[0],1))
    mu_prior = tm*mu + lm_batched*mu_l #+ control_factor  #saleh_added

    # predict next prior covariance (eq 10 - 12 in paper supplement)
    cov_prior = (tm**2 * covar) + (lm_batched**2 * covar_l) + process_covar
    return mu_prior, cov_prior


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


class deterministicTrans(nn.Module):
    def __init__(self, lad, lsd, num_hidden: Iterable[int], activation: str, dtype: torch.dtype = torch.float32):
        super().__init__()

        self._lad = lad
        self._lsd = lsd
        self._dtype = dtype
        layers = []
        prev_dim = self._lsd + self._lsd
        self._act_transformer = nn.Linear(self._lad, self._lsd)
        for n in num_hidden:
            layers.append(nn.Linear(prev_dim, n))
            layers.append(getattr(nn, activation)())
            prev_dim = n
        layers.append(nn.Linear(prev_dim, self._lsd))
        self._control = nn.Sequential(*layers).to(dtype=self._dtype)



    def forward(self, action: torch.Tensor, post_mean: torch.Tensor):
        x = self._act_transformer(action)
        x = self._control(torch.cat((x, post_mean), dim=-1))
        print("contextual_predict_simple.py  line 99")
        print("self._control(torch.cat((x, post_mean), dim=-1))",self._control(torch.cat((x, post_mean), dim=-1)))
        #x = self._control(action)

        return x

class ContextualDeterministicTrans(nn.Module):
    def __init__(self, lad, lsd, num_hidden: Iterable[int], activation: str, dtype: torch.dtype = torch.float32):
        super().__init__()

        self._lad = lad
        self._lsd = lsd
        self._dtype = dtype
        layers = []
        prev_dim = 3*self._lsd
        self._act_transformer = nn.Linear(self._lad, self._lsd)
        for n in num_hidden:
            layers.append(nn.Linear(prev_dim, n))
            layers.append(getattr(nn, activation)())
            prev_dim = n
        layers.append(nn.Linear(prev_dim, self._lsd))
        self._control = nn.Sequential(*layers).to(dtype=self._dtype)



    def forward(self, action: torch.Tensor, post_mean: torch.Tensor):
        x = self._act_transformer(action)
        x = self._control(torch.cat((x, post_mean), dim=-1))

        #x = self._control(action)

        return x


class Control(nn.Module):
    def __init__(self, lad, lsd, num_hidden: Iterable[int], activation: str, dtype: torch.dtype = torch.float32):
        super().__init__()

        self._lad = lad
        self._lsd = lsd
        self._dtype = dtype
        layers = []
        prev_dim = self._lad

        for n in num_hidden:
            layers.append(nn.Linear(prev_dim, n))
            layers.append(getattr(nn, activation)())
            prev_dim = n
        layers.append(nn.Linear(prev_dim, self._lsd))
        self._control = nn.Sequential(*layers).to(dtype=self._dtype)



    def forward(self, action: torch.Tensor):
        x = self._control(action)

        return x

class ContexualControl(nn.Module):
    def __init__(self, ltd, lad, lsd, num_hidden: Iterable[int], activation: str, dtype: torch.dtype = torch.float32):
        super().__init__()

        self._lad = lad
        self._ltd = ltd
        self._lsd = lsd
        self._dtype = dtype
        layers = []
        prev_dim = 2*self._ltd

        self._act_transformer = nn.Linear(self._lad, self._ltd) #To make them have similar dimension
        for n in num_hidden:
            layers.append(nn.Linear(prev_dim, n))
            layers.append(getattr(nn, activation)())
            prev_dim = n
        layers.append(nn.Linear(prev_dim, self._lsd))
        self._contextual_control = nn.Sequential(*layers).to(dtype=self._dtype)



    def forward(self, action: torch.Tensor, latent_context:torch.Tensor):
        x = self._act_transformer(action)
        x = self._contextual_control(torch.cat((x,latent_context),dim=-1))

        return x

class ProcessNoise(nn.Module):
    def __init__(self, lsd, init_trans_covar, num_hidden: Iterable[int], activation: str, dtype: torch.dtype = torch.float32):
        super().__init__()

        self._lsd = lsd
        self._dtype = dtype
        init_trans_cov = elup1_inv(init_trans_covar)
        self._log_process_noise = nn.Parameter(nn.init.constant_(torch.empty(1, self._lsd, dtype=self._dtype), init_trans_cov))




    def forward(self):
        x = self._log_process_noise

        return x

class ContexualProcessNoise(nn.Module):
    def __init__(self, ltd, lsd, num_hidden: Iterable[int], activation: str, dtype: torch.dtype = torch.float32):
        super().__init__()

        self._ltd = ltd
        self._lsd = lsd
        self._dtype = dtype
        layers = []
        prev_dim = self._ltd

        for n in num_hidden:
            layers.append(nn.Linear(prev_dim, n))
            layers.append(getattr(nn, activation)())
            prev_dim = n
        layers.append(nn.Linear(prev_dim, self._lsd))
        self._contextual_log_process_noise = nn.Sequential(*layers).to(dtype=self._dtype)



    def forward(self, latent_context:torch.Tensor):
        x = self._contextual_log_process_noise(latent_context)
        return x

class Coefficient(nn.Module):
    '''
    Implements nn Module for coefficient net that is both state and context dependent
    TODO: Make ltd and lsd to be of similar dimension
    TODO: Make separate coefficient net for contexts and targets and select hierarchically
    '''
    def __init__(self, lsd, num_basis:int,  num_hidden: Iterable[int], activation: str, dtype: torch.dtype = torch.float32):
        super().__init__()

        self._lsd = lsd
        self._dtype = dtype
        self._num_basis = num_basis
        layers = []
        prev_dim = self._lsd
        # prev_dim = self._lsd
        for n in num_hidden:
            layers.append(nn.Linear(prev_dim, n))
            layers.append(getattr(nn, activation)())
            prev_dim = n
        layers.append(nn.Linear(prev_dim, self._num_basis))
        layers.append(nn.Softmax(dim=-1))
        self._coeff = nn.Sequential(*layers).to(dtype=self._dtype)




    def forward(self, state: torch.Tensor):
        x = self._coeff(state)

        return x

class ContexualCoefficient(nn.Module):
    '''
    Implements nn Module for coefficient net that is both state and context dependent
    TODO: Make ltd and lsd to be of similar dimension
    TODO: Make separate coefficient net for contexts and targets and select hierarchically
    '''
    def __init__(self, ltd, lsd, num_basis:int, num_hidden: Iterable[int], activation: str, dtype: torch.dtype = torch.float32):
        super().__init__()

        self._ltd = ltd
        self._lsd = lsd
        self._num_basis = num_basis
        self._dtype = dtype
        layers = []
        prev_dim = self._lsd + self._ltd
        # prev_dim = self._lsd
        for n in num_hidden:
            layers.append(nn.Linear(prev_dim, n))
            layers.append(getattr(nn, activation)())
            prev_dim = n
        layers.append(nn.Linear(prev_dim, self._num_basis))
        layers.append(nn.Softmax(dim=-1))
        self._contextual_coeff = nn.Sequential(*layers).to(dtype=self._dtype)




    def forward(self, state: torch.Tensor, latent_context:torch.Tensor):
        x = self._contextual_coeff(torch.cat((state,latent_context),dim=-1))

        return x

class ContexualTransitionMatrix(nn.Module):
    '''
    Implements nn Module for coefficient net that is both state and context dependent
    TODO: Make ltd and lsd to be of similar dimension
    TODO: Make separate coefficient net for contexts and targets and select hierarchically
    '''
    def __init__(self, ltd, lsd, num_basis:int, num_hidden: Iterable[int], activation: str, dtype: torch.dtype = torch.float32):
        super().__init__()

        self._ltd = ltd
        self._lsd = lsd
        self._num_basis = num_basis
        self._dtype = dtype
        layers = []
        prev_dim = self._ltd + self._ltd
        # prev_dim = self._lsd
        for n in num_hidden:
            layers.append(nn.Linear(prev_dim, n))
            layers.append(getattr(nn, activation)())
            prev_dim = n
        layers.append(nn.Linear(prev_dim, self._lsd*self._lsd))
        self._contextual_transMatrix = nn.Sequential(*layers).to(dtype=self._dtype)




    def forward(self, state: torch.Tensor, latent_context:torch.Tensor):
        x = self._contextual_transMatrix(torch.cat((state,latent_context),dim=-1))

        return x

class TransitionMatrix(nn.Module):
    '''
    Implements nn Module for coefficient net that is both state and context dependent
    TODO: Make ltd and lsd to be of similar dimension
    TODO: Make separate coefficient net for contexts and targets and select hierarchically
    '''
    def __init__(self, lsd, num_basis:int, num_hidden: Iterable[int], activation: str, dtype: torch.dtype = torch.float32):
        super().__init__()

        self._lsd = lsd
        self._num_basis = num_basis
        self._dtype = dtype
        layers = []
        prev_dim = self._lsd + self._ltd
        # prev_dim = self._lsd
        for n in num_hidden:
            layers.append(nn.Linear(prev_dim, n))
            layers.append(getattr(nn, activation)())
            prev_dim = n
        layers.append(nn.Linear(prev_dim, self._lsd*self._lsd))
        with torch.no_grad():
            layers[-1].weight.copy(torch.flip(0.2*torch.eye(self._lsd, dtype=self._dtype),[0,1]).reshape(self._lsd*self._lsd))
        self._transMatrix = nn.Sequential(*layers).to(dtype=self._dtype)




    def forward(self, state: torch.Tensor):
        x = self._transMatrix(torch.cat((state),dim=-1))

        return x

class AcPredict(nn.Module):

    @staticmethod
    def get_default_config() -> ConfigDict:
        config = ConfigDict(
            num_basis=15,
            bandwidth=3,
            trans_net_hidden_units=[],
            control_net_hidden_units=[15],
            process_noise_hidden_units=[15],
            trans_net_hidden_activation="Tanh",
            control_net_hidden_activation='ReLU',
            process_noise_hidden_activation='ReLU',
            learn_trans_covar=True,
            context_flag_coeff=True,
            context_flag_control=True,
            context_flag_noise=True,
            context_var_flag_coeff=True,
            context_var_flag_control=True,
            context_var_flag_noise=True,
            hyper_transition_matrix=True,
            multi_gaussian_l_transform=True,
            trans_covar=1,
            learn_initial_state_covar=True,
            initial_state_covar=1,
            learning_rate=7e-3,
            enc_out_norm='post',
            clip_gradients=True,
            never_invalid=True
        )
        config.finalize_adding()
        return config

    def __init__(self, ltd: int, latent_obs_dim: int, act_dim: int, config: ConfigDict = None, dtype: torch.dtype = torch.float32):
        """
        RKN Cell (mostly) as described in the original RKN paper
        :param latent_obs_dim: latent observation dimension
        :param config: config dict object, for configuring the cell
        :param dtype: datatype
        """
        super(AcPredict, self).__init__()
        self._ltd = ltd
        self._lod = latent_obs_dim
        self._lsd = self._lod
        self._lad = act_dim
        #self._device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda_if_available else "cpu")

        if config == None:

            self.c = self.get_default_config()
        else:
           self.c = config

        self._dtype = dtype


        if self.c.hyper_transition_matrix:
            self._learn_transition_model()
        else:
            self._build_transition_model()
        self._build_task_model()
        # self._control_net = self._build_control_net(self.c.control_net_hidden_units,
        #                                             self.c.control_net_hidden_activation)
        if self.c.context_mu_flag_control and self.c.context_var_flag_control:
            self._control_net = ContexualControl(self._ltd, self._lad, self._lsd, self.c.control_net_hidden_units,
                                                                             self.c.control_net_hidden_activation)
        elif self.c.context_mu_flag_control or self.c.context_var_flag_control:
            self._control_net = ContexualControl(int(self._ltd/2), self._lad, self._lsd, self.c.control_net_hidden_units,
                                                                             self.c.control_net_hidden_activation)
        else:
            self._control_net = Control(self._lad, self._lsd, self.c.control_net_hidden_units,
                                                 self.c.control_net_hidden_activation)

        if self.c.context_mu_flag_coeff and self.c.context_var_flag_coeff:
            self._coefficient_net = ContexualCoefficient(self._ltd,self._lsd, self.c.num_basis, self.c.trans_net_hidden_units,
                                                            self.c.trans_net_hidden_activation)
        elif self.c.context_mu_flag_coeff or self.c.context_var_flag_coeff:
            self._coefficient_net = ContexualCoefficient(int(self._ltd/2),self._lsd, self.c.num_basis, self.c.trans_net_hidden_units,
                                                           self.c.trans_net_hidden_activation)
        else:
            self._coefficient_net = Coefficient(self._lsd, self.c.num_basis,
                                                         self.c.trans_net_hidden_units,
                                                         self.c.trans_net_hidden_activation)

        # TODO: This is currently a different noise for each dim, not like in original paper (and acrkn)
        if self.c.context_mu_flag_noise and self.c.context_var_flag_noise:
            self._log_process_noise = ContexualProcessNoise(self._ltd, self._lsd, self.c.process_noise_hidden_units,
                                                                self.c.process_noise_hidden_activation)
        elif self.c.context_mu_flag_noise or self.c.context_var_flag_noise:
            self._log_process_noise = ContexualProcessNoise(int(self._ltd/2), self._lsd, self.c.process_noise_hidden_units,
                                                            self.c.process_noise_hidden_activation)
        else:
            self._log_process_noise = ProcessNoise(self._lsd, self.c.trans_covar, self.c.process_noise_hidden_units,
                                                   self.c.process_noise_hidden_activation)


    @property
    def _device(self):
        return self._tm_11_full.device

    #   @torch.jit.script_method
    def forward(self, post_mean: torch.Tensor, post_cov: Iterable[torch.Tensor], action: torch.Tensor, latent_context: torch.Tensor) -> \
            Tuple[torch.Tensor, Iterable[torch.Tensor]]:
        """
        forward pass trough the cell. For proper recurrent model feed back outputs 3 and 4 (next prior belief at next
        time step

        :param post_mean: prior mean at time t
        :param post_cov: prior covariance at time t
        :param action: action at time t
        :return: prior mean at time t + 1, prior covariance time t + 1

        """

        next_prior_mean, next_prior_cov = self._contextual_predict(post_mean, post_cov, action, latent_context)

        return next_prior_mean, next_prior_cov

    def _learn_transition_model(self) -> None:
        """
        Builds the basis functions for transition model and the noise
        :return:
        """
        if self.c.context_mu_flag_coeff and self.c.context_var_flag_coeff:
            self._tm = ContexualTransitionMatrix(self._ltd,self._lsd, self.c.num_basis, self.c.trans_net_hidden_units,
                                                            self.c.trans_net_hidden_activation)
        elif self.c.context_mu_flag_coeff or self.c.context_var_flag_coeff:
            self._tm = ContexualTransitionMatrix(int(self._ltd/2),self._lsd, self.c.num_basis, self.c.trans_net_hidden_units,
                                                           self.c.trans_net_hidden_activation)
        else:
            self._tm =TransitionMatrix(self._lsd, self.c.num_basis,
                                                         self.c.trans_net_hidden_units,
                                                         self.c.trans_net_hidden_activation)


    def get_learnt_transition_model(self, post_mean: torch.Tensor, latent_context: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Compute the locally-linear transition model given the current posterior mean
        :param post_mean: current posterior mean
        :return: transition matrices (4 Blocks), transition covariance (vector of size lsd)
        """
        # prepare transition model
        # latent_context = latent_context[:, None, :] ### add n_ls later
        # latent_context = latent_context.expand(latent_context.shape[0], post_mean.shape[1], self._ltd) ### add n_ls later
        # with tsensor.explain():
        #     post_mean
        # with tsensor.explain():
        #     latent_context

        if self.c.context_mu_flag_coeff and self.c.context_var_flag_coeff:
            tm_full = torch.reshape(self._tm(latent_context,latent_context), [-1, self._lsd, self._lsd])
        elif self.c.context_mu_flag_coeff:
            tm_full = torch.reshape(self._tm(latent_context[:,:int(self._ltd/2)],latent_context[:,:int(self._ltd/2)]), [-1,  self._lsd, self._lsd])
        elif self.c.context_var_flag_coeff:
            tm_full = torch.reshape(self._tm(latent_context[:,:int(self._ltd/2)], latent_context[:,int(self._ltd/2):]), [-1, self._lsd, self._lsd])
        else:
            tm_full = torch.reshape(self._tm(post_mean), [-1, self._lsd, self._lsd])
        print(tm_full)

        # coefficients = torch.reshape(self._coefficient_net(latent_context),
        #                              [-1, self.c.num_basis, 1, 1])

        tm_full += torch.eye(self._lsd)[None,:,:].cuda()

        if self.c.context_mu_flag_noise and self.c.context_var_flag_noise:
            process_cov = elup1(self._log_process_noise(latent_context))
        elif self.c.context_mu_flag_noise:
            process_cov = elup1(self._log_process_noise(latent_context[:,:int(self._ltd/2)]))
        elif self.c.context_var_flag_noise:
            process_cov = elup1(self._log_process_noise(latent_context[:,int(self._ltd/2):]))
        else:
            process_cov = elup1(self._log_process_noise())


        return tm_full, process_cov

    def _build_transition_model(self) -> None:
        """
        Builds the basis functions for transition model and the noise
        :return:
        """
        self._transition_matrices = nn.Parameter(0.8*torch.ones(self._lsd, dtype=self._dtype)[None, :].repeat(self.c.num_basis, 1))
        #self._transition_matrices = nn.Parameter(torch.rand((self._lsd, self._lsd), dtype=self._dtype)[None, :, :].repeat(self.c.num_basis, 1, 1))

    def _build_task_model(self) -> None:
        """
        Builds the basis functions for transition model and the noise
        :return:
        """
        self._task_matrix = nn.Parameter(0.9*torch.ones(self._lsd, dtype=self._dtype))
        #self._transition_matrices = nn.Parameter(torch.rand((self._lsd, self._lsd), dtype=self._dtype)[None, :, :].repeat(self.c.num_basis, 1, 1))

    def get_transition_model(self, post_mean: torch.Tensor, latent_context: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Compute the locally-linear transition model given the current posterior mean
        :param post_mean: current posterior mean
        :return: transition matrices (4 Blocks), transition covariance (vector of size lsd)
        """
        # prepare transition model
        # latent_context = latent_context[:, None, :] ### add n_ls later
        # latent_context = latent_context.expand(latent_context.shape[0], post_mean.shape[1], self._ltd) ### add n_ls later
        # with tsensor.explain():
        #     post_mean
        # with tsensor.explain():
        #     latent_context

        if self.c.context_mu_flag_coeff and self.c.context_var_flag_coeff:
            coefficients = torch.reshape(self._coefficient_net(post_mean,latent_context), [-1, self.c.num_basis, 1])
        elif self.c.context_mu_flag_coeff:
            coefficients = torch.reshape(self._coefficient_net(post_mean,latent_context[:,:int(self._ltd/2)]), [-1, self.c.num_basis, 1])
        elif self.c.context_var_flag_coeff:
            coefficients = torch.reshape(self._coefficient_net(post_mean, latent_context[:,int(self._ltd/2):]), [-1, self.c.num_basis, 1])
        else:
            coefficients = torch.reshape(self._coefficient_net(post_mean), [-1, self.c.num_basis, 1])
        # coefficients = torch.reshape(self._coefficient_net(latent_context),
        #                              [-1, self.c.num_basis, 1, 1])

        tm_full =  self._transition_matrices[None,: ,:]

        lm_full = self._task_matrix

        tm_full = (coefficients * tm_full).sum(dim=1)
        #tm_full += torch.eye(self._lsd)[None,:,:].cuda()

        if self.c.context_mu_flag_noise and self.c.context_var_flag_noise:
            process_cov = elup1(self._log_process_noise(latent_context))
        elif self.c.context_mu_flag_noise:
            process_cov = elup1(self._log_process_noise(latent_context[:,:int(self._ltd/2)]))
        elif self.c.context_var_flag_noise:
            process_cov = elup1(self._log_process_noise(latent_context[:,int(self._ltd/2):]))
        else:
            process_cov = elup1(self._log_process_noise())


        return tm_full, lm_full, process_cov


    def _contextual_predict(self, post_mean: torch.Tensor, post_covar: List[torch.Tensor], action: torch.Tensor, latent_context: torch.Tensor) \
            -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """ Performs prediction step
        :param post_mean: last posterior mean
        :param post_covar: last posterior covariance
        :param action: action
        :param latent_context: task specific context
        :return: current prior state mean and covariance
        """
        # compute state dependent transition matrix
        mu_context = latent_context[:, :int(self._ltd / 2)]
        var_context = latent_context[:, int(self._ltd / 2):]
        if self.c.hyper_transition_matrix:
            tm, process_covar = self.get_learnt_transition_model(post_mean, latent_context)
        else:
            tm, lm, process_covar = self.get_transition_model(post_mean, latent_context)
        if self.c.context_mu_flag_control and self.c.context_var_flag_control:
            control_factor = self._control_net(action, latent_context)
        elif self.c.context_mu_flag_control:
            control_factor = self._control_net(action, mu_context)
        elif self.c.context_var_flag_control:
            control_factor = self._control_net(action, var_context)
        else:
            control_factor = self._control_net(action)
            #control_factor = self.det
        #control_factor = self._control_net(torch.cat((action,latent_context),dim=-1))


        # predict next prior mean

        if self.c.multi_gaussian_l_transform:
            mu_prior, var_prior = gaussian_multi_linear_transform(tm, lm, post_mean, post_covar, mu_context, var_context,
                                                                  control_factor, process_covar)
        else:
            mu_prior, var_prior = gaussian_linear_transform(tm, post_mean, post_covar, control_factor, process_covar)



        return mu_prior, var_prior



    ##### For Factorized Ignore ##############
    def _bandToFull(self,cov_list):
        '''
        :param cov_list:
        :return:
        '''
        pass

    def _fullToBand(self,mat):
        '''
        :param mat:
        :return:
        '''
        pass


    def _linear_transform(self, mean, cov, matrix):
        return matrix*mean, matrix*cov*matrix.transpose()


    def _contextual_predict_factorized(self, post_mean: torch.Tensor, post_covar: List[torch.Tensor], action: torch.Tensor, latent_context: torch.Tensor) \
            -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """ Performs prediction step
        :param post_mean: last posterior mean
        :param post_covar: last posterior covariance
        :param action: action
        :param latent_context: task specific context
        :return: current prior state mean and covariance
        """
        # compute state dependent transition matrix
        coefficients = torch.reshape(self._coefficient_net(post_mean), [-1, self.c.num_basis, 1, 1])
        coeff_factor = self.W_coeff*coefficients
        coeff_transform_matrix = coeff_factor*torch.eye(self._lod, dtype=self._dtype)
        mean_factor_trans, cov_factor_trans = self._linear_transform(post_mean,post_covar,self.W_enc)
        mean_factor_multiplexed, cov_factor_multiplexed= self._linear_transform(mean_factor_trans, cov_factor_trans, coeff_transform_matrix)
        prior_mean, prior_cov = self._linear_transform(mean_factor_multiplexed,cov_factor_multiplexed,self.W_enc)

        control_factor = self._control_net(action)
        print("contextual_prediction.py line 660")
        print("control_factor:",control_factor)
        print("type(control_factor)",type(control_factor))

        return prior_mean + control_factor, prior_cov

    def _build_transition_model_factorized(self) -> None:
        """
        Builds the basis functions for transition model and the noise
        :return:
        """
        # build state independent basis matrices
        np_mask = np.ones([self._lod, self._lod], dtype=np.float32)
        np_mask = np.triu(np_mask, -self.c.bandwidth) * np.tril(np_mask, self.c.bandwidth)
        self._band_mask = torch.from_numpy(np.expand_dims(np_mask, 0))

        self.W_enc = nn.Parameter(torch.eye(self._lod, dtye=self._dtype))
        self.W_coeff = nn.Parameter(torch.eye(self.c.num_basis, dtye=self._dtype))

        self.W_dec = nn.Parameter(torch.eye(self._lod, dtye=self._dtype))

        self._coefficient_net = self._build_coefficient_net(self.c.trans_net_hidden_units,
                                                            self.c.trans_net_hidden_activation)

        init_trans_cov = elup1_inv(self.c.trans_covar)
        # TODO: This is currently a different noise for each dim, not like in original paper (and acrkn)
        self._log_transition_noise = \
            nn.Parameter(nn.init.constant_(torch.empty(1, self._lsd, dtype=self._dtype), init_trans_cov))

