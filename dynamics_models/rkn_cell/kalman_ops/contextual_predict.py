import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from utils.ConfigDict import ConfigDict
from typing import Iterable, Tuple, List

nn = torch.nn


def bmv(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Batched Matrix Vector Product"""
    return torch.bmm(mat, vec[..., None])[..., 0]

def gaussian_linear_transform(tm: List, mean: torch.Tensor, covar:torch.Tensor, control_factor:torch.Tensor
                              , process_covar: torch.Tensor):

    #print("saleh_added contextual_predic.py line 17") #
    control_factor = torch.zeros_like(control_factor)#saleh_added
    # predict next prior mean
    obs_dim = int(mean.shape[-1]/2)
    mu = mean[:, :obs_dim]
    ml = mean[:, obs_dim:]
    #[tm11, tm12, tm21, tm22] = [t.repeat((mu.shape[0], 1, 1)) for t in tm]
    [tm11, tm12, tm21, tm22] = [t for t in tm]

    #nmu = bmv(tm11, mu) + bmv(tm12, ml)
    #nml = bmv(tm21, mu) + bmv(tm22, ml)


    nmu = torch.matmul(tm11, mu.T).T + torch.matmul(tm12, ml.T).T
    nml = torch.matmul(tm21, mu.T).T + torch.matmul(tm22, ml.T).T

    mu_prior = torch.cat([nmu, nml], dim=-1) #+ control_factor ## remove + tem

    # predict next prior covariance (eq 10 - 12 in paper supplement)
    cu, cl, cs = covar
    cov_prior = cov_linear_transform(tm, [cu, cl, cs], process_covar)
    return mu_prior, cov_prior

def gaussian_locally_linear_transform(tm: List, mean: torch.Tensor, covar:torch.Tensor, control_factor:torch.Tensor
                              , process_covar: torch.Tensor):

    #print("saleh_added contextual_predic.py line 44") #
    control_factor = torch.zeros_like(control_factor)#saleh_added
    [tm11, tm12, tm21, tm22] = tm

    # predict next prior mean
    obs_dim = int(mean.shape[-1]/2)
    mu = mean[:, :obs_dim]
    ml = mean[:, obs_dim:]

    nmu = bmv(tm11, mu) + bmv(tm12, ml)
    nml = bmv(tm21, mu) + bmv(tm22, ml)

    mu_prior = torch.cat([nmu, nml], dim=-1) #+ control_factor

    # predict next prior covariance (eq 10 - 12 in paper supplement)
    cu, cl, cs = covar
    cov_prior = cov_locally_linear_transform(tm, [cu, cl, cs], process_covar)
    return mu_prior, cov_prior

def gaussian_linear_transform_task(lm: torch.Tensor, mu_l: torch.Tensor, covar_l:torch.Tensor):
    lm_batched = lm.repeat((mu_l.shape[0],1, 1))
    mu_prior = bmv(lm_batched, mu_l)

    cov_prior = cov_linear_transform_task(lm_batched, covar_l)
    return mu_prior, cov_prior

def gaussian_linear_transform_task_factorized(lm: torch.Tensor, mu_l: torch.Tensor, covar_l:torch.Tensor):
    lm_batched = lm.repeat((mu_l.shape[0],1))
    mu_prior = lm_batched*mu_l

    cov_prior = cov_linear_transform_task_factorized(lm_batched, covar_l)
    return mu_prior, cov_prior

def gaussian_l_linear_transform_task_factorized(lm: torch.Tensor, mu_l: torch.Tensor, covar_l:torch.Tensor):
    mu_prior = lm*mu_l

    cov_prior = cov_linear_transform_task_factorized(lm, covar_l)
    return mu_prior, cov_prior

def gaussian_non_linear_transform_task(lm: torch.Tensor, mu_l: torch.Tensor, covar_l:torch.Tensor):
    #lm_batched = lm.repeat((mu_l.shape[0],1, 1))
    mu_prior = bmv(lm, mu_l)

    cov_prior = cov_linear_transform_task(lm, covar_l)
    return mu_prior, cov_prior

def cov_linear_transform_task(tm: torch.Tensor, covar: torch.Tensor):
    # predict next prior covariance (eq 10 - 12 in paper supplement)
    tm11 = tm[:, :covar.shape[-1], :]
    tm21 = tm[:, covar.shape[-1]:, :]

    ncu = dadat(tm11, covar)
    ncl = dadat(tm21, covar)
    ncs = dadbt(tm21, covar, tm11)
    return [ncu, ncl, ncs]

def cov_linear_transform_task_factorized(lm: torch.Tensor, covar: torch.Tensor):

    # predict next prior covariance (eq 10 - 12 in paper supplement)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    covar_new = lm**2 * covar

    lod = int(covar.shape[-1] / 2)

    ncu = covar_new[:, :lod]
    ncl = covar_new[:, lod:]
    ncs = torch.zeros(ncu.shape[0], ncu.shape[1]).to(device)
    return [ncu, ncl, ncs]




def cov_linear_transform(tm: List, covar: List, process_covar):

    # predict next prior covariance (eq 10 - 12 in paper supplement)
    cu, cl, cs = covar
    [tm11, tm12, tm21, tm22] = [t for t in tm]

    # prepare process noise
    trans_covar_upper = process_covar[..., :cu.shape[-1]]
    trans_covar_lower = process_covar[..., cu.shape[-1]:]

    ncu = torch.matmul(tm11**2, cu.T).T + 2.0 * torch.matmul(tm11 * tm12, cs.T).T + torch.matmul(tm12**2,
                                                                                              cl.T).T + trans_covar_upper
    ncl = torch.matmul(tm21**2, cu.T).T + 2.0 * torch.matmul(tm21 * tm22, cs.T).T + torch.matmul(tm22**2,
                                                                                              cl.T).T + trans_covar_lower
    ncs = torch.matmul(tm21 * tm11, cu.T).T + torch.matmul(tm22 * tm11, cs.T).T + torch.matmul(tm21 * tm12,
                                                                                               cs.T).T + torch.matmul(
        tm22 * tm12, cl.T).T

    return [ncu,ncl,ncs]

def cov_locally_linear_transform(tm: List, covar: List, process_covar):

    # predict next prior covariance (eq 10 - 12 in paper supplement)
    cu, cl, cs = covar
    [tm11, tm12, tm21, tm22] = tm

    # prepare process noise
    trans_covar_upper = process_covar[..., :cu.shape[-1]]
    trans_covar_lower = process_covar[..., cu.shape[-1]:]

    ncu = dadat(tm11, cu) + 2.0 * dadbt(tm11, cs, tm12) + dadat(tm12, cl) + trans_covar_upper
    ncl = dadat(tm21, cu) + 2.0 * dadbt(tm21, cs, tm22) + dadat(tm22, cl) + trans_covar_lower
    ncs = dadbt(tm21, cu, tm11) + dadbt(tm22, cs, tm11) + dadbt(tm21, cs, tm12) + dadbt(tm22, cl, tm12)
    return [ncu,ncl,ncs]


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
        x = self._control(action) # it is not! zero
        #print("self._control(action)",self._control(action))
        #print("self._control(action) ","line 210 contextual_predict.py")


        return x

class TaskNetMu(nn.Module):
    def __init__(self, ltd, lsd, num_hidden: Iterable[int], activation: str, dtype: torch.dtype = torch.float32):
        super().__init__()

        self._ltd = int(ltd/2)
        self._lsd = lsd
        self._dtype = dtype
        layers = []
        prev_dim = self._ltd

        for n in num_hidden:
            layers.append(nn.Linear(prev_dim, n))
            layers.append(getattr(nn, activation)())
            prev_dim = n
        layers.append(nn.Linear(prev_dim, self._lsd))
        self._mu_task = nn.Sequential(*layers).to(dtype=self._dtype)

    def forward(self, latent_task_mu: torch.Tensor):
        x = self._mu_task(latent_task_mu)

        return x


class TaskNetVar(nn.Module):
    def __init__(self, ltd, lsd, num_hidden: Iterable[int], diagonal, activation: str, dtype: torch.dtype = torch.float32):
        super().__init__()

        self._ltd = int(ltd/2)
        self._lod = int(lsd/2)
        self._diagonal = diagonal
        self._device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")

        if self._diagonal:
            self._lvd = 2 * self._lod
        else:
            self._lvd = 3 * self._lod
        self._dtype = dtype
        layers = []
        prev_dim = self._ltd

        for n in num_hidden:
            layers.append(nn.Linear(prev_dim, n))
            layers.append(getattr(nn, activation)())
            prev_dim = n
        layers.append(nn.Linear(prev_dim, self._lvd))
        layers.append(nn.Softplus())
        self._var_task = nn.Sequential(*layers).to(dtype=self._dtype)

    def forward(self, latent_task_var: torch.Tensor):
        x = self._var_task(latent_task_var)
        x = torch.add(x, 0.0001)
        if self._diagonal:
            return [x[:,:self._lod],x[:,self._lod:],torch.zeros(x.shape[0],self._lod).to(self._device)]
        else:
            return [x[:, :self._lod], x[:, self._lod:2 * self._lod], x[:, 2 * self._lod:]]



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
        #print('line 313 contextual_predict.py')
        #print('state:',x)

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
            #print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',layers[-1].weight)
        self._transMatrix = nn.Sequential(*layers).to(dtype=self._dtype)




    def forward(self, state: torch.Tensor):
        x = self._transMatrix(torch.cat((state),dim=-1))

        return x

class AcPredict(nn.Module):

    @staticmethod
    def get_default_config() -> OmegaConf:
        s = """
                num_basis: 15
                bandwidth: 3
                enc_net_hidden_units: 
                    - 120
                dec_net_hidden_units: 
                    - 120
                trans_net_hidden_units:
                    - 120
                control_net_hidden_units:
                    - 120
                    - 120
                    - 120
                process_noise_hidden_units:
                    - 30
                trans_net_hidden_activation: "Tanh"
                control_net_hidden_activation: 'ReLU'
                process_noise_hidden_activation: 'ReLU'
                learn_trans_covar: False
                trans_covar: 0.1
                learn_initial_state_covar: True
                initial_state_covar: 10
                learning_rate: 7e-3
                enc_out_norm: 'post'
                clip_gradients: True
                never_invalid: False
                """

        return OmegaConf.create(s)

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
        self._lsd = 2 * self._lod
        self._lad = act_dim

        if config == None:

            self.c = self.get_default_config()
        else:
           self.c = config

        self._dtype = dtype



        self._build_transition_model()
        self._build_task_model()

        self._task_net_mu = TaskNetMu(self._ltd, self._lsd, self.c.task_net_hidden_units,self.c.task_net_hidden_activation)
        self._task_net_var = TaskNetVar(self._ltd, self._lsd, self.c.task_net_hidden_units, self.c.nl_diagonal,
                                      self.c.task_net_hidden_activation)


        self._control_net = Control(self._lad, self._lsd, self.c.control_net_hidden_units,
                                                 self.c.control_net_hidden_activation)


        self._coefficient_net = Coefficient(self._lsd, self.c.num_basis,
                                                         self.c.trans_net_hidden_units,
                                                         self.c.trans_net_hidden_activation)


        # TODO: This is currently a different noise for each dim, not like in original paper (and acrkn)

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

        #print('contexual_pred.py line 450: ')
        #print("action",action) #all 0 checked
        next_prior_mean, next_prior_cov = self._contextual_predict(post_mean, post_cov, action, latent_context)

        return next_prior_mean, next_prior_cov

    def _build_transition_model(self) -> None:
        """
        Builds the basis functions for transition model and the nosie
        :return:
        """
        # build state independent basis matrices
        np_mask = np.ones([self._lod, self._lod], dtype=np.float32)
        np_mask = np.triu(np_mask,  -self.c.bandwidth) * np.tril(np_mask, self.c.bandwidth)
        # These are constant matrices (not learning) used by the model - no gradients required

        if not self.c.kalman_linear:
            self._band_mask = nn.Parameter(torch.from_numpy(np.expand_dims(np_mask, 0)), requires_grad=False)
            self._eye_matrix = nn.Parameter(torch.eye(self._lod)[None, :, :], requires_grad=False)

            self._tm_11_full = nn.Parameter(torch.zeros(self.c.num_basis, self._lod, self._lod, dtype=self._dtype))
            self._tm_12_full = \
                nn.Parameter(0.2 * torch.eye(self._lod, dtype=self._dtype)[None, :, :].repeat(self.c.num_basis, 1, 1))
            self._tm_21_full =\
                nn.Parameter(-0.2 * torch.eye(self._lod, dtype=self._dtype)[None, :, :].repeat(self.c.num_basis, 1, 1))
            self._tm_22_full = nn.Parameter(torch.zeros(self.c.num_basis, self._lod, self._lod, dtype=self._dtype))
        else:
            self._band_mask = nn.Parameter(torch.from_numpy(np_mask), requires_grad=False)
            self._eye_matrix = nn.Parameter(torch.eye(self._lod), requires_grad=False)

            self._tm_11_full = nn.Parameter(torch.zeros(self._lod, self._lod, dtype=self._dtype))
            self._tm_12_full = \
                nn.Parameter(0.2 * torch.eye(self._lod, dtype=self._dtype))
            self._tm_21_full = \
                nn.Parameter(-0.2 * torch.eye(self._lod, dtype=self._dtype))
            self._tm_22_full = nn.Parameter(torch.zeros(self._lod, self._lod, dtype=self._dtype))

        self._transition_matrices_raw = [self._tm_11_full, self._tm_12_full, self._tm_21_full, self._tm_22_full]

    def _build_task_model(self) -> None:
        """
        Builds the basis functions for transition model and the noise
        :return:
        """
        if self.c.additive_linear_task:
            self._task_matrix = nn.Parameter(torch.rand((self._lsd, self._lod), dtype=self._dtype)[None, :])
        if self.c.additive_linear_task_factorized:
            self._task_matrix = nn.Parameter(torch.rand((self._lsd), dtype=self._dtype)[None, :])
        if self.c.additive_l_linear_task_factorized:
            self._task_matrix = nn.Parameter(torch.rand((self.c.num_basis, self._lsd), dtype=self._dtype)[None, :])

    def get_transition_model(self, post_mean: torch.Tensor, latent_context: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Compute the locally-linear transition model given the current posterior mean
        :param post_mean: current posterior mean
        :return: transition matrices (4 Blocks), transition covariance (vector of size lsd)
        """

        if not self.c.kalman_linear:


            coefficients = torch.reshape(self._coefficient_net(post_mean), [-1, self.c.num_basis, 1, 1])

            tm11_full, tm12_full, tm21_full, tm22_full = [x[None, :, :, :] for x in self._transition_matrices_raw]

            tm11_full = (coefficients * tm11_full).sum(dim=1)
            tm11 = tm11_full * self._band_mask
            tm11 += self._eye_matrix

            tm12_full = (coefficients * tm12_full).sum(dim=1)
            tm12 = tm12_full * self._band_mask

            tm21_full = (coefficients * tm21_full).sum(dim=1)
            tm21 = tm21_full * self._band_mask

            tm22_full = (coefficients * tm22_full).sum(dim=1)
            tm22 = tm22_full * self._band_mask
            tm22 += self._eye_matrix

        else:
            tm11_full, tm12_full, tm21_full, tm22_full = self._transition_matrices_raw
            tm11 = tm11_full * self._band_mask
            tm11 += self._eye_matrix

            tm12 = tm12_full * self._band_mask

            tm21 = tm21_full * self._band_mask

            tm22 = tm22_full * self._band_mask
            tm22 += self._eye_matrix


        if self.c.additive_linear_task or self.c.additive_linear_task_factorized:
            lm_full = self._task_matrix
        elif self.c.additive_l_linear_task_factorized:
            lm_full = self._task_matrix
            lm_full = (coefficients_task * lm_full).sum(dim=1)
        else:
            lm_full = None


        process_cov = elup1(self._log_process_noise())


        return [tm11, tm12, tm21, tm22], lm_full, process_cov



    def _contextual_predict(self, post_mean: torch.Tensor, post_covar: List[torch.Tensor], action: torch.Tensor, latent_context: torch.Tensor) \
            -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """ Performs prediction step
        :param post_mean: last posterior mean
        :param post_covar: last posterior covariance
        :param action: action
        :param latent_context: task specific context
        :return: current prior state mean and covariance
        """

        #print('contextual_predict.py line 568 :')
        #print("action",action) #checked all 0
        action=torch.zeros_like(action)

        mu_context = latent_context[:, :int(self._ltd / 2)]
        var_context = latent_context[:, int(self._ltd / 2):]

        # compute state dependent transition matrix

        [tm11, tm12, tm21, tm22], lm, process_covar = self.get_transition_model(post_mean, latent_context)
        #print('contextual_predict.py line 576 to 580:')

        #print("self._control_net(action)",' contextual_predict.py line 576 to 580:')
        control_factor = self._control_net(action) ## !should be removed
        #print("control_factor", control_factor)


        # predict next prior mean
        #print("contextual_predict.py line 581")
        #print("control_factor",control_factor) #its not 0 here
        if self.c.kalman_linear:
            mu_prior_0, var_prior_0 = gaussian_linear_transform([tm11, tm12, tm21, tm22], post_mean, post_covar, control_factor,
                                                        process_covar)
        else:
            mu_prior_0, var_prior_0 = gaussian_locally_linear_transform([tm11, tm12, tm21, tm22], post_mean, post_covar,
                                                                control_factor,
                                                                process_covar)


        if self.c.additive_nl_task:
            mu_prior_1 = self._task_net_mu(mu_context)
            var_prior_1 = self._task_net_var(var_context)
            mu_prior = mu_prior_0 + mu_prior_1
            var_prior = [x+y for x,y in zip(var_prior_0,var_prior_1)]
        elif self.c.additive_nl_task_deterministic:
            mu_prior_1 = self._task_net_mu(mu_context)
            mu_prior = mu_prior_0 + mu_prior_1
            var_prior = var_prior_0
        elif self.c.additive_linear_task:
            mu_prior_1, var_prior_1 = gaussian_linear_transform_task(lm, mu_context, var_context)
            mu_prior = mu_prior_0 + mu_prior_1
            var_prior = [x + y for x, y in zip(var_prior_0, var_prior_1)]
        elif self.c.additive_linear_task_factorized:
            mu_prior_1, var_prior_1 = gaussian_linear_transform_task_factorized(lm, mu_context, var_context)
            mu_prior = mu_prior_0 + mu_prior_1
            var_prior = [x + y for x, y in zip(var_prior_0, var_prior_1)]
        elif self.c.additive_l_linear_task_factorized:
            mu_prior_1, var_prior_1 = gaussian_l_linear_transform_task_factorized(lm, mu_context, var_context)
            mu_prior = mu_prior_0 + mu_prior_1
            var_prior = [x + y for x, y in zip(var_prior_0, var_prior_1)]

        else:
            mu_prior = mu_prior_0
            var_prior = var_prior_0


        return mu_prior, var_prior



    def _linear_transform(self, mean, cov, matrix):
        return matrix*mean, matrix*cov*matrix.transpose()






