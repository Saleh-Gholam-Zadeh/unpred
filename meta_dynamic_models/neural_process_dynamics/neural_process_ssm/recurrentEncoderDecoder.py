import sys
import sys

sys.path.append('.')

import torch
import numpy as np
from utils.TimeDistributed import TimeDistributed
from dynamics_models.rkn.acrkn.Encoder import Encoder
from dynamics_models.rkn.acrkn.Decoder import SplitDiagGaussianDecoder, SimpleDecoder
from dynamics_models.rkn.acrkn.acRKNLayer import acRKNLayer
from dynamics_models.rkn.acrkn.acRKNContextLayer import AcRKNContextLayer
from omegaconf import OmegaConf
from typing import Tuple

optim = torch.optim
nn = torch.nn


def elup1_inv(x: torch.Tensor) -> torch.Tensor:
    """
    inverse of elu+1, numpy only, for initialization
    :param x: input
    :return:
    """
    return np.log(x) if x < 1.0 else (x - 1.0)


def elup1(x: torch.Tensor) -> torch.Tensor:
    """
    elu + 1 activation faction to ensure positive covariances
    :param x: input
    :return: exp(x) if x < 0 else x + 1
    """
    return torch.exp(x).where(x < 0.0, x + 1.0)


def tanh2(x, min_y, max_y):
    scale_x = 1 / ((max_y - min_y) / 2)
    return (max_y - min_y) / 2 * (torch.tanh(x * scale_x) + 1.0) + min_y


class acrknContextGen(nn.Module): #not used anywhere
    @staticmethod
    def get_default_config() -> OmegaConf:
        s = """
        num_basis:15
        bandwidth:3
        trans_net_hidden_units: []
        control_net_hidden_units:
            - 120
        trans_net_hidden_activation: "Tanh",
        control_net_hidden_activation: 'ReLU'
        learn_trans_covar: True
        trans_covar: 0.1
        learn_initial_state_covar: True
        initial_state_covar: 10
        learning_rate: 7e-3
        enc_out_norm: 'post'
        clip_gradients: True
        never_invalid: True
        """
        return OmegaConf.create(s)

    def __init__(self, target_dim: int, lod: int, lad: int, config: OmegaConf = None,
                 use_cuda_if_available: bool = True):
        super(acrknContextGen, self).__init__()
        print("saleh_add: line 68 recurrentEncoderDecoder.py ContextGen ")      #inja nemiad asan
        """
               :param target_dim:
               :param lod:
               :param config:
               :param use_cuda_if_available:
               """
        # inja nemiad asan // in class estefade nemishe kollan
        self._device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        print("line 77 recurrentEncoderDecoder")
        self._inp_dim =target_dim # target_dim #here we should change. input and output are not anymore of the same dimensions
        self._lad = lad
        self._lod = lod
        print("saleh_add: line 81 recurrentEncoderDecoder.py ContextGen")
        print("lad=",lad)
        print("lod=",lod)
        self._lsd = 2 * self._lod
        if config == None:
            self.c = self.get_default_config()
        else:
            self.c = config

        # parameters
        self._enc_out_normalization = self.c.enc_out_norm
        self._learning_rate = self.c.learning_rate
        # main model

        # Its not ugly, its pythonic :)

        ###### ACRKN ENCODER LAYER OBJECTS DEFINED
        Encoder._build_hidden_layers = self._build_enc_hidden_layers
        enc = Encoder(lod, output_normalization=self._enc_out_normalization)
        self._enc = TimeDistributed(enc, num_outputs=2).to(self._device)

        ###### ACRKN CELL OBJECT DEFINED
        self._rkn_layer = acRKNLayer(latent_obs_dim=lod, act_dim=lad, cell_config=self.c).to(self._device)

        ###### ACRKN DECODER OBJECT DEFINED
        SplitDiagGaussianDecoder._build_hidden_layers_mean = self._build_dec_hidden_layers_mean
        SplitDiagGaussianDecoder._build_hidden_layers_var = self._build_dec_hidden_layers_var
        self._dec = TimeDistributed(SplitDiagGaussianDecoder(out_dim=target_dim), num_outputs=2).to(self._device)

        ##### build (default) initial state

        if self.c.learn_initial_state_covar:
            init_state_covar = elup1_inv(self.c.initial_state_covar)
            self._init_state_covar_ul = \
                nn.Parameter(nn.init.constant_(torch.empty(1, self._lsd), init_state_covar))
        else:
            self._init_state_covar_ul = self.c.initial_state_covar * torch.ones(1, self._lsd)

        self._initial_mean = torch.zeros(1, self._lsd).to(self._device)
        self._icu = torch.nn.Parameter(self._init_state_covar_ul[:, :self._lod].to(self._device))
        self._icl = torch.nn.Parameter(self._init_state_covar_ul[:, self._lod:].to(self._device))
        self._ics = torch.zeros(1, self._lod).to(self._device)

        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches
        print("saleh_add: line 125 recurrentEncoderDecoder.py ContextGen ")

    def _build_enc_hidden_layers(self):
        print("saleh_add: line 128 recurrentEncoderDecoder.py ContextGen ")
        layers = []
        # hidden layer 1
        layers.append(nn.Linear(in_features=self._inp_dim, out_features=self._lod))
        layers.append(nn.ReLU())
        return nn.ModuleList(layers), self._lod

    def _build_dec_hidden_layers_mean(self):
        print("saleh_add: line 136 recurrentEncoderDecoder.py ContextGen ")
        return nn.ModuleList([
            nn.Linear(in_features=2 * self._lod, out_features=self._inp_dim),
            nn.ReLU(),

        ]), self._inp_dim

    def _build_dec_hidden_layers_var(self):
        print("saleh_add: line 144 recurrentEncoderDecoder.py ContextGen ")
        return nn.ModuleList([
            nn.Linear(in_features=3 * self._lod, out_features=self._inp_dim),
            nn.ReLU()
        ]), self._inp_dim

    def forward(self, obs_batch: torch.Tensor, act_batch: torch.Tensor, obs_valid_batch: torch.Tensor, decode=False) -> \
    Tuple[float, float]:
        """Single update step on a batch
        :param obs_batch: batch of observation sequences
        :param act_batch: batch of action sequences
        :param obs_valid_batch: batch of observation valid flag sequences
        :param target_batch: batch of target sequences
        :param decode: whether to decode next_prior
        """
        # with tsensor.explain():
        #     obs_batch = obs_batch
        print("saleh_add: line 158 recurrentEncoderDecoder.py ContextGen ")
        w, w_var = self._enc(obs_batch)
        post_mean, post_cov, prior_mean, prior_cov = self._rkn_layer(w, w_var, act_batch, self._initial_mean,
                                                                     [self._icu, self._icl, self._ics], obs_valid_batch)
        print("line 162 recurrentEncoderDecoder.py   ContextGen" )
        print("Decde",decode)
        if decode:
            out_mean, out_var = self._dec(prior_mean, torch.cat(prior_cov, dim=-1))
            return prior_mean, prior_cov, out_mean, out_var
        else:
            return prior_mean, prior_cov


class acrknContextualDecoder(nn.Module):
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

    def __init__(self, ltd: int,input_dim: int ,target_dim: int, lod: int, lad: int, config: OmegaConf = None,
                 use_cuda_if_available: bool = True): #saleh_added
    # def __init__(self, ltd: int, input_dim: int, target_dim: int, lod: int, lad: int, config: OmegaConf = None,
    #              use_cuda_if_available: bool = True): #saleh_commented_out
        super(acrknContextualDecoder, self).__init__()
        """
               :param target_dim:
               :param lod:
               :param config:
               :param use_cuda_if_available:
               """

        self._device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        # change the target dim to 1D
        self._ltd = ltd
        #inp_dim shows the dimension of input which is not necessarily the same as target
        #self._inp_dim =target_dim# target_dim          #these are not the same anymore  #saleh_commented_out

        self._inp_dim = input_dim #saleh_added
        print("saleh_added line 223 recurrentEncoderDecoder.py", "self._inp_dim = ",self._inp_dim) #shows number of features
        print("saleh_added line 224 recurrentEncoderDecoder.py", "target_dim = ", target_dim)      #shows dimension of output (in my case it is 1)

        self._lad = lad
        self._lod = lod
        self._lsd = 2 * self._lod
        if config == None:
            self.c = self.get_default_config()
        else:
            self.c = config

        # parameters
        self._enc_out_normalization = self.c.enc_out_norm
        # main model

        ###### ACRKN ENCODER LAYER OBJECTS DEFINED
        Encoder._build_hidden_layers = self._build_enc_hidden_layers
        enc = Encoder(lod, output_normalization=self._enc_out_normalization, activation=self.c.variance_act)
        self._enc = TimeDistributed(enc, num_outputs=2).to(self._device)

        ###### ACRKN CELL OBJECT DEFINED
        self._rkn_layer = AcRKNContextLayer(ltd=self._ltd, latent_obs_dim=lod, act_dim=lad,
                                            config=self.c).to(self._device)

        ###### ACRKN DECODER OBJECT DEFINED
        SplitDiagGaussianDecoder._build_hidden_layers_mean = self._build_dec_hidden_layers_mean
        SplitDiagGaussianDecoder._build_hidden_layers_var = self._build_dec_hidden_layers_var
        # SimpleDecoder._build_hidden_layers = self._build_dec_hidden_layers
        self._dec = TimeDistributed(SplitDiagGaussianDecoder(out_dim=target_dim, activation=self.c.variance_act),
                                    num_outputs=2).to(self._device)

        print("line 257 recurrentEncoderDecoder.py Double check with vishak: out_dim=target_dim  ")

        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches

        ##### build (default) initial state

        if self.c.learn_initial_state_covar:
            init_state_covar = elup1_inv(self.c.initial_state_covar)
            self._init_state_covar_ul = \
                nn.Parameter(nn.init.constant_(torch.empty(1, self._lsd), init_state_covar))
        else:
            self._init_state_covar_ul = self.c.initial_state_covar * torch.ones(1, self._lsd)

        self._initial_mean = torch.zeros(1, self._lsd).to(self._device)
        self._icu = torch.nn.Parameter(self._init_state_covar_ul[:, :self._lod].to(self._device))
        self._icl = torch.nn.Parameter(self._init_state_covar_ul[:, self._lod:].to(self._device))
        self._ics = torch.zeros(1, self._lod).to(self._device)

        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches

    def _build_enc_hidden_layers(self):
        layers = []
        last_hidden = self._inp_dim
        # hidden layers
        for hidden_dim in self.c.enc_net_hidden_units:
            layers.append(nn.Linear(in_features=last_hidden, out_features=hidden_dim))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.25))
            last_hidden = hidden_dim
        return nn.ModuleList(layers), last_hidden

    def _build_dec_hidden_layers_mean(self): #here I should  add/edit the last linear layer
        layers = []
        if self.c.decoder_conditioning:
            last_hidden = self._lod * 2 + int(self._ltd / 2)
        else:
            last_hidden = self._lod * 2
        # hidden layers
        for hidden_dim in self.c.dec_net_hidden_units:
            layers.append(nn.Linear(in_features=last_hidden, out_features=hidden_dim))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.25))
            last_hidden = hidden_dim
        # here
        #add softmax for classification --->loss? cross entropy for classificatiin
        return nn.ModuleList(layers), last_hidden

    def _build_dec_hidden_layers_var(self):
        layers = []
        if self.c.decoder_conditioning:
            last_hidden = self._lod * 3 + int(self._ltd / 2)
        else:
            last_hidden = self._lod * 3
        # hidden layers
        for hidden_dim in self.c.dec_net_hidden_units:
            layers.append(nn.Linear(in_features=last_hidden, out_features=hidden_dim))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.25))
            last_hidden = hidden_dim
        return nn.ModuleList(layers), last_hidden


    def forward(self, obs_batch: torch.Tensor, act_batch: torch.Tensor, latent_task: torch.Tensor,
                obs_valid_batch: torch.Tensor, multiStep=0, decode=True) -> Tuple[float, float]:
        """Single update step on a batch
        :param obs_batch: batch of observation sequences
        :param act_batch: batch of action sequences
        :param obs_valid_batch: batch of observation valid flag sequences
        :param target_batch: batch of target sequences
        :param decode: whether to decode next_prior
        """
        conditional = False
        latent_task_mu = torch.unsqueeze(latent_task, 1).repeat(1, obs_batch.shape[1], 1)
        if conditional:
            w, w_var = self._enc(torch.cat([obs_batch, latent_task_mu[:, :, :self._ltd]], dim=-1))
        else:
            w, w_var = self._enc(obs_batch)
        post_mean, post_cov, prior_mean, prior_cov = self._rkn_layer(w, w_var, act_batch, latent_task,
                                                                     self._initial_mean,
                                                                     [self._icu, self._icl, self._ics], obs_valid_batch)
        if decode: #True

            # print('recurrentEncoderDecoder.py  line 317    ')
            # print('decode = ' , decode)

            if self.c.decoder_conditioning: #False

                # print('recurrentEncoderDecoder.py  line 324    ')
                # print('self.c.decoder_conditioning = ', self.c.decoder_conditioning)

                out_mean, out_var = self._dec(
                    torch.cat([prior_mean, latent_task_mu[:, :, :int(self._ltd / 2)]], dim=-1),
                    torch.cat([torch.cat(prior_cov, dim=-1), latent_task_mu[:, :, int(self._ltd / 2):]], dim=-1))
            else: #True:    miad inja:

                #print('recurrentEncoderDecoder.py  line 331    ')
                #print('self.c.decoder_conditioning = ', self.c.decoder_conditioning) #False
                out_mean, out_var = self._dec(prior_mean, torch.cat(prior_cov, dim=-1)) # in the case of 1D target (only predicting number of packets) the size is [350, 75,1] , [350,75,1]
                #print('recurrentEncoderDecoder.py line 334: checking shapes','out_mean.shape',out_mean.shape,'out_var.shape', out_var.shape)    #in the case of predicting multiple target [350, 75,#num_features] , [350,75,#num_features] . 350=training batch size, 75=context_size 50=dimension of output // or
                #print('out_mean.shape',out_mean.shape)
                #print('out_var.shape', out_var.shape)

                #print('recurrentEncoderDecoder.py line 338: saleh_added softmax')
                #print('out_mean (without any softmax)=', out_mean)
                #out_mean = nn.functional.softmax(out_mean,dim=2) #softmax on hist_bin (last feature excluded)
                #print('out_mean=', out_mean)

                #print('recurrentEncoderDecoder.py  line 338  ')
                # print('out_mean.shape =',out_mean.shape)  #torch.Size([350, 75=ctx, num_bin]) softmax should be applied on the dim=2 not on 0,1
                # print('out_var.shape  =',out_var.shape)   #torch.Size([350, 75=ctx, num_bin])
                # out_mean = self._dec(prior_mean) # commented by vaishak

            return out_mean, out_var
        else:

            return prior_mean, prior_cov




# cell_conf = AcRKNCell.get_default_config()
# AcRKN = acrknContextGen(2,3,4,cell_conf)
# print(AcRKN)
# for name, param in AcRKN.named_parameters():
#     if param.requires_grad:
#         print(name)
