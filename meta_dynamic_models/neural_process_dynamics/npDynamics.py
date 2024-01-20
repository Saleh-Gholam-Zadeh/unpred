import sys

sys.path.append('.')
import numpy as np
import torch
from omegaconf import OmegaConf

optim = torch.optim
nn = torch.nn

class npDyn(nn.Module):
    @staticmethod
    def get_default_config() -> OmegaConf:
        config = """
        clip_gradients: True
        latent_obs_dim: 30
        agg_dim: 60
        """
        return OmegaConf.create(config)

    '''
    Note that the aggregator is considered as an integral part of the encoder
    '''
    def __init__(self, encoder, decoder, dec_type='acrkn', latent_variance=True, config: OmegaConf=None, use_cuda_if_available: bool=True):
        '''
        encoder: context encoder and aggregator
        decoder: target decoder
        dec_type: 'acrkn' or 'ffnn'
        latent_variance: If True, decode the task level variance as well
        '''
        super(npDyn, self).__init__()
        if config == None:
            self.c = self.get_default_config()
        else:
            self.c = config
        self._device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        # model architecture
        self._encoder, self._decoder = encoder.to(self._device), decoder.to(self._device)
        self._dec_type = dec_type
        self._latent_variance = latent_variance
        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches


    def forward(self, context_inp, context_out, target_inp, multiStep=0):
        ##### Encode context to latent space
        mu_z, cov_z = self._encoder(context_inp, context_out)

        ###### Sample from the distribution if we have a VAE style sampling procedure with reparameterization
        ##### Conditioned on global latent task represenation make predictions on targets
        if self._latent_variance:
            latent_task = torch.cat((mu_z,cov_z),dim=-1)
        else:
            latent_task = mu_z
        if len(latent_task.shape) < 2:
            latent_task = torch.unsqueeze(latent_task,dim=0)
        target_obs, target_act, target_obs_valid = target_inp

        if self._dec_type == 'acrkn': #true
            mu_x, cov_x = self._decoder(target_obs,
                                              target_act, latent_task,
                                              target_obs_valid, multiStep)
        elif self._dec_type == 'ffnn':
            mu_x, cov_x = self._decoder(target_obs,
                                        target_act, latent_task,
                                        target_obs_valid)
        # print('npDynamics.py line 66')
        # print(mu_x.shape)   #torch.Size([300, 75, 50])
        # print(cov_x.shape)  #torch.Size([300, 75, 50])
        # print(mu_z.shape) #latent  torch.Size([300, 60])
        # print(cov_z.shape) #latent  torch.Size([300, 60])


        return mu_x, cov_x, mu_z, cov_z










