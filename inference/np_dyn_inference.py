import time as t
from typing import Tuple

import numpy as np
import torch
from meta_dynamic_models.neural_process_dynamics.npDynamics import npDyn
from utils.dataProcess import split_k_m, get_sliding_context_batch_mbrl, get_ctx_target_multistep, get_ctx_target_impute,\
    squeeze_sw_batch, diffToStateMultiStep, diffToState, diffToStateImpute
from torch.utils.data import TensorDataset, DataLoader
from utils.dataProcess import split_k_m
from utils import ConfigDict

optim = torch.optim
nn = torch.nn


class Infer:

    def __init__(self, model: npDyn, data, config:ConfigDict = None, run = None, log=True, use_cuda_if_available: bool = True):

        """
        :param model: nn module for acrkn
        :param use_cuda_if_available:  if to use gpu
        """

        self._device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        self._data = data
        self._model = model
        if config is None:
            raise TypeError('Pass a Config Dict')
        else:
            self.c = config

        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches

    def predict(self, obs: torch.Tensor, act: torch.Tensor, y_context: torch.Tensor,
                imp: float = 0.0, k=32, test_gt_known=True,
                batch_size: int = -1, multiStep=0, tar="observations") -> Tuple[float, float]:
        """
        Predict using the model
        :param obs: observations to evaluate on
        :param act: actions to evaluate on
        :param obs_valid: observation valid flag
        :param y_context: the label information for the context sets
        :param batch_size: batch_size for evaluation, this does not change the results and is only to allow evaluating on more
         data than you can fit in memory at once. Default: -1, .i.e. batch_size = number of sequences.
        :param multiStep: how many multiStep ahead predictions do you need. You can also do this by playing with obs_valid flag.
        """
        # rescale only batches so the data can be kept in unit8 to lower memory consumptions
        self._model = self._model.eval()
        out_mean_list = []
        out_var_list = []
        gt_list = []
        obs_valid_list = []
        self._context_size = k
        print('"np_dyn_inference.py line 56   ,k = ',k)
        dataset = TensorDataset(obs, act, y_context)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        for batch_idx, (obs, act, target) in enumerate(loader):
            with torch.no_grad():
                # Assign data tensors to devices
                print("np_dyn_inference.py line 63, batach_idx=",batch_idx)
                obs_batch = (obs).to(self._device)
                act_batch = act.to(self._device)
                target_batch = (target).to(self._device)

                # Split to context and targets
                if self._context_size is None:
                    k = int(obs_batch.shape[1] / 2)
                else:
                    k = self._context_size
                m = obs_batch.shape[1] - k
                ctx_obs_batch, ctx_act_batch, ctx_target_batch, tar_obs_batch, tar_act_batch, tar_tar_batch, tar_obs_valid_batch = \
                    get_ctx_target_impute(obs_batch, act_batch, target_batch, k, num_context=None,
                                          test_gt_known=test_gt_known,
                                          tar_imp=imp,
                                          random_seed=True)

                ### Unlike in learning during inference we don't have access to Y_target

                context_Y = ctx_target_batch

                target_X = (tar_obs_batch, tar_act_batch, tar_obs_valid_batch) #different from maml

                context_X = torch.cat([ctx_obs_batch, ctx_act_batch], dim=-1)

                target_Y = tar_tar_batch

                # Forward Pass
                out_mean, out_var, _, _ = self._model(context_X, context_Y, target_X)

                # Diff To State
                if tar == "delta":
                    out_mean = \
                        torch.from_numpy(
                            diffToStateImpute(out_mean, tar_obs_batch, tar_obs_valid_batch, self._data,
                                              standardize=True)[0])
                    target_Y = \
                        torch.from_numpy(diffToState(tar_tar_batch, tar_obs_batch, self._data, standardize=True)[0])

                out_mean_list.append(out_mean.cpu())
                out_var_list.append(out_var.cpu())
                gt_list.append(
                    target_Y.cpu())  # if test_gt_known flag is False then we get list of Nones

        return torch.cat(out_mean_list), torch.cat(out_var_list), torch.cat(gt_list)
