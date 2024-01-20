import os
import time as t
from typing import Tuple
import datetime

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import wandb
import matplotlib.pyplot as plt
import sys

from meta_dynamic_models.neural_process_dynamics.npDynamics import npDyn
from utils.dataProcess import split_k_m, get_ctx_target_impute
from utils.Losses import mse, mae, gaussian_nll , CrossEntropy, kl_rmse
from utils.PositionEmbedding import PositionEmbedding as pe
from utils import ConfigDict
from utils.plotTrajectory import plotImputation
from utils.latentVis import plot_clustering
from utils.dataProcess import get_mutual_information, run_test, linear_correl, count_elements_greater_than_MI

import time
import torch.autograd.profiler as profiler


optim = torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

nn = torch.nn

import random
import torch





def avg_pred(scores):
    '''
    scores: 2D matrices of batch_size*classes. each rows contains the score of each class
    return: convert the scores to probability using softmax and then take expected value
    '''
    device = scores.device
    scores  = scores.float()
    softmax = nn.Softmax(dim=-1)
    probs   = softmax(scores)
    classes = torch.arange(scores.size()[-1], device=device).float()
    avg     = torch.matmul(probs,classes)
    return torch.unsqueeze(avg,dim=-1)
#avg_pred(torch.tensor([[0, 0, 0, 5, 5],[5, 5, 0, 0, 0],[1, 0, 5, 0, 0]]))


def k_max_class(scores2, k_ind=1):
    '''
    receive raw scores and take the k_ind max of them and replace the rest with  "-Inf"
    scores2: 2D matrices of batch_size*classes. each rows contains the score of each class
    k_ind: how many of the scores should be considered (rest of scores will be replaced by -inf to have prob of 0)

    '''
    device = scores2.device
    vals_max, inds = torch.topk(scores2,k_ind)  # inds in each rows show the first k_ind max valuse #we dont need vals_max
    one_hot = nn.functional.one_hot(inds, num_classes=scores2.size()[-1])
    to_be_taken_inds = torch.sum(one_hot, dim=-2)  # to have all ones (to be take inds) in one vector
    cut_scores = torch.mul(scores2, to_be_taken_inds)  #
    cut_scores[cut_scores == 0] = float("-Inf")
    return cut_scores.float().to(device)


# a = torch.tensor([[0, 0, 0, 5, 5], [5, 5, 0, 0, 0], [1, 0, 5, 0, 0]])
# new_score = k_max_class(a)
# print(new_score)
# print(avg_pred(new_score))


class Learn:

    def __init__(self, model, loss: str, imp: float = 0.0, config: ConfigDict = None, run = None, log=True, use_cuda_if_available: bool = True  ,normalizer=None , num_class=None):
        """
        :param model: nn module for np_dynamics
        :param loss: type of loss to train on 'nll' or 'mse' or 'mae' (added by saleh) or 'CrossEntropy' (added by saleh)
        :param imp: how much to impute
        :param use_cuda_if_available: if gpu training set to True
        """
        assert run is not None, 'pass a valid wandb run'
        self._device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        self._loss = loss
        self._imp = imp
        self._model = model
        self._pe = pe(self._device)
        self._exp_name = run.name #+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self._num_class = num_class
        if config is None:
            raise TypeError('Pass a Config Dict')
        else:
            self.c = config

        self._tar_type=self.c.data_reader.tar_type   #saleh_added

        #print("self._tar_type = ", self._tar_type)
        self._normalizer = normalizer

        self._learning_rate = self.c.learn.lr

        self._save_path = os.getcwd() + '/experiments/saved_models/' + run.name + '.ckpt'
        self._cluster_vis = self.c.learn.latent_vis

        self._optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate)
        self._scheduler = ReduceLROnPlateau(self._optimizer, mode='min', patience=6, factor=0.5, verbose=True)
        self.cool_down = 0

        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches
        self._log = bool(log)
        self.save_model = self.c.learn.save_model
        if self._log:
            self._run = run

        self.pred_len     = self.c.data_reader.pred_len
        self.context_size = self.c.data_reader.context_size
        self.log_stat = self.c.learn.log_stat

    def train_step(self, train_dict: dict, batch_size: int)  -> Tuple[float, float, float]:
        """
        Train once on the entire dataset
        :param train_obs: training observations
        :param train_act: training actions
        :param train_targets: training targets
        :param train_task_idx: task ids per episode
        :param batch_size: batch size for each gradient update
        :return: average loss (nll) and  average metric (rmse), execution time
        """

        mse_per_batch = []

        #we set them to None in case user don't want to calculate them
        cnt_dep_var_tr_epoch = None
        actual_total_MI_train_epoch = None
        sum_r_tr_epcoh= None
        avg_MI_permute_tr_epoch =  None
        MI_pvl = None

        self._model.train()
        avg_loss = avg_metric_nll = avg_metric_mse  = avg_metric_mae = avg_metric_CrossEntropy = avg_metric_combined_kl_rmse = avg_metric_kl = 0
        t0 = t.time()
        # b = list(loader)[0]
        z_vis_list = []
        task_id_list = []

        #for key , inner_dict in train_dict.items():
        #from here --> shifted 1 inden back
        #print(key)
        train_obs = train_dict
        dataset = TensorDataset(torch.from_numpy(train_obs))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        obs_list = []
        residuals_list = []  # Empty list to collect residuals

        for batch_idx, obs in enumerate(loader):
            # Set Optimizer to Zero
            self._optimizer.zero_grad()
            # Assign tensors to device
            #print("batch_idx:",batch_idx)
            obs_batch = obs[0].to(self._device)
            # if(self._device==torch.device("cpu")):
            #     obs_batch=obs_batch[:5,:,:]


            ctx_obs_batch , tar_obs_batch = obs_batch[:,:self.context_size,:].float(),obs_batch[:,self.context_size:,-1:].float()
            Y = tar_obs_batch.to(self._device)
            X_enc = ctx_obs_batch.to(self._device)                                                                               # obs: -----ctx_obs-------/-------tar_obs------
            #Y = torch.cat([ctx_target_batch[:,-1:,:], tar_tar_batch[:,:-1,:]] , dim=1)[:,:self.pred_len,:].to(self._device)     # tar:  -----ctx_tar-------/------tar_tar-------

            try:
                #print("X_enc.shape", X_enc.shape)
                pred_logits,_ = self._model(X_enc.float())
            except:
                #print("X_enc.shape",X_enc.shape)
                pred_logits = self._model(X_enc.float())


            if self._loss == 'nll':
                loss = gaussian_nll(Y, out_mean, out_var)
            elif self._loss == 'mae':
                loss = mae(pred_logits, Y)
            elif self._loss == 'mse':
                #loss = mse(tar_tar_batch, out_mean)
                mseloss = nn.MSELoss()
                loss = mseloss(pred_logits, Y)

            else:
                raise NotImplementedError

            mse_per_batch.append(loss.item())


            # Backward Pass
            # with profiler.profile() as prof:
            loss.backward()
            #print(prof)
            # Clip Gradients
            if self.c.np.clip_gradients:
                torch.nn.utils.clip_grad_norm(self._model.parameters(), 5.0)

            # Backward Pass Via Optimizer
            self._optimizer.step()

            # if (self._loss  in ['cross_entropy']):
            #     scores = k_max_class(pred_logits) #newly added
            #     k_averaged = avg_pred(scores) #newly added
            #     #out_mean_pred = out_mean
            #
            #
            with torch.no_grad():  #
                if (self._loss == 'cross_entropy'):
                    out_mean_pred = torch.argmax(pred_logits, dim=-1).unsqueeze(dim=-1).float()   #yek adad beyne 0 ta num_class bar migardune baraye dim e -1
                    out_mean_pred = (2  * out_mean_pred + 1 - self._normalizer[0])/self._normalizer[1] # convert back from bin_number to percentage Then normalize it
                else:
                    out_mean_pred = pred_logits

                # try:
                #     metric_nll = gaussian_nll(Y, k_averaged, out_var)
                # except:
                #     metric_nll = np.nan

                if(self.c.data_reader.standardize and not(self._loss == 'cross_entropy')):  # we always consider the normalized error
                    metric_mse = mse( out_mean_pred,Y)
                    #metric_packet_mse = mse( out_mean_pred[:, :, -1:], Y[:, :, -1:])
                    metric_mae = mae(out_mean_pred, Y)
                else:
                    # print(type(out_mean_pred))
                    # print(type(Y))
                    # print(type((self._normalizer['packets'][1])))
                    metric_mse = mse(out_mean_pred,Y)/( torch.from_numpy(self._normalizer[1]) )**2
                    #metric_packet_mse = mse( out_mean_pred[:, :, -1:], Y[:, :, -1:])/torch.from_numpy(self._normalizer['packets'][1]).to(self._device)**2
                    metric_mae = mae( out_mean_pred,Y)/(torch.from_numpy(self._normalizer[1]))
                try:
                    metric_CrossEntropy = CrossEntropy( pred_logits,Y_class,C=self._num_class,manual=False)
                except:
                    metric_CrossEntropy = np.nan

                try:
                    metric_combined_kl_rmse, metric_kl,_= kl_rmse( pred_logits,Y,should_normalize=not(self.c.data_reader.standardize))
                except:
                    metric_combined_kl_rmse , metric_kl= np.nan , np.nan



                #z_vis_list.append(mu_z.detach().cpu().numpy())
                #task_id_list.append(task_id.detach().cpu().numpy())
            avg_loss += loss.detach().cpu().numpy()
            try:
                avg_metric_nll += metric_nll.detach().cpu().numpy()
            except:
                avg_metric_nll = np.nan

            avg_metric_mse += metric_mse.detach().cpu().numpy()
            #avg_metric_packet_mse += metric_packet_mse.detach().cpu().numpy()
            avg_metric_mae += metric_mae.detach().cpu().numpy()

            try:
                avg_metric_CrossEntropy += metric_CrossEntropy.detach().cpu().numpy()
            except:
                avg_metric_CrossEntropy = np.nan

            try:
                avg_metric_combined_kl_rmse += metric_combined_kl_rmse.detach().cpu().numpy()
            except:
                avg_metric_combined_kl_rmse = np.nan

            try:
                avg_metric_kl += metric_kl.detach().cpu().numpy()
            except:
                avg_metric_kl = np.nan

            if self.log_stat:
                # Calculate residuals and accumulate them in the list
                with torch.no_grad():  # Ensures no gradient computation for the following operations
                    residuals = Y - pred_logits  # residuals [B, 1, 1]
                residuals_list.append(residuals.cpu())
                obs_list.append(X_enc.detach().cpu())


        # outside of dataLoader loop
        if self.log_stat:
            # Stack residuals after the epoch
            stacked_residuals = torch.cat(residuals_list, dim=0)
            stacked_ctx = torch.cat(obs_list, dim=0)
            stacked_arr = torch.cat([stacked_ctx, stacked_residuals], dim=1)
            print(stacked_arr.shape)
            # results0, pvl0, cnt_dep0 = run_test(tr_batched.squeeze().swapaxes(0,1), 0.05 * n, cfg.data_reader.pred_len, log=False, bonfer=True)

            _, pvl_tr_epoch, cnt_dep_var_tr_epoch = run_test(stacked_arr.numpy().squeeze().swapaxes(0, 1),
                                                             number_output_functions=self.pred_len, log=False,
                                                             bonfer=True)
            # _ --was--> dep_var_tr_epoch

            _, actual_total_MI_train_epoch, MI_pvl, avg_MI_permute_tr_epoch, total_MI_epoch_tr_for_each_permutation = get_mutual_information(
                stacked_arr.numpy().squeeze().swapaxes(0, 1), number_output_functions=self.pred_len,
                perm_test_flag=True, N=100)

            # Pearson_r
            sum_r_tr_epcoh = linear_correl(stacked_arr.numpy().squeeze().swapaxes(0, 1))

        #to here

            #print("loss")
        # taking sqrt of final avg_mse gives us rmse across an epoch without being sensitive to batch size
        assert len(mse_per_batch)==len(loader) , "something went wrong"
        with torch.no_grad():
            if self._loss == 'nll':
                avg_loss = avg_loss /  len(mse_per_batch)


            elif self._loss == 'mse':
                avg_loss = np.sqrt(avg_loss /  len(mse_per_batch))

            elif self._loss == 'mae':
                avg_loss = avg_loss / (  len(mse_per_batch) )

            elif self._loss == 'cross_entropy':
                avg_loss = (avg_loss / len(mse_per_batch))

            elif self._loss == 'kl_rmse':
                avg_loss = (avg_loss / len(mse_per_batch))

            else:
                raise NotImplementedError



        with torch.no_grad():
            self._tr_sample_gt = Y.detach().cpu().numpy() [:,:,-1:]  #   #packets for the sake of plots
            self._tr_sample_pred_mu = pred_logits.detach().cpu().numpy()[:,:,-1:] #   #packets for the sake of plots

            try:
                self._tr_sample_pred_var = out_var.detach().cpu().numpy()
            except:
                self._tr_sample_pred_var = np.nan

            self._tr_sample_tar_obs = tar_obs_batch.detach().cpu().numpy()[:,:,-1:]   # saleh_added

        avg_metric_nll = avg_metric_nll / len(mse_per_batch)
        avg_metric_rmse = np.sqrt(avg_metric_mse / len(mse_per_batch))
        #avg_metric_packet_rmse = np.sqrt(avg_metric_packet_mse / len(mse_per_batch))
        avg_metric_mae = avg_metric_mae / len(mse_per_batch)
        avg_metric_CrossEntropy = avg_metric_CrossEntropy / len(mse_per_batch)
        avg_metric_combined_kl_rmse     = avg_metric_combined_kl_rmse / len(mse_per_batch)
        avg_metric_kl = avg_metric_kl / len(mse_per_batch)


        return avg_loss, avg_metric_nll, avg_metric_rmse,avg_metric_mae,avg_metric_CrossEntropy,avg_metric_combined_kl_rmse,avg_metric_kl , None, None, t.time() - t0 , cnt_dep_var_tr_epoch, actual_total_MI_train_epoch , sum_r_tr_epcoh, avg_MI_permute_tr_epoch,MI_pvl

    @torch.no_grad()
    def eval(self, test_dict: np.ndarray, batch_size: int = -1) -> Tuple[float, float]:
        """
        Evaluate model
        :param obs: observations to evaluate on
        :param act: actions to evaluate on
        :param targets: targets to evaluate on
        :param task_idx: task index
        :batch_size: batch_size for evaluation, this does not change the results and is only to allow evaluating on more
         data than you can fit in memory at once. Default: -1, .i.e. batch_size = number of sequences.
        """
        # rescale only batches so the data can be kept in unit8 to lower memory consumptions
        self._model.eval()
        val_mse_per_batch = []


        avg_loss = avg_metric_nll = avg_metric_mse = avg_metric_mae= avg_metric_CrossEntropy = avg_metric_combined_kl_rmse = avg_metric_kl = 0.0
        avg_metric = 0.0
        z_vis_list = []
        task_id_list = []

        cnt_dep_var_val_epoch = None
        actual_total_MI_val_epoch = None
        sum_r_val_epcoh= None
        avg_MI_permute_val_epoch =  None
        MI_pvl_val = None



        #for key , inner_dict in test_dict.items():
        #from here
        test_obs = test_dict
        dataset = TensorDataset(torch.from_numpy(test_obs))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        obs_list = []
        residuals_list = []  # Empty list to collect residuals
        for batch_idx, (obs_batch) in enumerate(loader):
            with torch.no_grad():
                # Assign tensors to devices

                obs_batch = obs_batch[0]

                # if(self._device== torch.device("cpu")):
                #     obs_batch=obs_batch[:5,:,:]

                # Split to context and targets
                k = self.context_size
                m = self.pred_len

                ctx_obs_batch, tar_obs_batch = obs_batch[:, :self.context_size, :].float(), obs_batch[:, self.context_size:, -1:].float()
                Y = tar_obs_batch.to(self._device)
                X_enc = ctx_obs_batch.to(self._device)


                try:
                    pred_logits, _ = self._model(X_enc)
                except:
                    pred_logits = self._model(X_enc)
                # print('eval:')
                # print("X_enc.shape:",X_enc.shape)
                # print("(GT)Y.shape:",Y.shape)
                # print("pred_logits",pred_logits.shape)

                #print("hiprssm_dyn_trainer.py line 372  newly added")
                if (self._loss in [ 'cross_entropy']):
                    scores = k_max_class(pred_logits)  # newly added
                    k_averaged = avg_pred(scores)  # newly added
                    #out_mean_pred = out_mean


                if (self._loss == 'cross_entropy'):
                    # GT should be transoformed accordingly
                    Y_denorm = Y * self._normalizer[1] + self._normalizer[0]
                    # Discretize ground truth into classes (0 to C-1)
                    Y_class = (Y_denorm // 2).clamp(max=69)

                    # Reshape Y to match the shape of discretized_ground_truth
                    B,T2,_ = Y.shape
                    Y_class = Y_class.view(B, T2, -1)  # Flatten the last dimension to match the number of classes

                    # print("pred_logits.shape:", pred_logits.shape)
                    # print("y.shape:", Y_class.shape)

                    # print("!!!PLEASE DOUBLE CHECK INPUT AND TARGET") #done
                    #loss = CrossEntropy(pred_logits, Y_class, C=self._num_class, manual=False)

                    out_mean_pred = torch.argmax(pred_logits, dim=-1).unsqueeze(dim=-1).float()  # yek adad beyne 0 ta C-1 bar migardune
                    out_mean_pred = (2*out_mean_pred +1 - self._normalizer[0])/self._normalizer[1]

                else:
                    out_mean_pred = pred_logits

                self._te_sample_gt       = Y.detach().cpu().numpy() [:,:,-1:] #   #packets for the sake of plots
               # self._te_sample_valid    = tar_obs_valid_batch.detach().cpu().numpy()

                self._te_sample_pred_mu  = pred_logits.detach().cpu().numpy()[:,:,-1:] #   #packets for the sake of plots
                try:
                    self._te_sample_pred_var = out_var.detach().cpu().numpy()
                except:
                    self._te_sample_pred_var = np.nan

                self._te_sample_tar_obs  = tar_obs_batch.detach().cpu().numpy()[:,:,-1:]  #added_by saleh


                ## Calculate Loss
                if self._loss == 'nll':
                    loss = gaussian_nll(Y, out_mean_pred, out_var)

                elif self._loss == 'mse':
                    loss = mse( pred_logits,Y)

                elif self._loss == 'mae':
                    loss = mae(pred_logits,Y)

                elif self._loss == 'cross_entropy':   # target(GT) should come 2nd
                    loss = CrossEntropy(pred_logits,Y_class  ,C=self._num_class , manual=False)

                elif self._loss == 'kl_rmse':  # lambda1*KL(P||Q) + lambda2*mse     # use it with unormalized data
                    loss,distr_kl,_ = kl_rmse(pred_logits, Y, lambda1=0.5, lambda2=0.5, normalization=self._normalizer)

                else:
                    raise NotImplementedError

                val_mse_per_batch.append(loss)

                #metric_nll = gaussian_nll(tar_tar_batch, out_mean_pred, out_var)
                try:
                    metric_nll = gaussian_nll(Y, k_averaged, out_var)
                except:
                    metric_nll = np.nan

                if(self.c.data_reader.standardize):  # we always consider the normalized the error
                    metric_mse = mse(Y, out_mean_pred)
                   #metric_packet_mse = mse(Y[:, :, -1:], out_mean_pred[:, :, -1:])
                    metric_mae = mae(Y, out_mean_pred)
                else:
                    metric_mse = mse(Y, pred_logits)/(self._normalizer[1])**2
                    #metric_packet_mse = mse(out_mean_pred[:, :, -1:], pred_logits[:, :, -1:])/(self._normalizer['packets'][1])**2
                    metric_mae = mae(Y, out_mean_pred)/(self._normalizer[1])


                try:
                    metric_CrossEntropy = CrossEntropy(pred_logits,Y,C=self._num_class,manual=False)
                except:
                    metric_CrossEntropy = np.nan

                try:
                    metric_comined , metric_kl,  _ = kl_rmse(pred_logits, Y, lambda1=0.5, lambda2=0.5, normalization=self._normalizer)
                except:
                    metric_comined , metric_distr_kl = np.nan , np. nan




                try:
                    z_vis_list.append(mu_z.detach().cpu().numpy())
                except:
                    z_vis_list = []
                #task_id_list.append(task_id.detach().cpu().numpy())




                avg_loss += loss.detach().cpu().numpy()
                try:
                    avg_metric_nll += metric_nll.detach().cpu().numpy()
                except:
                    avg_metric_nll = np.nan


                avg_metric_mse += metric_mse.detach().cpu().numpy()
                #avg_metric_packet_mse += metric_packet_mse.detach().cpu().numpy()
                avg_metric_mae += metric_mae.detach().cpu().numpy()
                try:
                    avg_metric_CrossEntropy += metric_CrossEntropy.detach().cpu().numpy()
                except:
                    avg_metric_CrossEntropy = np.nan

                try:
                    avg_metric_combined_kl_rmse += metric_comined.detach().cpu().numpy()
                except:
                    avg_metric_combined_kl_rmse= np.nan

                try:
                    avg_metric_kl += metric_kl.detach().cpu().numpy()
                except:
                    avg_metric_kl = np.nan

            if self.log_stat:
                # Calculate residuals and accumulate them in the list
                with torch.no_grad():  # Ensures no gradient computation for the following operations
                    residuals = Y - pred_logits  # residuals [B, 1, 1]
                residuals_list.append(residuals.cpu())
                obs_list.append(X_enc.detach().cpu())

        # outside of dataLoader loop
        if self.log_stat:
            # Stack residuals after the epoch
            stacked_residuals = torch.cat(residuals_list, dim=0)
            stacked_ctx = torch.cat(obs_list, dim=0)
            stacked_arr = torch.cat([stacked_ctx, stacked_residuals], dim=1)
            #print(stacked_arr.shape)

            ############
            _, pvl_val_epoch, cnt_dep_var_val_epoch = run_test(stacked_arr.numpy().squeeze().swapaxes(0, 1),
                                                               number_output_functions=self.pred_len, log=False,
                                                               bonfer=True)
            _, actual_total_MI_val_epoch, MI_pvl_val, avg_MI_permute_val_epoch, total_MI_epoch_val_for_each_permutation = get_mutual_information(
                stacked_arr.numpy().squeeze().swapaxes(0, 1), number_output_functions=self.pred_len,
                perm_test_flag=True, N=100)

            # Pearson_r
            sum_r_val_epcoh = linear_correl(stacked_arr.numpy().squeeze().swapaxes(0, 1),number_output_functions=self.pred_len)

            ###############





        # to here
        # taking sqrt of final avg_mse gives us rmse across an epoch without being sensitive to batch size
        #print("line 337 transformer_trainer.py   self._loss=",self._loss)
        if self._loss == 'nll':
            avg_loss = avg_loss / len(val_mse_per_batch)

        elif self._loss == 'mse':
            avg_loss = np.sqrt(avg_loss /  len(val_mse_per_batch))

        elif self._loss == 'mae':
            avg_loss = avg_loss /  len(val_mse_per_batch)

        elif self._loss == 'cross_entropy':
            avg_loss = avg_loss / len(val_mse_per_batch)

        elif self._loss == 'kl_rmse':
            avg_loss = avg_loss /  len(val_mse_per_batch)

        else:
            raise NotImplementedError
        self._scheduler.step(avg_loss)

        if self._scheduler.in_cooldown:
            print("Learning rate was reduced!")
            self.cool_down = self.cool_down+1


        avg_metric_nll = avg_metric_nll /  len(val_mse_per_batch)
        avg_metric_rmse = np.sqrt(avg_metric_mse / len(val_mse_per_batch))
        #avg_metric_packet_rmse = np.sqrt(avg_metric_packet_mse /  len(val_mse_per_batch))
        avg_metric_mae =(avg_metric_mae /  len(val_mse_per_batch))
        avg_metric_CrossEntropy = (avg_metric_CrossEntropy /  len(val_mse_per_batch))
        avg_metric_combined_kl_rmse = (avg_metric_combined_kl_rmse / len(val_mse_per_batch))
        avg_metric_kl = (avg_metric_kl /  len(val_mse_per_batch))

        try:
            z_vis = np.concatenate(z_vis_list, axis=0)
        except:
            z_vis = 0
        #z_vis = 0
        task_labels = []
        return avg_loss, avg_metric_nll, avg_metric_rmse, avg_metric_mae, avg_metric_CrossEntropy,avg_metric_combined_kl_rmse,avg_metric_kl ,None, None, cnt_dep_var_val_epoch, actual_total_MI_val_epoch , sum_r_val_epcoh, avg_MI_permute_val_epoch,MI_pvl_val

    def train(self, train_obs: torch.Tensor, epochs: int, batch_size: int,val_obs: torch.Tensor = None,val_interval: int = 1,val_batch_size: int = -1) -> None:
        '''
        :param train_obs: training observations for the model (includes context and targets)
        :param train_act: training actions for the model (includes context and targets)
        :param train_targets: training targets for the model (includes context and targets)
        :param train_task_idx: task_index for different training sequence
        :param epochs: number of epochs to train on
        :param batch_size: batch_size for gradient descent
        :param val_obs: validation observations for the model (includes context and targets)
        :param val_act: validation actions for the model (includes context and targets)
        :param val_targets: validation targets for the model (includes context and targets)
        :param val_task_idx: task_index for different testing sequence
        :param val_interval: how often to perform validation
        :param val_batch_size: batch_size while performing inference
        :return:
        '''


        """ Train Loop"""
        torch.cuda.empty_cache() #### Empty Cache
        if val_batch_size == -1:
            val_batch_size = 4 * batch_size
        best_loss = np.inf
        best_nll = np.inf
        best_rmse = np.inf
        #best_packet_rmse = np.inf
        best_mae = np.inf
        best_CrossEntropy = np.inf
        best_combined_kl_rmse = np.inf
        best_kl = np.inf




        if self._log:
            wandb.watch(self._model, log='all')
            artifact = wandb.Artifact('saved_model', type='model')
        init_lr = self._optimizer.param_groups[0]['lr']
        print("initial_learning_rate",init_lr)
        print("===========================...epoching...=================================")
        for i in range(epochs):
            print("===================================================================================")
            #print("epochs = :",i , "/",epochs)
            print(f"Epoch {i+1}: Learning Rate: {self._optimizer.param_groups[0]['lr']}")
            old_lr = self._optimizer.param_groups[0]['lr']

            print("sceduler_cooldown_counter:", self.cool_down )
            if (self._optimizer.param_groups[0]['lr']< init_lr*1e-4):
                print("scheduler terminates the training ")
                return

            # if i == 0:
            #     print('<<<<<<<<<<<<Set Encoder:>>>>>>>>>>>>', self.c.set_encoder)
            #     print('<<<<<<<<<<<<Neural Process:>>>>>>>>>>>>', self.c.np)
            #     print('<<<<<<<<<<<<SSM:>>>>>>>>>>>>', self.c.ssm_decoder)


            train_loss, train_metric_nll, train_metric_rmse,train_metric_mae, train_metric_CrossEntropy,train_metric_combined_kl_rmse,train_metric_kl ,z_vis, z_labels, time , cnt_dep_var_tr, SUM_MI_train , sum_r_tr, avg_MI_permute_tr,MI_tr_pv = self.train_step(train_obs,batch_size)
            try:
                print("Training Iteration {:04d}: {}:{:.5f}, {}:{:.5f}, {}:{:.5f}, {}:{:.5f}".format(i + 1, self._loss, train_loss, 'target_rmse:', train_metric_rmse, 'target_mae:', train_metric_mae, 'target_CrossEntropy:', train_metric_CrossEntropy, 'target_kl_rmse:', train_metric_combined_kl_rmse))
                print("Took", time, " seconds")
            except:
                #print("Training Iteration {:04d}: {}:{:.5f}, {}:{:.5f}, {}:{:.5f}, {}:{:.5f}, Took {:4f} seconds".format(i + 1, self._loss, train_loss,  'target_rmse:',train_metric_rmse[0],'target_mae:', train_metric_mae[0], 'target_CrossEntropy:', train_metric_CrossEntropy,'target_kl_rmse:',train_metric_combined_kl_rmse,time))
                print("Training Iteration {:04d}: {}:{:.5f}, {}:{:.5f}, {}:{:.5f}".format(i + 1, self._loss, train_loss,  'target_rmse:',train_metric_rmse[0],'target_mae:', train_metric_mae[0], 'target_CrossEntropy:', train_metric_CrossEntropy))
                print("Took",time," seconds")

            # self._writer.add_scalar(self._loss + "/train_loss", train_loss, i)
            # self._writer.add_scalar("nll/train_metric", train_metric_nll, i)
            # self._writer.add_scalar("rmse/train_metric", train_metric_rmse, i)
            if self._log:
                try:
                    if self.log_stat:
                        wandb.log( {self._loss + "/train_loss": train_loss, "nll/train_metric": train_metric_nll, "rmse/train_metric": train_metric_rmse, "mae/train_metric": train_metric_mae,   "CrossEntropy/train_metric": train_metric_CrossEntropy, "kl/train_metric": train_metric_kl,"Train_stat/num_dep_train":cnt_dep_var_tr, "Train_stat/MI_train":float(SUM_MI_train), "Train_stat/Pearson_correlation":sum_r_tr,"Train_stat/avg_MI_permute":avg_MI_permute_tr,"Train_stat/MI_pvalue":MI_tr_pv ,"epochs": i}, commit=False)
                    else:
                        wandb.log({self._loss + "/train_loss": train_loss, "nll/train_metric": train_metric_nll, "rmse/train_metric": train_metric_rmse , "mae/train_metric": train_metric_mae, "CrossEntropy/train_metric": train_metric_CrossEntropy,   "kl/train_metric": train_metric_kl ,"epochs": i}, commit=False)

                except:
                    if self.log_stat:
                        wandb.log({self._loss + "/train_loss": train_loss, "nll/train_metric": train_metric_nll, "rmse/train_metric": train_metric_rmse[0] , "mae/train_metric": train_metric_mae[0], "CrossEntropy/train_metric": train_metric_CrossEntropy, "kl/train_metric": train_metric_kl,"Train_stat/num_dep_train":cnt_dep_var_tr, "Train_stat/MI_train":float(SUM_MI_train), "Train_stat/Pearson_correlation":sum_r_tr,"Train_stat/avg_MI_permute":avg_MI_permute_tr,"Train_stat/MI_pvalue":MI_tr_pv ,"epochs": i}, commit=False)

                    else:
                        wandb.log({self._loss + "/train_loss": train_loss, "nll/train_metric": train_metric_nll, "rmse/train_metric": train_metric_rmse[0] , "mae/train_metric": train_metric_mae[0], "CrossEntropy/train_metric": train_metric_CrossEntropy,  "kl/train_metric": train_metric_kl ,"epochs": i}, commit=False)


            # print("val_act:", val_act)   #saleh_added
            # print("val_targets:", val_targets)  # saleh_added
            # print("saleh_added hiprssm_dyn_trainer.py line 316")#saleh_added
            # val_act = torch.zeros_like(val_act)     #saleh_added
            # val_targets = torch.zeros_like(val_targets)  # saleh_added
            #print(i)
            if val_obs is not None  and np.mod(i,val_interval)  == 0:
                val_loss, val_metric_nll, val_metric_rmse ,val_metric_mae, val_metric_CrossEntropy, val_metric_combined_kl_rmse, val_metric_kl ,z_vis_val, z_labels_val ,cnt_dep_var_val, SUM_MI_val , sum_r_val, avg_MI_permute_val, MI_val_pv = self.eval(val_obs, batch_size=val_batch_size)

                new_lr = self._optimizer.param_groups[0]['lr']
                if new_lr!=old_lr :
                    print("Learning rate was reduced ")
                    self.cool_down=self.cool_down+1

                if val_loss < best_loss:
                    if self.save_model:
                        print('>>>>>>>Saving Best Model<<<<<<<<<<',"epoch:",i+1)
                        torch.save(self._model.state_dict(), self._save_path)
                    if self._log:
                        wandb.run.summary['best_loss'] = val_loss
                    best_loss = val_loss
                if val_metric_nll < best_nll:
                    if self._log:
                        wandb.run.summary['best_nll'] = val_metric_nll
                    best_nll = val_metric_nll
                if val_metric_rmse < best_rmse:
                    if self._log:
                        wandb.run.summary['best_rmse'] = val_metric_rmse
                    best_rmse = val_metric_rmse

                # if val_metric_packet_rmse < best_packet_rmse:
                #     if self._log:
                #         if (self._loss == 'mse' or self._loss == 'kl_rmse'):  # it means inpu and output are normalized
                #             wandb.run.summary['best_packet_rmse_normalized'] = val_metric_packet_rmse
                            #wandb.run.summary['best_packet_rmse_Denormalized'] = val_metric_packet_rmse * self._normalizer['std']  #rmse_norm * std
                        # elif(self._loss == 'cross_rntropy' ):
                        #     wandb.run.summary['best_packet_rmse_Denormalized'] = val_metric_packet_rmse
                        #     wandb.run.summary['best_packet_rmse_normalized'] = val_metric_packet_rmse / self._normalizer['std']  #rmse_Denorm / std



                    #best_packet_rmse = val_metric_packet_rmse

                if val_metric_mae < best_mae:
                    if self._log:
                        wandb.run.summary['best_mae'] = val_metric_mae
                    best_mae = val_metric_mae

                if val_metric_CrossEntropy < best_CrossEntropy:
                    if self._log:
                        wandb.run.summary['best_CrossEntropy'] = val_metric_CrossEntropy
                    best_CrossEntropy = val_metric_CrossEntropy

                if val_metric_combined_kl_rmse < best_combined_kl_rmse:
                    if self._log:
                        wandb.run.summary['best_combined_kl_rmse'] = val_metric_combined_kl_rmse
                    best_combined_kl_rmse = val_metric_combined_kl_rmse

                if val_metric_kl < best_kl:
                    if self._log:
                        wandb.run.summary['best_kl'] = val_metric_kl
                    best_kl = val_metric_kl


                try:
                    print("Validation: {}: {:.5f}, {}: {:.5f}, {}: {:.5f}, {}: {:.5f}, {}: {:.5f}, {}: {:.5f}".format(self._loss, val_loss, 'target_nll',val_metric_nll, 'target_rmse', val_metric_rmse , 'target_mae', val_metric_mae, 'target_CrossEntropy', val_metric_CrossEntropy , 'target_combined_kl_rmse',val_metric_combined_kl_rmse , 'target_kl', val_metric_kl  ))
                except:
                    print("Validation: {}: {:.5f}, {}: {:.5f}, {}: {:.5f}, {}: {:.5f}, {}: {:.5f}, {}: {:.5f}".format(self._loss, val_loss, 'target_nll',val_metric_nll, 'target_rmse', val_metric_rmse[0] , 'target_mae', val_metric_mae[0], 'target_CrossEntropy', val_metric_CrossEntropy , 'target_combined_kl_rmse',val_metric_combined_kl_rmse , 'target_kl', val_metric_kl ))

                if self._log:
                    if self.log_stat:
                        wandb.log({self._loss + "/val_loss": val_loss, "nll/test_metric": val_metric_nll, "rmse/test_metric": val_metric_rmse, "mae/test_metric": val_metric_mae, "CrossEntropy/test_metric": val_metric_CrossEntropy, "Val_stat/num_dep_val":cnt_dep_var_val, "Val_stat/MI_val":float(SUM_MI_val), "Val_stat/Pearson_correlation_val":sum_r_val,"Val_stat/avg_MI_permute_val":avg_MI_permute_val,"Val_stat/MI_pvalue_val":MI_val_pv , "epochs": i})
                    else:
                        wandb.log({self._loss + "/val_loss": val_loss, "nll/test_metric": val_metric_nll, "rmse/test_metric": val_metric_rmse, "mae/test_metric": val_metric_mae, "CrossEntropy/test_metric": val_metric_CrossEntropy, "combined_kl_rmse/test_metric":val_metric_combined_kl_rmse, "epochs": i})

                if self._cluster_vis:
                    z_vis_concat = np.concatenate((z_vis, z_vis_val), axis=0)
                    z_labels_concat = np.concatenate((z_labels, z_labels_val), axis=0)
                    ####### Visualize the tsne embedding of the latent space in tensorboard
                    if self._log and i == (epochs - 1):
                        print('>>>>>>>>>>>>>Visualizing Latent Space<<<<<<<<<<<<<<', '---Epoch----', i)
                        ind = np.random.permutation(z_vis.shape[0])
                        z_vis = z_vis[ind, :]
                        z_vis = z_vis[:2000, :]
                        z_labels = z_labels[ind]
                        z_labels = z_labels[:2000]
                        ####### Visualize the tsne/pca in matplotlib / pyplot

                        plot_clustering(z_vis_concat, z_labels_concat, engine='matplotlib',
                                        exp_name=self._exp_name + '_' + str(i), wandb_run=self._run)
                        plot_clustering(z_vis, z_labels, engine='matplotlib',
                                        exp_name=self._exp_name + '_' + str(i), wandb_run=self._run)
                        plot_clustering(z_vis_val, z_labels_val, engine='matplotlib',
                                        exp_name=self._exp_name + '_' + str(i), wandb_run=self._run)

        if self.c.learn.save_model:
            artifact.add_file(self._save_path)
            wandb.log_artifact(artifact)
        if self.c.learn.plot_traj:
            #print('plotimp() from hiprssm_dyn_trainer.py line 416')

            if(self._tar_type=="delta"):
                print("plotImputation with delta")      #                                                                                                                                        #the lasst 2 param are used to convert from diff to actual
                plotImputation(self._tr_sample_gt, self._tr_sample_valid, self._tr_sample_pred_mu, self._tr_sample_pred_var,self._run, log_name='train', exp_name=self._exp_name ,tar_type='delta',tar_obs=self._tr_sample_tar_obs ,normalizer=self._normalizer)
                plotImputation(self._te_sample_gt, self._te_sample_valid, self._te_sample_pred_mu, self._te_sample_pred_var,self._run, log_name='test',  exp_name=self._exp_name, tar_type='delta',tar_obs=self._te_sample_tar_obs, normalizer=self._normalizer)

                #plotImputation(self._tr_sample_gt, self._tr_sample_valid, self._tr_sample_pred_mu, self._tr_sample_pred_var,self._run, log_name='train_delta', exp_name=self._exp_name ,tar_type=None,tar_obs=None)
                #plotImputation(self._te_sample_gt, self._te_sample_valid, self._te_sample_pred_mu, self._te_sample_pred_var,self._run, log_name='test_delta',  exp_name=self._exp_name, tar_type=None,tar_obs=None)

            if (self._tar_type == "observations" and self._loss!="cross_entropy"  and self._loss!="kl_rmse"):
                print("plotImputation with observation")

                plotImputation(self._tr_sample_gt, None, self._tr_sample_pred_mu,self._tr_sample_pred_var, self._run, log_name='train', exp_name=self._exp_name,tar_type="observations", tar_obs=self._tr_sample_tar_obs,normalizer=self._normalizer)
                plotImputation(self._te_sample_gt, None, self._te_sample_pred_mu,self._te_sample_pred_var, self._run, log_name='test', exp_name=self._exp_name ,tar_type="observations" , tar_obs=self._te_sample_tar_obs,normalizer=self._normalizer)

            if (self._tar_type == "observations" and self._loss == "cross_entropy" ):
                pass
                # print("plotImputation with observation and CrossEntropy")
                #
                # plotImputation(self._tr_sample_gt, None, self._tr_sample_pred_mu,self._tr_sample_pred_var, self._run, log_name='train', exp_name=self._exp_name,tar_type="observations", tar_obs=self._tr_sample_tar_obs, normalizer=None, loss=self._loss)
                # plotImputation(self._te_sample_gt, None, self._te_sample_pred_mu,self._te_sample_pred_var, self._run, log_name='test',  exp_name=self._exp_name,tar_type="observations", tar_obs=self._te_sample_tar_obs, normalizer=None, loss=self._loss)


            if (self._tar_type == "observations" and  self._loss == "kl_rmse"):

                print("plotImputation with observation and  kl_rmse")

                plotImputation(self._tr_sample_gt, self._tr_sample_valid, self._tr_sample_pred_mu,self._tr_sample_pred_var, self._run, log_name='train', exp_name=self._exp_name,tar_type="observations", tar_obs=self._tr_sample_tar_obs, normalizer=self._normalizer, loss=self._loss)
                plotImputation(self._te_sample_gt, self._te_sample_valid, self._te_sample_pred_mu,self._te_sample_pred_var, self._run, log_name='test',  exp_name=self._exp_name,tar_type="observations", tar_obs=self._te_sample_tar_obs, normalizer=self._normalizer, loss=self._loss)
