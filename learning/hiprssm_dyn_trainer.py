import os
import time as t
from typing import Tuple
import datetime

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import wandb

from meta_dynamic_models.neural_process_dynamics.npDynamics import npDyn
from utils.dataProcess import split_k_m, get_ctx_target_impute
from utils.Losses import mse, mae, gaussian_nll , CrossEntropy
from utils.PositionEmbedding import PositionEmbedding as pe
from utils import ConfigDict
from utils.plotTrajectory import plotImputation
from utils.latentVis import plot_clustering

optim = torch.optim
nn = torch.nn

import random
import torch





def avg_pred(scores):
    '''
    scores: 2D matrices of batch_size*classes. each rows contains the score of each class
    return: convert the scores to probability using softmax and then take expected value
    '''
    scores  = scores.float()
    softmax = nn.Softmax(dim=-1)
    probs   = softmax(scores)
    classes = torch.arange(scores.size()[-1]).float()
    avg     = torch.matmul(probs,classes)
    return torch.unsqueeze(avg,dim=-1)
#avg_pred(torch.tensor([[0, 0, 0, 5, 5],[5, 5, 0, 0, 0],[1, 0, 5, 0, 0]]))


def k_max_class(scores2, k_ind=5):
    '''
    receive raw scores and take the k_ind max of them and replace the rest with  "-Inf"
    scores2: 2D matrices of batch_size*classes. each rows contains the score of each class
    k_ind: how many of the scores should be considered (rest of scores will be replaced by -inf to have prob of 0)

    '''

    vals_max, inds = torch.topk(scores2,k_ind)  # inds in each rows show the first k_ind max valuse #we dont need vals_max
    one_hot = nn.functional.one_hot(inds, num_classes=scores2.size()[-1])
    to_be_taken_inds = torch.sum(one_hot, dim=-2)  # to have all ones (to be take inds) in one vector
    cut_scores = torch.mul(scores2, to_be_taken_inds).float()  #
    cut_scores[cut_scores == 0] = float("-Inf")
    return cut_scores


# a = torch.tensor([[0, 0, 0, 5, 5], [5, 5, 0, 0, 0], [1, 0, 5, 0, 0]])
# new_score = k_max_class(a)
# print(new_score)
# print(avg_pred(new_score))


class Learn:

    def __init__(self, model: npDyn, loss: str, imp: float = 0.0, config: ConfigDict = None, run = None, log=True, use_cuda_if_available: bool = True  ,normalizer=None):
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
        print("trainer line 51:    self._exp_name = ",self._exp_name)
        if config is None:
            raise TypeError('Pass a Config Dict')
        else:
            self.c = config

        self._tar_type=self.c.data_reader.tar_type   #saleh_added
        print("hiprssm_dyn_trainer.py line 44 ")
        print("self._tar_type = ", self._tar_type)
        self._normalizer = normalizer

        self._learning_rate = self.c.learn.lr

        self._save_path = os.getcwd() + '/experiments/saved_models/' + run.name + '.ckpt'
        self._cluster_vis = self.c.learn.latent_vis

        self._optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate)
        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches
        self._log = bool(log)
        self.save_model = self.c.learn.save_model
        if self._log:
            self._run = run

    def train_step(self, train_obs: np.ndarray, train_act: np.ndarray, train_targets: np.ndarray, train_task_idx: np.ndarray,
                   batch_size: int)  -> Tuple[float, float, float]:
        """
        Train once on the entire dataset
        :param train_obs: training observations
        :param train_act: training actions
        :param train_targets: training targets
        :param train_task_idx: task ids per episode
        :param batch_size: batch size for each gradient update
        :return: average loss (nll) and  average metric (rmse), execution time
        """

        # def my_collate(batch):  # batch size 4 [{tensor image, tensor label},{},{},{}] could return something like G = [None, {},{},{}]
        #     #batch = list(filter(lambda x: torch.sum(x,dim), batch))
        #     bad_idx=[]
        #     print("len_batch = ",len(batch))
        #     for idx, (obs, act, targets, task_id) in enumerate(batch):
        #         #batch = list(filter(lambda x: not(all(x[...,:,-1]==0)),batch))  # this gets rid of nones in batch. For example above it would result to G = [{},{},{}]
        #         #print(targets.shape)
        #         if(torch.sum(targets)==0):
        #             bad_idx.append(idx)
        #     print(bad_idx)
        #     return torch.utils.data.dataloader.default_collate(batch)



        self._model.train()
        dataset = TensorDataset(train_obs, train_act, train_targets, train_task_idx)
        #loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2  , collate_fn = my_collate)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        avg_loss = avg_metric_nll = avg_metric_mse = avg_metric_mae= avg_metric_CrossEntropy  = 0

        t0 = t.time()
        #b = list(loader)[0]
        z_vis_list = []
        task_id_list = []

        print("line 76 hiprssm_dyn_trainer.py : Device = ", self._device)

        for batch_idx, (obs, act, targets, task_id) in enumerate(loader):
            # Assign tensors to device
            #print("batch_idx:",batch_idx)
            obs_batch = (obs).to(self._device)

            act_batch = act.to(self._device)
            #print("batch_idx:",batch_idx)

            target_batch = (targets).to(self._device)
            task_id = (task_id).to(self._device)
            #print("batch_idx:",batch_idx)

            # Split to context and targets
            k = int(obs_batch.shape[1] / 2)
            m = obs_batch.shape[1] - k
            #print("batch_idx:",batch_idx)


            #print("hiprssm_dyn_trainer line 96:")
            #print("act_batch",act_batch) #checked all 0
            act_batch = torch.zeros_like(act_batch)


            ctx_obs_batch, ctx_act_batch, ctx_target_batch, tar_obs_batch, tar_act_batch, tar_tar_batch, tar_obs_valid_batch = \
                get_ctx_target_impute(obs_batch, act_batch, target_batch, k, num_context=None, tar_imp=self._imp,
                                      random_seed=True)

            #print("hiprssm_dyn_trainer line 105 to 109:")
            #print("ctx_act_batch:", ctx_act_batch) #checked all 0
            #print("tar_act_batch:", tar_act_batch) #checked all 0
            ctx_act_batch = torch.zeros_like(ctx_act_batch)
            tar_act_batch = torch.zeros_like(tar_act_batch)


            #print("tar_tar_batch:", tar_tar_batch)

            #print("batch_idx:",batch_idx)
            if batch_idx == 0:
                print("Fraction of Valid Target Observations:",
                      np.count_nonzero(tar_obs_valid_batch) / np.prod(tar_obs_valid_batch.shape))

            ctx_obs_valid_batch = torch.ones(ctx_obs_batch.shape[0], ctx_obs_batch.shape[1],1)
            ctx_obs_valid_batch = ctx_obs_valid_batch.bool().to(self._device)


            tar_obs_valid_batch = torch.from_numpy(tar_obs_valid_batch).bool().to(self._device)
            context_Y = ctx_target_batch
            target_X = (tar_obs_batch, tar_act_batch, tar_obs_valid_batch)

            context_X = torch.cat([ctx_obs_batch, ctx_act_batch], dim=-1)


            # Set Optimizer to Zero
            self._optimizer.zero_grad()
            #print("optimizer.zero_grad()")

            # Forward Pass
           #print("line 135 hiprssm_dyn_trainer.py Forward Pass (self._model(context_X, context_Y, target_X) )started...")
            out_mean, out_var, mu_z, cov_z = self._model(context_X, context_Y, target_X)
            #print("self._model(context_X, context_Y, target_X)")

            ## Calculate Loss
            #if self._loss == 'crossentrop': here update the code




            if self._loss == 'nll':
                loss = gaussian_nll(tar_tar_batch, out_mean, out_var)
            elif self._loss == 'mae':
                loss = mae(tar_tar_batch, out_mean)
            elif self._loss == 'mse':
                loss = mse(tar_tar_batch, out_mean)
            elif self._loss == 'cross_entropy':
                #print("!!!PLEASE DOUBLE CHECK INPUT AND TARGET") #done
                loss = CrossEntropy(out_mean , tar_tar_batch.squeeze() , C=20, manual=False)
                # print("loss = ",loss)
                # loss2 = CrossEntropy(out_mean , tar_tar_batch , C=20, manual=True)  #Double checked always the same
                # print("loss2 = ", loss2)

            else:
                raise NotImplementedError


            # Backward Pass
            loss.backward()
            #print("hiprssm_dyn_trainer.py line 146 loss.backward()")

            # Clip Gradients
            if self.c.np.clip_gradients:
                torch.nn.utils.clip_grad_norm(self._model.parameters(), 5.0)

            # Backward Pass Via Optimizer
            self._optimizer.step()
            #print("optimizer.step()")



            #print("hiprssm_dyn_trainer.py line 242  newly added")
            scores = k_max_class(out_mean) #newly added
            k_averaged = avg_pred(scores) #newly added

            out_mean_pred = out_mean
            if (self._loss == 'cross_entropy'):
                out_mean_pred = torch.argmax(out_mean, dim=-1).unsqueeze(dim=-1)  #yek adad beyne 0 ta 19 bar migardune baraye dim e -1

            with torch.no_grad():  #
                #metric_nll = gaussian_nll(tar_tar_batch, out_mean_pred, out_var)
                metric_nll = gaussian_nll(tar_tar_batch, k_averaged, out_var)
                #metric_mse = mse(tar_tar_batch, out_mean_pred)
                metric_mse = mse(tar_tar_batch, k_averaged)
                #metric_mae = mae(tar_tar_batch, out_mean_pred)
                metric_mae = mae(tar_tar_batch, k_averaged)
                metric_CrossEntropy = CrossEntropy( out_mean,tar_tar_batch,C=20,manual=False)

                z_vis_list.append(mu_z.detach().cpu().numpy())
                task_id_list.append(task_id.detach().cpu().numpy())
            avg_loss += loss.detach().cpu().numpy()
            avg_metric_nll += metric_nll.detach().cpu().numpy()
            avg_metric_mse += metric_mse.detach().cpu().numpy()
            avg_metric_mae += metric_mae.detach().cpu().numpy()
            avg_metric_CrossEntropy += metric_CrossEntropy.detach().cpu().numpy()
        #print("loss")
        # taking sqrt of final avg_mse gives us rmse across an epoch without being sensitive to batch size
        if self._loss == 'nll':
            avg_loss = avg_loss / len(list(loader))


        elif self._loss == 'mse':
            avg_loss = np.sqrt(avg_loss / len(list(loader)))

        elif self._loss == 'mae':
            avg_loss = avg_loss / len(list(loader))

        elif self._loss == 'cross_entropy':
            avg_loss = (avg_loss / len(list(loader)))

        else:
            raise NotImplementedError



        with torch.no_grad():
            self._tr_sample_gt = tar_tar_batch.detach().cpu().numpy()
            self._tr_sample_valid = tar_obs_valid_batch.detach().cpu().numpy()
            self._tr_sample_pred_mu = k_averaged.detach().cpu().numpy()
            self._tr_sample_pred_var = out_var.detach().cpu().numpy()

            self._tr_sample_tar_obs = tar_obs_batch.detach().cpu().numpy()[:,:,-1:]   # saleh_added

        avg_metric_nll = avg_metric_nll / len(list(loader))
        avg_metric_rmse = np.sqrt(avg_metric_mse / len(list(loader)))
        avg_metric_mae = avg_metric_mae / len(list(loader))
        avg_metric_CrossEntropy = avg_metric_CrossEntropy / len(list(loader))

        #z_vis = 0
        z_vis = np.concatenate(z_vis_list, axis=0)
        task_labels = np.concatenate(task_id_list, axis=0)
        return avg_loss, avg_metric_nll, avg_metric_rmse,avg_metric_mae,avg_metric_CrossEntropy , z_vis, task_labels, t.time() - t0

    def eval(self, obs: np.ndarray, act: np.ndarray, targets: np.ndarray, task_idx: np.ndarray,
             batch_size: int = -1) -> Tuple[float, float]:
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
        #print("saleh_added: hiprssm_dyn_trainer line 199  act:",act)   #all 0 checked #saleh_added
        #print("type(act)=", type(act)) #torch
        act = torch.zeros_like(act)
        #print("type(act)=",type(act))
        dataset = TensorDataset(obs, act, targets, task_idx)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        avg_loss = avg_metric_nll = avg_metric_mse = avg_metric_mae= avg_metric_CrossEntropy = 0.0
        avg_metric = 0.0
        z_vis_list = []
        task_id_list = []

        for batch_idx, (obs_batch, act_batch, targets_batch, task_id) in enumerate(loader):
            with torch.no_grad():
                # Assign tensors to devices

                #print("saleh_added: hiprssm_dyn_trainer line 196  act_batch:", act_batch)  # saleh_added #checked all_zero
                act_batch = torch.zeros_like(act_batch)

                obs_batch = (obs_batch).to(self._device)
                act_batch = act_batch.to(self._device)
                target_batch = (targets_batch).to(self._device)

                # Split to context and targets
                k = int(obs_batch.shape[1] / 2)
                m = obs_batch.shape[1] - k

                ctx_obs_batch, ctx_act_batch, ctx_target_batch, tar_obs_batch, tar_act_batch, tar_tar_batch, tar_obs_valid_batch = \
                    get_ctx_target_impute(obs_batch, act_batch, target_batch, k, num_context=None, tar_imp=self._imp,
                                          random_seed=True)
                #print("saleh_added hiprssm_dyn_trainer.py line 210      ctx_act_batch:",ctx_act_batch)#saleh_added #checked all 0
                ctx_act_batch = torch.zeros_like(ctx_act_batch) #saleh_added

                #print("saleh_added hiprssm_dyn_trainer.py line 210      tar_act_batch:",tar_act_batch) #saleh_added #checked all 0
                tar_act_batch = torch.zeros_like(tar_act_batch) #saleh_added


                ctx_obs_valid_batch = torch.ones(ctx_obs_batch.shape[0],ctx_obs_batch.shape[1],1)
                ctx_obs_valid_batch = ctx_obs_valid_batch.bool().to(self._device)

                tar_obs_valid_batch = torch.from_numpy(tar_obs_valid_batch).bool().to(self._device)


                # Make context and target
                target_X = (tar_obs_batch, tar_act_batch, tar_obs_valid_batch)

                #print("hiprssm_dyn_trainer.py line 244",)
                #print('target_x"',target_X)

                context_X = torch.cat([ctx_obs_batch, ctx_act_batch], dim=-1)
                context_Y = ctx_target_batch


                # Forward Pass
                out_mean, out_var, mu_z, cov_z = self._model(context_X, context_Y, target_X)

                #print("hiprssm_dyn_trainer.py line 372  newly added")
                scores = k_max_class(out_mean)  # newly added
                k_averaged = avg_pred(scores)  # newly added

                out_mean_pred = out_mean
                if (self._loss == 'cross_entropy'):
                    out_mean_pred = torch.argmax(out_mean, dim=-1).unsqueeze(dim=-1)  # yek adad beyne 0 ta 19 bar migardune

                self._te_sample_gt       = tar_tar_batch.detach().cpu().numpy()
                self._te_sample_valid    = tar_obs_valid_batch.detach().cpu().numpy()
                #self._te_sample_pred_mu  = out_mean_pred.detach().cpu().numpy()
                self._te_sample_pred_mu = k_averaged.detach().cpu().numpy()
                self._te_sample_pred_var = out_var.detach().cpu().numpy()

                self._te_sample_tar_obs  = tar_obs_batch.detach().cpu().numpy()[:,:,-1:]  #added_by saleh


                ## Calculate Loss
                if self._loss == 'nll':
                    loss = gaussian_nll(tar_tar_batch, out_mean_pred, out_var)

                elif self._loss == 'mse':
                    loss = mse(tar_tar_batch, out_mean_pred)

                elif self._loss == 'mae':
                    loss = mae(tar_tar_batch, out_mean_pred)

                elif self._loss == 'cross_entropy':   # target(GT) should come 2nd
                    loss = CrossEntropy(out_mean,tar_tar_batch  ,C=20 , manual=False)

                else:
                    raise NotImplementedError

                #metric_nll = gaussian_nll(tar_tar_batch, out_mean_pred, out_var)
                metric_nll = gaussian_nll(tar_tar_batch, k_averaged, out_var)

                #metric_mse = mse(tar_tar_batch, out_mean_pred)
                metric_mse = mse(tar_tar_batch, k_averaged)

                #metric_mae = mae(tar_tar_batch, out_mean_pred)
                metric_mae = mae(tar_tar_batch, k_averaged)
                metric_CrossEntropy = CrossEntropy(out_mean,tar_tar_batch,C=20,manual=False)

#
                z_vis_list.append(mu_z.detach().cpu().numpy())
                task_id_list.append(task_id.detach().cpu().numpy())

                avg_loss += loss.detach().cpu().numpy()
                avg_metric_nll += metric_nll.detach().cpu().numpy()
                avg_metric_mse += metric_mse.detach().cpu().numpy()
                avg_metric_mae += metric_mae.detach().cpu().numpy()
                avg_metric_CrossEntropy += metric_CrossEntropy.detach().cpu().numpy()

        # taking sqrt of final avg_mse gives us rmse across an apoch without being sensitive to batch size
        print("line 337 dyn_trainer.py   self._loss=",self._loss)
        if self._loss == 'nll': #true
            avg_loss = avg_loss / len(list(loader))

        elif self._loss == 'mse': #true
            avg_loss = np.sqrt(avg_loss / len(list(loader)))

        elif self._loss == 'mae': #true
            avg_loss = avg_loss / len(list(loader))

        elif self._loss == 'cross_entropy': #true
            avg_loss = avg_loss / len(list(loader))

        else:
            raise NotImplementedError

        avg_metric_nll = avg_metric_nll / len(list(loader))
        avg_metric_rmse = np.sqrt(avg_metric_mse / len(list(loader)))
        avg_metric_mae =(avg_metric_mae / len(list(loader)))
        avg_metric_CrossEntropy = (avg_metric_CrossEntropy / len(list(loader)))

        z_vis = np.concatenate(z_vis_list, axis=0)
        #z_vis = 0
        task_labels = np.concatenate(task_id_list, axis=0)
        return avg_loss, avg_metric_nll, avg_metric_rmse, avg_metric_mae, avg_metric_CrossEntropy ,z_vis, task_labels

    def train(self, train_obs: torch.Tensor, train_act: torch.Tensor,
              train_targets: torch.Tensor, train_task_idx: torch.Tensor, epochs: int, batch_size: int,
              val_obs: torch.Tensor = None, val_act: torch.Tensor = None,
              val_targets: torch.Tensor = None, val_task_idx: torch.Tensor = None, val_interval: int = 1,
              val_batch_size: int = -1) -> None:
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
        best_mae = np.inf
        best_CrossEntropy = np.inf

        #print("train_act:", train_act) #all 0 checked # saleh_added
        #print("saleh_added hiprssm_dyn_trainer.py line 318")  # saleh_added
        train_act = torch.zeros_like(train_act)  # saleh_added

        #print("val_act:", val_act) #all 0 checked  # saleh_added
        #print("saleh_added hiprssm_dyn_trainer.py line 322")  # saleh_added
        val_act = torch.zeros_like(val_act)  # saleh_added


        if self._log:
            wandb.watch(self._model, log='all')
            artifact = wandb.Artifact('saved_model', type='model')

        print("===========================...epoching...=================================")
        for i in range(epochs):
            print("===================================================================================")
            print("epochs = :",i , "/",epochs)
            if i == 0:
                print('<<<<<<<<<<<<Set Encoder:>>>>>>>>>>>>', self.c.set_encoder)
                print('<<<<<<<<<<<<Neural Process:>>>>>>>>>>>>', self.c.np)
                print('<<<<<<<<<<<<SSM:>>>>>>>>>>>>', self.c.ssm_decoder)




            train_loss, train_metric_nll, train_metric_rmse,train_metric_mae, train_metric_CrossEntropy ,z_vis, z_labels, time = self.train_step(train_obs,
                                                                                                     train_act,
                                                                                                     train_targets,
                                                                                                     train_task_idx,
                                                                                                     batch_size)
            print("Training Iteration {:04d}: {}:{:.5f}, {}:{:.5f}, {}:{:.5f}, {}:{:.5f}, {}:{:.5f}, Took {:4f} seconds".format(i + 1, self._loss, train_loss, 'target_nll:', train_metric_nll, 'target_rmse:', train_metric_rmse, 'target_mae:', train_metric_mae, 'target_CrossEntropy:', train_metric_CrossEntropy,time))
            # self._writer.add_scalar(self._loss + "/train_loss", train_loss, i)
            # self._writer.add_scalar("nll/train_metric", train_metric_nll, i)
            # self._writer.add_scalar("rmse/train_metric", train_metric_rmse, i)
            if self._log:
                wandb.log({self._loss + "/train_loss": train_loss, "nll/train_metric": train_metric_nll, "rmse/train_metric": train_metric_rmse, "mae/train_metric": train_metric_mae, "CrossEntropy/train_metric": train_metric_CrossEntropy, "epochs": i})


            # print("val_act:", val_act)   #saleh_added
            # print("val_targets:", val_targets)  # saleh_added
            # print("saleh_added hiprssm_dyn_trainer.py line 316")#saleh_added
            # val_act = torch.zeros_like(val_act)     #saleh_added
            # val_targets = torch.zeros_like(val_targets)  # saleh_added
            if val_obs is not None and val_targets is not None and i % val_interval == 0:
                val_loss, val_metric_nll, val_metric_rmse,val_metric_mae, val_metric_CrossEntropy ,z_vis_val, z_labels_val = self.eval(val_obs, val_act,val_targets,val_task_idx, batch_size=val_batch_size)
                if val_loss < best_loss:
                    if self.save_model:
                        print('>>>>>>>Saving Best Model<<<<<<<<<<')
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

                if val_metric_mae < best_mae:
                    if self._log:
                        wandb.run.summary['best_mae'] = val_metric_mae
                    best_mae = val_metric_mae

                if val_metric_CrossEntropy < best_CrossEntropy:
                    if self._log:
                        wandb.run.summary['best_CrossEntropy'] = val_metric_CrossEntropy
                    best_CrossEntropy = val_metric_CrossEntropy



                print("Validation: {}: {:.5f}, {}: {:.5f}, {}: {:.5f}, {}: {:.5f}, {}: {:.5f}".format(self._loss, val_loss, 'target_nll',val_metric_nll, 'target_rmse', val_metric_rmse , 'target_mae', val_metric_mae, 'target_CrossEntropy', val_metric_CrossEntropy))

                if self._log:
                    wandb.log({self._loss + "/val_loss": val_loss, "nll/test_metric": val_metric_nll, "rmse/test_metric": val_metric_rmse, "mae/test_metric": val_metric_mae, "CrossEntropy/test_metric": val_metric_CrossEntropy, "epochs": i})

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
            print('plotimp() from hiprssm_dyn_trainer.py line 416')

            if(self._tar_type=="delta"):
                print("plotImputation with delta")      #                                                                                                                                        #the lasst 2 param are used to convert from diff to actual
                plotImputation(self._tr_sample_gt, self._tr_sample_valid, self._tr_sample_pred_mu, self._tr_sample_pred_var,self._run, log_name='train', exp_name=self._exp_name ,tar_type='delta',tar_obs=self._tr_sample_tar_obs ,normalizer=self._normalizer)
                plotImputation(self._te_sample_gt, self._te_sample_valid, self._te_sample_pred_mu, self._te_sample_pred_var,self._run, log_name='test',  exp_name=self._exp_name, tar_type='delta',tar_obs=self._te_sample_tar_obs, normalizer=self._normalizer)

                #plotImputation(self._tr_sample_gt, self._tr_sample_valid, self._tr_sample_pred_mu, self._tr_sample_pred_var,self._run, log_name='train_delta', exp_name=self._exp_name ,tar_type=None,tar_obs=None)
                #plotImputation(self._te_sample_gt, self._te_sample_valid, self._te_sample_pred_mu, self._te_sample_pred_var,self._run, log_name='test_delta',  exp_name=self._exp_name, tar_type=None,tar_obs=None)

            if (self._tar_type == "observations" and self._loss!="cross_entropy"):
                print("plotImputation with observation")

                plotImputation(self._tr_sample_gt, self._tr_sample_valid, self._tr_sample_pred_mu,self._tr_sample_pred_var, self._run, log_name='train', exp_name=self._exp_name,tar_type="observations", tar_obs=self._tr_sample_tar_obs,normalizer=self._normalizer)
                plotImputation(self._te_sample_gt, self._te_sample_valid, self._te_sample_pred_mu,self._te_sample_pred_var, self._run, log_name='test', exp_name=self._exp_name ,tar_type="observations" , tar_obs=self._te_sample_tar_obs,normalizer=self._normalizer)

            if (self._tar_type == "observations" and self._loss == "cross_entropy"):

                print("plotImputation with observation and CrossEntropy")

                plotImputation(self._tr_sample_gt, self._tr_sample_valid, self._tr_sample_pred_mu,
                               self._tr_sample_pred_var, self._run, log_name='train', exp_name=self._exp_name,
                               tar_type="observations", tar_obs=self._tr_sample_tar_obs, normalizer=None)
                plotImputation(self._te_sample_gt, self._te_sample_valid, self._te_sample_pred_mu,
                               self._te_sample_pred_var, self._run, log_name='test', exp_name=self._exp_name,
                               tar_type="observations", tar_obs=self._te_sample_tar_obs, normalizer=None)
