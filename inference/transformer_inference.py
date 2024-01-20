from typing import Tuple
import numpy as np
import torch
from meta_dynamic_models.neural_process_dynamics.npDynamics import npDyn
from torch.utils.data import TensorDataset, DataLoader
from utils.dataProcess import split_k_m, get_sliding_context_batch_mbrl, get_ctx_target_multistep, get_ctx_target_impute,\
    squeeze_sw_batch, diffToStateMultiStep, diffToState, diffToStateImpute, norm, denorm
from utils import ConfigDict

optim = torch.optim
nn = torch.nn
class Infer:
    def __init__(self, model, config:ConfigDict = None, run = None, log=True, use_cuda_if_available: bool = True, normalizer=None):

        """
        :param model: nn module for HiP-RSSM
        :param use_cuda_if_available:  if to use gpu
        """
        #print("transformer_inference.py line 19,   object is created")
        assert run is not None, 'Enter a valid wandb run'
        self._device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        self._model = model
        self._data = norm
        self._normalizer = normalizer

        if config is None:
            raise TypeError('Pass a Config Dict')
        else:
            self.c = config

        self._loss = config.learn.loss
        self._shuffle_rng = np.random.RandomState(0)  # rng for shuffling batches
        self._log = log
        if self._log:
            self._run = run

        self.pred_len     = self.c.data_reader.pred_len
        self.context_size = self.c.data_reader.context_size

    @torch.no_grad()
    def predict(self, obs: torch.Tensor, act: torch.Tensor, y_context: torch.Tensor, task_idx: torch.Tensor, imp: float = 0.0, k=32, test_gt_known=True,
                batch_size: int = -1, multiStep=0, tar="observations") -> Tuple[float, float]:
        '''
        :param obs: observations to evaluate on
        :param act: actions to evaluate on
        :param y_context: the label information for the context sets
        :param imp: imputation
        :param test_gt_known: if ground truth known
        :param batch_size: batch_size for evaluation, this does not change the results and is only to allow evaluating on more
         data than you can fit in memory at once. Default: -1, .i.e. batch_size = number of sequences.
        :param multiStep: number of multi step ahead predictions
        :tar: "delta" or "observations"
        :return:
        '''
        # rescale only batches so the data can be kept in unit8 to lower memory consumptions
        self._model = self._model.eval()
        cur_obs_list = []
        out_mean_list = []
        out_var_list = []
        gt_list = []
        obs_valid_list = []
        z_vis_list = []
        task_id_list = []
        self._context_size = k
        dataset = TensorDataset(obs, act, y_context, task_idx)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        for batch_idx, (obs, act, target, task_id) in enumerate(loader):
            with torch.no_grad():
                # Assign data tensors to devices
                if(np.mod(batch_idx,50)==0):
                    print("transformer_inference line 70  predict func is called   ,","batch_idx:",batch_idx,"/",str(len(loader)))

                obs_batch = (obs).to(self._device)
                act_batch = act.to(self._device)
                target_batch = (target).to(self._device)

                # Split to context and targets
                if self._context_size is None:
                    k = int(obs_batch.shape[1] / 2)
                else:
                    k = self._context_size
                m = obs_batch.shape[1] - k
                ctx_obs_batch, ctx_act_batch, ctx_target_batch, tar_obs_batch, tar_act_batch, tar_tar_batch, tar_obs_valid_batch = get_ctx_target_impute(obs_batch, act_batch, target_batch, k, num_context=None, test_gt_known= test_gt_known,tar_imp=imp,random_seed=True)
                ctx_obs_valid_batch = torch.ones(ctx_obs_batch.shape[0], ctx_obs_batch.shape[1], 1)
                ctx_obs_valid_batch = ctx_obs_valid_batch.bool().to(self._device)

                ### Unlike in learning during inference we don't have access to Y_target
                context_y = ctx_target_batch
                tar_obs_valid_batch = torch.from_numpy(tar_obs_valid_batch).bool().to(self._device)
                #target_x = (tar_obs_batch, tar_act_batch, tar_obs_valid_batch)


                #context_x = torch.cat([ctx_obs_batch, ctx_act_batch], dim=-1)

                X = torch.cat([ctx_obs_batch, tar_obs_batch], dim=1).to(self._device)
                Y = tar_tar_batch.to(self._device)

                # Forward Pass
                #out_mean, out_var, mu_z, cov_z = self._model(context_x, context_y, target_x)
                pred_logits, _ = self._model(X)
                pred_logits = pred_logits[:, k:, :]  # more interested in the 2nd half


                # if len(mu_z.shape) < 2:
                #     mu_z = torch.unsqueeze(mu_z,0)

                # Diff To State
                if tar == "delta":
                    #when predicting differences convert back to actual observations (common procedure in model based RL and dynamics learning)
                    out_mean = \
                        torch.from_numpy(
                            diffToStateImpute(out_mean, tar_obs_batch[:,:,-1:], tar_obs_valid_batch, self._data.normalization,standardize=True)[0])
                    tar_tar_batch = \
                        torch.from_numpy(diffToState(tar_tar_batch, tar_obs_batch[:,:,-1:], self._data.normalization, standardize=True)[0])

                #out_mean_list.append(out_mean.cpu())
                out_mean_list.append(pred_logits.cpu())


                #out_var_list.append(out_var.cpu())



                #gt_list.append(tar_tar_batch.cpu())  # if test_gt_known flag is False then we get list of Nones
                gt_list.append(Y.cpu())



                #z_vis_list.append(mu_z.detach().cpu().numpy())
                #task_id_list.append(task_id.detach().cpu().numpy())
                obs_valid_list.append(tar_obs_valid_batch.cpu())
                cur_obs_list.append(tar_obs_batch.cpu())

        #z_vis = np.concatenate(z_vis_list, axis=0)
        # z_vis = 0
        #task_labels = np.concatenate(task_id_list, axis=0)
        #return torch.cat(out_mean_list), torch.cat(out_var_list), torch.cat(gt_list), torch.cat(obs_valid_list), z_vis, task_labels, torch.cat(cur_obs_list)

        return torch.cat(out_mean_list), np.nan, torch.cat(gt_list), torch.cat(obs_valid_list), np.nan, np.nan, torch.cat(cur_obs_list)

    # @torch.no_grad()
    # def predict_multiStep(self, obs: torch.Tensor, act: torch.Tensor, y_context: torch.Tensor, k=32, test_gt_known=True,
    #             batch_size: int = -1, multiStep=0, tar="observations") -> Tuple[float, float]:
    #     """
    #     Predict using the model
    #     :param obs: observations to evaluate on
    #     :param act: actions to evaluate on
    #     :param y_context: the label information for the context sets
    #     :param batch_size: batch_size for evaluation, this does not change the results and is only to allow evaluating on more
    #      data than you can fit in memory at once. Default: -1, .i.e. batch_size = number of sequences.
    #     :param multiStep: how many multiStep ahead predictions do you need. You can also do this by playing with obs_valid flag.
    #     """
    #     # rescale only batches so the data can be kept in unit8 to lower memory consumptions
    #     self._model = self._model.eval()
    #     self._context_size = k
    #     out_mean_list = []
    #     out_var_list = []
    #     gt_list = []
    #     dataset = TensorDataset(obs, act, y_context)
    #     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    #
    #
    #     for batch_idx, (obs_batch, act_batch, targets_batch) in enumerate(loader):
    #         with torch.no_grad():
    #             print("transformer_inference line 148  predict_multistep func is called")
    #             print("batch_idx:",batch_idx)
    #             # Assign tensors to devices
    #             obs_batch = (obs_batch).to(self._device)
    #             act_batch = act_batch.to(self._device)
    #             target_batch = (targets_batch).to(self._device)
    #
    #             # Split to context and targets
    #             if self._context_size is None:
    #                 k = int(obs_batch.shape[1] / 2)
    #             else:
    #                 k = self._context_size
    #             m = obs_batch.shape[1] - k
    #             ctx_obs_batch, ctx_act_batch, ctx_target_batch, tar_obs_batch, tar_act_batch, tar_tar_batch, tar_obs_valid_batch = get_ctx_target_multistep(obs_batch, act_batch, target_batch, k, num_context=None, test_gt_known=test_gt_known, tar_burn_in=5,random_seed=True) #everything from tar_burn_in set to False
    #             ctx_obs_valid_batch = torch.ones(ctx_obs_batch.shape[0], ctx_obs_batch.shape[1], 1)
    #             ctx_obs_valid_batch = ctx_obs_valid_batch.bool().to(self._device)
    #             tar_obs_valid_batch = torch.from_numpy(tar_obs_valid_batch).bool().to(self._device)
    #
    #
    #             # context_Y = ctx_target_batch
    #             # target_X = (tar_obs_batch, tar_act_batch, tar_obs_valid_batch)
    #             #
    #             #
    #             # context_X = torch.cat([ctx_obs_batch, ctx_act_batch], dim=-1)
    #             X = torch.cat([ctx_obs_batch, tar_obs_batch], dim=1).to(self._device)
    #             Y = tar_tar_batch.to(self._device)
    #
    #             # Forward Pass
    #
    #
    #             # Forward Pass
    #             pred_logits, _ = self._model(X)
    #             pred_logits = pred_logits[:, k:, :]  # more interested in the 2nd half
    #
    #             # Diff To State
    #             if tar=="delta":
    #             # when predicting differences convert back to actual observations (common procedure in model based RL and dynamics learning)
    #                 out_mean = diffToStateMultiStep(out_mean, tar_obs_batch, self._data, standardize=True)[0]
    #                 tar_tar_batch = diffToStateMultiStep(tar_obs_batch, tar_obs_batch, self._data, standardize=True)[0]
    #
    #             #out_mean_list.append(out_mean.cpu())
    #             out_mean_list.append(pred_logits.cpu())
    #
    #
    #             #out_var_list.append(out_var.cpu()) ## Need to Change later for differences #TODO
    #             gt_list.append(tar_tar_batch.cpu()) #if test_gt_known flag is False then we get list of Nones
    #
    #
    #     #return torch.cat(out_mean_list), torch.cat(out_var_list), torch.cat(gt_list)
    #     return torch.cat(out_mean_list), np.nan, torch.cat(gt_list)
    @torch.no_grad()
    def predict_mbrl(self, inner_dict: dict, k=32, batch_size: int = -1, tar="packets") -> Tuple[float, float]:
        """
        Predict using the model
        :param obs: observations to evaluate on
        :param act: actions to evaluate on
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
        observed_list =[]
        residual_list = []


        test_obs = inner_dict
        dataset = TensorDataset(torch.from_numpy(test_obs))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        for batch_idx, (obs_batch) in enumerate(loader):
            # if(np.mod(batch_idx,50) ==0): #dataloader batch haye 10 taee misazad az 3500(Meta_batch_size) ta
            #     print("transformer_inference line 148  predict_mbrl func is called","   ,batch_idx:", batch_idx,"/",str(len(loader)))
            # Assign tensors to devices
            obs_batch = (obs_batch[0]).to(self._device)        #[10,150,25]
            #act_batch = act_batch.to(self._device)          #[10,150,25]
            #target_batch = (target_batch).to(self._device)  #[10,150,25]
            with torch.no_grad():
                # Split to context and targets
                if self.context_size is None:
                    k = int(obs_batch.shape[1] / 2)
                else: # true
                    k = self.context_size  #75
                m = self.pred_len #150-75=75


                #ctx_obs_batch, ctx_act_batch, ctx_target_batch, tar_obs_batch, tar_act_batch, tar_tar_batch, tar_obs_valid_batch = get_ctx_target_impute(obs_batch, act_batch, target_batch, k, num_context=None, random_seed=True)
                ctx_obs_batch, tar_obs_batch = obs_batch[:, :self.context_size, :].float(), obs_batch[:, self.context_size:, -1:].float()
                Y = tar_obs_batch.to(self._device)
                X_enc = ctx_obs_batch.to(self._device)


                # Forward Pass
                with torch.no_grad():
                    try:
                        pred_logits, _ = self._model(X_enc)
                    except:
                        pred_logits = self._model(X_enc)
                    # print("X_enc.shape:", X_enc.shape)                   # X.shape:       torch.Size([350, 75, 25])
                    # print("(GT) Y.shape:", Y.shape)                      # Y.shape:       torch.Size([350, 75, 1])
                    # print("pred_logits.shape", pred_logits.shape)        # Pred_logits:   torch.Size([350, 75, 1])
                    # print("------------------------------------")


                # Diff To actual conversion
                if tar == "delta": #dirt solution: out_dim = out_mean.shape[-1] ---> tar_obs_batch[:,:,-out_dim:]
                # when predicting differences convert back to actual observations (common procedure in model based RL and dynamics learning)
                    out_mean      = diffToStateMultiStep(out_mean, tar_obs_batch[:,:,-1:], tar_obs_valid_batch, self._data.normalization, standardize=True)[0]
                    tar_tar_batch = diffToState(tar_tar_batch, tar_obs_batch[:,:,-1:], self._data.normalization, standardize=True)[0]

                out_mean_list.append(pred_logits.cpu())
                #out_var_list.append(squeezed_var.cpu())
                gt_list.append(Y.cpu())
                residual_list.append(Y.cpu()-pred_logits.cpu())
                observed_list.append(ctx_obs_batch.cpu())
                #here some plots can be added

        #return torch.cat(out_mean_list), torch.cat(out_var_list), torch.cat(gt_list)
        return torch.cat(out_mean_list), np.nan, torch.cat(gt_list) , torch.cat(observed_list) , torch.cat(residual_list)
    # @torch.no_grad()
    # def predict_mbrl(self, inner_dict: dict, k=32, batch_size: int = -1, tar="packets") -> Tuple[float, float]:
    #     """
    #     Predict using the model
    #     :param obs: observations to evaluate on
    #     :param act: actions to evaluate on
    #     :param y_context: the label information for the context sets
    #     :param batch_size: batch_size for evaluation, this does not change the results and is only to allow evaluating on more
    #      data than you can fit in memory at once. Default: -1, .i.e. batch_size = number of sequences.
    #     :param multiStep: how many multiStep ahead predictions do you need. You can also do this by playing with obs_valid flag.
    #     """
    #     # rescale only batches so the data can be kept in unit8 to lower memory consumptions
    #     self._model = self._model.eval()
    #     out_mean_list = []
    #     out_var_list = []
    #     gt_list = []
    #     observed_list =[]
    #     window_with_residual_target_list = []
    #
    #     test_obs = inner_dict['vals']
    #     dataset = TensorDataset(torch.from_numpy(test_obs))
    #     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    #
    #     for batch_idx, (obs_batch) in enumerate(loader):
    #         # if(np.mod(batch_idx,50) ==0): #dataloader batch haye 10 taee misazad az 3500(Meta_batch_size) ta
    #         #     print("transformer_inference line 148  predict_mbrl func is called","   ,batch_idx:", batch_idx,"/",str(len(loader)))
    #         # Assign tensors to devices
    #         obs_batch = (obs_batch[0]).to(self._device)        #[10,150,25]
    #         #act_batch = act_batch.to(self._device)          #[10,150,25]
    #         #target_batch = (target_batch).to(self._device)  #[10,150,25]
    #         with torch.no_grad():
    #             # Split to context and targets
    #             if self.context_size is None:
    #                 k = int(obs_batch.shape[1] / 2)
    #             else: # true
    #                 k = self.context_size  #75
    #             m = self.pred_len #150-75=75
    #
    #
    #             #ctx_obs_batch, ctx_act_batch, ctx_target_batch, tar_obs_batch, tar_act_batch, tar_tar_batch, tar_obs_valid_batch = get_ctx_target_impute(obs_batch, act_batch, target_batch, k, num_context=None, random_seed=True)
    #             ctx_obs_batch, tar_obs_batch = obs_batch[:, :self.context_size, :].float(), obs_batch[:, self.context_size:, -1:].float()
    #             Y = tar_obs_batch.to(self._device)
    #             X_enc = ctx_obs_batch.to(self._device)
    #
    #
    #             # Forward Pass
    #             with torch.no_grad():
    #                 try:
    #                     pred_logits, _ = self._model(X_enc)
    #                 except:
    #                     pred_logits = self._model(X_enc)
    #                 # print("X_enc.shape:", X_enc.shape)                   # X.shape:       torch.Size([350, 75, 25])
    #                 # print("(GT) Y.shape:", Y.shape)                      # Y.shape:       torch.Size([350, 75, 1])
    #                 # print("pred_logits.shape", pred_logits.shape)        # Pred_logits:   torch.Size([350, 75, 1])
    #                 # print("------------------------------------")
    #
    #                 B, T1, input_feat = X_enc.shape
    #                 _, T2, _ = Y.shape
    #                 #C = self._num_class
    #
    #                 if (self._loss == 'cross_entropy'):
    #                     out_mean_pred = torch.argmax(pred_logits, dim=-1).unsqueeze(dim=-1).float()  # yek adad beyne 0 ta num_class bar migardune baraye dim e -1
    #                     out_mean_pred = (2 * out_mean_pred + 1 - self._normalizer[0]) / self._normalizer[1]  # convert back from bin_number to percentage Then normalize it
    #                     pred_logits = out_mean_pred # change the variable name to be compatible with the rest of the code
    #                     print("because of crossentropy: pred_logits.shape", pred_logits.shape)
    #
    #                 else:
    #                     pass
    #
    #
    #
    #
    #             # Diff To actual conversion
    #             if tar == "delta": #dirt solution: out_dim = out_mean.shape[-1] ---> tar_obs_batch[:,:,-out_dim:]
    #             # when predicting differences convert back to actual observations (common procedure in model based RL and dynamics learning)
    #                 out_mean      = diffToStateMultiStep(out_mean, tar_obs_batch[:,:,-1:], tar_obs_valid_batch, self._data.normalization, standardize=True)[0]
    #                 tar_tar_batch = diffToState(tar_tar_batch, tar_obs_batch[:,:,-1:], self._data.normalization, standardize=True)[0]
    #
    #             out_mean_list.append(pred_logits.cpu())
    #
    #             #out_var_list.append(squeezed_var.cpu())
    #             gt_list.append(Y.cpu())
    #             observed_list.append(ctx_obs_batch.cpu())
    #             res_y = Y.cpu() - pred_logits.cpu()
    #             window_with_residual_target_batch = torch.cat( ( ctx_obs_batch.cpu() , res_y.float()) ,dim=1) # [batch,49,1] , [batch,1,1]
    #             window_with_residual_target_list.append(window_with_residual_target_batch)
    #             #here some plots can be added
    #
    #     #return torch.cat(out_mean_list), torch.cat(out_var_list), torch.cat(gt_list)
    #     return torch.cat(out_mean_list), np.nan, torch.cat(gt_list) , torch.cat(observed_list) , torch.cat(window_with_residual_target_list)



        #       [3500,75,1]            ,   [3500,75,1]          ,[3500,75,1]

    @torch.no_grad()
    def predict_rw(self, test_obs: torch.Tensor, k=32, batch_size: int = -1, tar="observations") :
        """
        Running window based Prediction
        :param obs: observations to evaluate on   #[T=1425,1]
        :param y_context: the label information for the context sets
        :param batch_size: batch_size for evaluation, this does not change the results and is only to allow evaluating on more
         data than you can fit in memory at once. Default: -1, .i.e. batch_size = number of sequences.
        """
        # rescale only batches so the data can be kept in unit8 to lower memory consumptions
        self._model = self._model.eval()
        print("k:",k)
        self._context_size = k
        # List to store predictions
        predictions_list = []
        # Create a tensor to store batches of windows
        batched_windows = []
        gt_list = []
        residual_list = []
        window_size = self._context_size # ctx_size
        Y_list = []

        # Prepare the batches of windows
        for i in range(len(test_obs) - window_size):
            window = test_obs[i:i + window_size]
            ground_truth = test_obs[i + window_size]
            # Ensure window size is [49, 1] that the model() accepts
            window = window.view(window_size, 1)
            batched_windows.append(window)
            Y_list.append(ground_truth)

        # Stack the batches into a single tensor [stacked, ctx_size=96, 1]
        stacked_windows = torch.stack(batched_windows)
        stacked_gt = torch.stack(Y_list)

        dataset = TensorDataset(stacked_windows, stacked_gt)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        # batch_size of dataloader=10 --> obs.shape=[3500,150,25] & obs_batch=[10,150,25]
        for batch_idx, (obs_batch, gt_batch) in enumerate(loader):
            # if(np.mod(batch_idx,5) ==0):
            #     print("   batch_idx:", batch_idx)
            #     pass
            # Assign tensors to devices
            obs_batch = (obs_batch.float()).to(self._device)  # [10,150,25]

            gt_batch = (gt_batch.float()).to(self._device)  # [10,150,1]
            with torch.no_grad():
                # Split to context and targets
                # if self._context_size is None:
                #     k = int(obs_batch.shape[1] / 2)
                # else: # true
                #     k = self._context_size  #75
                # m = obs_batch.shape[1] - k #150-75=75

                X = obs_batch
                # Y = gt_batch

                # try: for the transformer when returns attention map
                #     pred_logits, _ = self._model(X_enc)
                # except:
                #     pred_logits = self._model(X_enc)

                # Forward Pass
                try:
                    all_predictions_batch, _ = self._model(X)
                except:
                    all_predictions_batch = self._model(X)

                # print("running window (rw) inference")
                # print("X_enc.shape:", X.shape)  # X.shape:       torch.Size([350, 75, 25])
                # print("(GT) Y.shape:", gt_batch.shape)  # Y.shape:       torch.Size([350, 75, 1])
                # print("pred_logits.shape", all_predictions_batch.shape)  # Pred_logits:   torch.Size([350, 75, 1])
                # print("------------------------------------")

                B, T1, input_feat = X.shape
                #_, T2, _ = Y.shape
                # C = self._num_class

                if (self._loss == 'cross_entropy'):
                    out_mean_pred = torch.argmax(all_predictions_batch, dim=-1).unsqueeze(dim=-1).float()  # yek adad beyne 0 ta num_class bar migardune baraye dim e -1
                    out_mean_pred = (2 * out_mean_pred + 1 - self._normalizer[0]) / self._normalizer[1]  # convert back from bin_number to percentage Then normalize it
                    all_predictions_batch = out_mean_pred  # change the variable name to be compatible with the rest of the code
                    print("because of crossentropy: all_predictions_batch.shape", all_predictions_batch.shape)



                # Convert the predictions to a list
                predictions_batch = all_predictions_batch.view(-1, 1).cpu()
                predictions_list.append(predictions_batch)

                gt_list.append(gt_batch.view(-1, 1).cpu())
                residual_batch = gt_batch.view(-1, 1).cpu() - predictions_batch
                residual_list.append(residual_batch)

        #print("done with running window ...")
        # return torch.cat(out_mean_list), torch.cat(out_var_list), torch.cat(gt_list)

        return torch.cat(predictions_list), np.nan, torch.cat(gt_list), torch.cat(residual_list)
