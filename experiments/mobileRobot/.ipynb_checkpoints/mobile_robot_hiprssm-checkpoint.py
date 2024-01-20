import sys
sys.path.append('.')
from omegaconf import DictConfig, OmegaConf
import hydra
import os

import numpy as np
import torch
import wandb
import pickle

from data.mobileDataSeq import metaMobileData
from data.mobileDataSeq_Infer import metaMobileDataInfer
from meta_dynamic_models.neural_process_dynamics.neural_process.setFunctionContext import SetEncoder
from meta_dynamic_models.neural_process_dynamics.neural_process_ssm.recurrentEncoderDecoder import acrknContextualDecoder
from meta_dynamic_models.neural_process_dynamics.npDynamics import npDyn
from learning import hiprssm_dyn_trainer
from inference import hiprssm_dyn_inference
from utils.metrics import naive_baseline
from utils.dataProcess import split_k_m
from utils.metrics import root_mean_squared
from utils.latentVis import plot_clustering, plot_clustering_1d

nn = torch.nn


def generate_mobile_robot_data_set(data, dim):
    train_windows, test_windows = data.train_windows, data.test_windows

    train_targets = train_windows['target'][:,:,:dim]
    test_targets = test_windows['target'][:,:,:dim]

    train_obs = train_windows['obs'][:,:,:dim]
    test_obs = test_windows['obs'][:,:,:dim]

    train_task_idx = train_windows['task_index']
    test_task_idx = test_windows['task_index']

    train_act = train_windows['act'][:,:,:dim]
    test_act = test_windows['act'][:,:,:dim]
    print(test_act.shape, train_act.shape)

    return torch.from_numpy(train_obs).float(), torch.from_numpy(train_act).float(), torch.from_numpy(train_targets).float(), torch.from_numpy(train_task_idx).float(),\
           torch.from_numpy(test_obs).float(), torch.from_numpy(test_act).float(), torch.from_numpy(test_targets).float(), torch.from_numpy(test_task_idx).float()

@hydra.main(config_path='conf',config_name='config')
def my_app(cfg)->OmegaConf:
    global config
    model_cfg = cfg.model
    exp = Experiment(model_cfg)


class Experiment():
    def __init__(self, cfg):
        self.global_cfg = cfg
        self._experiment()


    def _experiment(self):
        """Data"""
        cfg = self.global_cfg
        torch.cuda.empty_cache()

        tar_type = cfg.data_reader.tar_type  # 'delta' - if to train on differences to current states
        # 'next_state' - if to trian directly on the  next states

        data = metaMobileData(cfg.data_reader)

        train_obs, train_act, train_targets, train_task_idx, test_obs, test_act, test_targets, test_task_idx = generate_mobile_robot_data_set(
            data,  cfg.data_reader.dim)
        act_dim = train_act.shape[-1]

        """Naive Baseline"""
        naive_baseline(test_obs[:, :-1, :], test_obs[:, 1:, :], steps=[1, 3, 5, 7, 10], data=data, denorma=True)

        ####
        impu = cfg.data_reader.imp

        save_path = os.getcwd() + '/experiments/saved_models/' + cfg.wandb.exp_name + '.ckpt'

        ##### Define WandB Stuffs
        expName = cfg.wandb.exp_name
        if cfg.wandb.log:
            mode = "online"
        else:
            mode = "disabled"

        ## Initializing wandb object and sweep object
        wandb_run = wandb.init(project=cfg.wandb.project_name, name=expName,
                               mode=mode)  # wandb object has a set of configs associated with it as well

        ### Model, Train and Inference Modules
        encoder = SetEncoder(
            train_obs.shape[-1] + train_act.shape[-1] + train_targets.shape[-1],
            lod=cfg.np.agg_dim, config=cfg.set_encoder)


        decoder = acrknContextualDecoder(ltd=cfg.np.agg_dim*2, target_dim=train_targets.shape[-1],
                                         lod=cfg.np.latent_obs_dim,
                                         lad=train_act.shape[-1], config=cfg.ssm_decoder)

        np_model = npDyn(encoder, decoder, dec_type='acrkn', config=cfg.np)
        np_learn = hiprssm_dyn_trainer.Learn(np_model, loss=cfg.learn.loss, imp=impu, config=cfg, run=wandb_run,
                                           log=cfg.wandb['log'])

        if cfg.learn.load == False:
            #### Train the Model
            np_learn.train(train_obs, train_act, train_targets, train_task_idx, cfg.learn.epochs, cfg.learn.batch_size,
                           test_obs, test_act,
                           test_targets, test_task_idx)

        if not cfg.wandb.sweep:
            ##### Load best model
            model_at = wandb_run.use_artifact('saved_model' + ':latest')
            model_path = model_at.download()  ###return the save durectory path in wandb local
            np_model.load_state_dict(torch.load(save_path))
            print('>>>>>>>>>>Loaded The Model From Local Folder<<<<<<<<<<<<<<<<<<<')

            ###### Inference

            ##########  Initialize inference class
            np_infer = hiprssm_dyn_inference.Infer(np_model, data=data, config=cfg, run=wandb_run)
            batch_size = 10
            k = int(train_obs.shape[1] / 2)
            pred_mean, pred_var, gt, obs_valid, _, _, cur_obs = np_infer.predict(test_obs, test_act, test_targets, test_task_idx,
                                                                        imp=impu, k=k,
                                                                        test_gt_known=True, batch_size=batch_size, tar=tar_type)
            print(pred_mean.shape, pred_var.shape, gt.shape, obs_valid.shape)



            rmse_next_state, pred_obs, gt_obs = root_mean_squared(pred_mean, gt, data,
                                                                      tar="observations", denorma=True)
            wandb_run.summary['rmse_denorma_next_state'] = rmse_next_state

            print("Root mean square Error is:", rmse_next_state)


            multiSteps = [1,3,5,10,20,30,40,50]
            for step in multiSteps:
                 pred_mean, pred_var, gt_multi = np_infer.predict_mbrl(test_obs, test_act, test_targets, k=k,
                                                                 batch_size=batch_size,
                                                                 multiStep=step, tar=tar_type)

                 rmse_next_state, pred_obs, gt_obs = root_mean_squared(pred_mean, gt_multi, data, tar="observations", denorma=True)
                 print(step,rmse_next_state)
                 wandb_run.summary['rmse_multi_step_' + str(step)] = rmse_next_state


def main():
    my_app()



## https://stackoverflow.com/questions/32761999/how-to-pass-an-entire-list-as-command-line-argument-in-python/32763023
if __name__ == '__main__':
    main()