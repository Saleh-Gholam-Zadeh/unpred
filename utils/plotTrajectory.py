import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import wandb
#plt.figure(figsize=(3, 3))
import random

from utils.dataProcess import diffToState, diffToStateImpute, denorm
from datetime import datetime


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def plotImputation(gts, valid_flags, pred_mus, pred_stds, wandb_run, dims=None, num_traj: int =35, log_name='test', exp_name='trial', show=False ,tar_type=None ,tar_obs=None ,normalizer=None ,loss=None): #num_traj=2 --->saleh changed to 5
    '''
    gts:groundtruth_target, which can be diff or actual observation
    valid_flag: determines if this point is visible to the model or not
    pred_mus:prediction of the model can be normalized diff or normalized obs
    tar_type: delta or observations
    tar_obs: normalized observations used for converting back from diff to actual observation
    normalizer: dict of means and stds used to denormalize the data when we want to convert back from diff to actual obs
    '''
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    print("plottrajectory.py line 11   plotImputation() is called ")
    print("num_traj:",num_traj)


    now = datetime.now()
    now_str = str(now).replace(" ", "_")[:19] +"_"

    folder_name = os.path.join(os.getcwd() , 'experiments/pam/runs/train_validation_normalized_plots', str(exp_name), now_str)
    if not(os.path.exists(folder_name)):
        os.makedirs(folder_name)

    trjs = np.random.randint(gts.shape[0],size=num_traj)
    print("trjs = ",trjs)
    gts_actual=None
    pred_mus_actual=None
    if(tar_type=='delta'):
        print("convert diff to actual observation before plot")

        pred_mus_convertedfrom_diff_normalized = diffToStateImpute(pred_mus, tar_obs, valid_flags, normalizer, standardize=True)[0] #added by saleh
        pred_mus_actual                        =  denorm(pred_mus_convertedfrom_diff_normalized, normalizer=normalizer , tar_type='label')

        gts_converted_from_diff_normalized     =        diffToState(gts,      tar_obs,              normalizer, standardize=True ,gt_flag=True)[0]    #added by saleh
        gts_actual                             =        denorm(gts_converted_from_diff_normalized, normalizer=normalizer , tar_type='label')

    if(tar_type=="observations" and normalizer is not None  and loss=='mse'):
        print("convert normalized observation to actual observation before plot")
        pred_mus_actual   =  denorm(pred_mus[:,:,-1:], normalizer=normalizer , tar_type='packets')
        gts_actual        =  denorm(gts[:,:,-1:],      normalizer=normalizer , tar_type='packets')

    if(tar_type=="observations" and normalizer is not None  and loss=='kl_rmse'):
        print("convert normalized observation to actual observation before plot")
        pred_mus_actual   =  denorm(pred_mus[:,:,-1:], normalizer=normalizer , tar_type='packets')
        gts_actual        =  denorm(gts[:,:,-1:],       normalizer=normalizer , tar_type='packets')

    if (tar_type == "observations" and normalizer is None):
        print("actual observation is already what we are working with")
        pred_mus_actual  =  pred_mus
        gts_actual = gts


    if dims is None:
        dims = np.arange(gts.shape[-1])
    for traj in trjs:
        for dim in dims:
            gt = gts[traj,:,dim]

            if gts_actual is not None:# added by saleh
                gt_actual =gts_actual[traj,:,dim] # added by saleh

            if valid_flags is not None:
                valid_flag = valid_flags[traj,:,0]
            pred_mu = pred_mus[traj,:,dim]

            if(pred_mus_actual is not None): # added by saleh
                pred_mu_actual = pred_mus_actual[traj,:,dim] # added by saleh


            # if (pred_stds is not None or pred_stds is not np.nan):
            #     pred_std = pred_stds[traj,:,dim]

            fig,axs = plt.subplots(2,figsize=(20, 12))
            #plt.Figure()
            #plt.figure(figsize=(20, 5))
            axs[0].plot(gt ,'o--', label='gt' , markersize=2 )
            if(tar_type is not None):
                axs[0].set_title(str(log_name)+"_"+str(tar_type)+"_"+str(traj)+"_"+exp_name)
            else:
                axs[0].set_title(str(log_name)+"_"+str(traj) +"_"+exp_name)
            axs[0].set_ylim([-5, 5])
            axs[0].grid(which='both',axis="both")
            axs[0].minorticks_on()
            if valid_flags is not None:
                axs[0].scatter(torch.arange(len(valid_flag))[np.logical_not(valid_flag)],gt[np.logical_not(valid_flag)],facecolor='red',s=20)
            axs[0].plot(pred_mu,'o--', color='black' , label='pred_mu', markersize=2)

            # if pred_stds is not None:#saleh_uncommented
            #     plt.fill_between(np.arange(len(gt)), pred_mu - pred_std, pred_mu + pred_std, alpha=0.2, color='grey') #saleh_uncommented
            if(gts_actual is not None and pred_mus_actual is not None):
                axs[1].plot(gt_actual, 'o--', label='gt_actual' , markersize=2)
                axs[1].plot(pred_mu_actual, 'o--', label='pred_actual', color='black', markersize=2)
                axs[1].set_title( str(log_name) + "_actual" + "_" + str(traj)  +"_"+exp_name)
                axs[1].set_ylim([-1, 25])
                axs[1].grid(which='both', axis="both")
                axs[1].minorticks_on()

            if show == True:
                axs[0].legend()
                if (gts_actual is not None and pred_mus_actual is not None):
                        axs[1].legend()
                plt.show()
                plt.close()
            else:
                print('saving trajectory plotImputation: ',folder_name + "/traj_" + str(traj) + exp_name +'_plotImputation_' + log_name + ".png")
                axs[0].legend()
                if (gts_actual is not None and pred_mus_actual is not None):
                    axs[1].legend()


                fig.savefig(       folder_name + "/traj_" + str(traj)  +"_" + exp_name +'_plotImputation_' + now_str +log_name + ".png")
                image = plt.imread(folder_name + "/traj_" + str(traj)  +"_" + exp_name +'_plotImputation_' + now_str +log_name + ".png")
                if wandb_run is not None:
                    key = 'Imp_Trajectory_' + str(traj) +'_' + log_name
                    wandb_run.log({key: wandb.Image(image)})
                    #os.remove(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                    #plt.close()
                plt.close()

def plotImputationDiff(gts, valid_flags, pred_mus, pred_stds, wandb_run, dims=[0,1,2,3,4,5], num_traj: int =5, log_name='test', exp_name='trial', show=False): #num_traj=2 --->saleh changed to 5
    print("plottrajectory.py line 45  plotImputationDiff() is called ")


    folder_name = os.getcwd() + '/experiments/pam/runs/latent_plots'
    trjs = np.random.randint(gts.shape[0],size=num_traj)
    for traj in trjs:
        for dim in dims:
            gt = gts[traj,:,dim]
            if valid_flags is not None:
                valid_flag = valid_flags[traj,:,0]
            pred_mu = pred_mus[traj,:,dim]
            pred_std = pred_stds[traj,:,dim]
            plt.Figure()
            plt.figure(figsize=(20, 5))
            plt.plot(gt,'o--',label='gt')
            plt.title("plotImputationDiff")
            if valid_flags is not None:
                plt.scatter(torch.arange(len(valid_flag))[np.logical_not(valid_flag)],gt[np.logical_not(valid_flag)],facecolor='red',s=14)
            plt.plot(pred_mu,'o--', color='black',label='pred_mu')
            plt.fill_between(np.arange(len(gt)), pred_mu - pred_std, pred_mu + pred_std, alpha=0.2, color='grey')
            if show == True:
                plt.legend()
                plt.show()
                plt.close()
            else:
                plt.legend()
                plt.savefig(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + '_plotImputationDiff'+ ".png")
                image = plt.imread(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + '_plotImputationDiff'+ ".png")
                if wandb_run is not None:
                    key = 'Imp_Trajectory_' + str(traj) + '_dim_' + str(dim) +'_' + log_name
                    wandb_run.log({key: wandb.Image(image)})
                    #os.remove(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                    plt.close()
                plt.close()

def plotLongTerm(gts, pred_mus, pred_stds, wandb_run, dims=[0], num_traj=2, log_name='test', exp_name='trial', show=False):
    print("plottrajectory.py line 76  plotLongTerm() is called ")

    folder_name = os.getcwd() + '/experiments/pam/runs/latent_plots'
    trjs = np.random.randint(gts.shape[0],size=num_traj)
    for traj in trjs:
        for dim in dims:
            gt = gts[traj,:,dim]
            pred_mu = pred_mus[traj,:,dim]
            pred_std = pred_stds[traj,:,dim]
            plt.Figure()
            plt.figure(figsize=(20, 5))
            plt.plot(gt,'o--',label='gt')
            plt.plot(pred_mu,'o--', color='black' ,label='pred_mu')
            plt.title("plotLongTerm")

            plt.fill_between(np.arange(len(gt)), pred_mu - pred_std, pred_mu + pred_std, alpha=0.2, color='grey')
            if show == True:
                plt.legend()
                plt.show()
                plt.close()
            else:
                plt.legend()
                plt.savefig(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + '_plotLongTerm' +".png")
                image = plt.imread(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + '_plotLongTerm' + ".png")
                if wandb_run is not None:
                    key = 'MultiStep_Trajectory_' + str(traj) + '_dim_' + str(dim) +'_' + log_name
                    wandb_run.log({key: wandb.Image(image)})
                    #os.remove(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                    plt.close()
                plt.close()


def plotMbrl(gts, pred_mus, pred_stds, wandb_run, dims=[0,1,2,3], num_traj=2, log_name='test', exp_name='trial', show=False):

    print("plottrajectory.py line 76  plotMbrl() is called ")

    folder_name = os.getcwd() + '/experiments/pam/runs/latent_plots'
    trjs = np.random.randint(gts.shape[0],size=num_traj)
    for traj in trjs:
        for dim in dims:
            gt = gts[traj,:,dim]
            pred_mu = pred_mus[traj,:,dim]
            pred_std = pred_stds[traj,:,dim]
            plt.Figure()
            plt.figure(figsize=(20, 5))
            plt.plot(gt,'o--',label='gt')
            plt.plot(pred_mu,'o--', color='black' , label='pred_mu')
            plt.fill_between(np.arange(len(gt)), pred_mu - pred_std, pred_mu + pred_std, alpha=0.2, color='grey')
            plt.title("plotMbrl")
            if show == True:
                plt.legend()
                plt.show()
                plt.close()
            else:
                plt.legend()
                plt.savefig(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + '_plotMbrl' + ".png")
                image = plt.imread(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + '_plotMbrl' + ".png")
                if wandb_run is not None:
                    key = 'MBRL_Trajectory_' + str(traj) + '_dim_' + str(dim) +'_' + log_name
                    wandb_run.log({key: wandb.Image(image)})
                    #os.remove(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                    plt.close()
                plt.close()



if __name__ == '__main__':
    global ax
    gt = np.random.rand(10,50,1)
    pred = np.random.rand(10,50,1)
    std = np.random.uniform(low=0.01, high=0.1, size=(10,50,1))
    rs = np.random.RandomState(seed=23541)
    obs_valid = rs.rand(gt.shape[0], gt.shape[1], 1) < 1 - 0.5
    pred = np.random.rand(10, 50, 1)
    plotSimple(gt[1,:,0],obs_valid[1,:,0],pred[1,:,0],pred_std=std[1,:,0])
    plotMbrl(gt[1,:,0],pred[1,:,0],pred_std=std[1,:,0])