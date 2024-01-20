from matplotlib import pyplot as plt
import numpy as np
import wandb
from utils.dataProcess import norm, denorm
import datetime
import os

#def root_mean_squared(pred, target, data=[], tar='observations', fromStep=0, denorma=False, plot=None):
def root_mean_squared(pred, target,observed=None ,normalizer=None, tar='observations', fromStep=0, denorma=False, plot=None,steps=None ,num_traj=20 ,WB=None , loss=None, key=None):
    """
    root mean squared error
    :param target: ground truth positions
    observed: only for plot purposes. shows the observed part that prediction is based on it
    :param pred_mean_var: mean and covar (as concatenated vector, as provided by model)
    :return: root mean squared error between targets and predicted mean, predicted variance is ignored
    """

    if type(pred) is not np.ndarray:
        pred = pred.cpu().detach().numpy()
    if type(target) is not np.ndarray:
        target = target.cpu().detach().numpy()
    pred = pred[..., :target.shape[-1]]
    if type(observed) is not np.ndarray and observed is not None:
        observed = observed.cpu().detach().numpy()



    assert target.shape == pred.shape, "shape mismatch // target.shape = " + str(target.shape) + "and predicted_mean.shape =" + str(pred.shape) + " are provided"

    if loss !='cross_entropy':
        text = "WO_Denorm"
        if denorma == True:
            text = "Denormalized"
            #print("line 29 metrics.py denorma==True")
            #print("target BEFORE denorm = ",target )

            #print("normalizer:", normalizer)
            pred = denorm(pred, normalizer, tar)
            target = denorm(target, normalizer, tar)

            observed = denorm(observed[:,:,-1:], normalizer, tar) #tar="packet"

            # pred_plot   = denorm(pred_plot, normalizer, tar)
            # target_plot = denorm(target_plot, normalizer, tar)

            #print("target AFTER denorm = ", target)

    else: #we dont need to de-normalize the pred // but gt(target) and observed should be
        text = "Denormalized"
        target = denorm(target, normalizer, tar)
        #print("target in cross entropy setting:", target)
        if observed is not None:
            observed = denorm(observed[:,:,-1:], normalizer, tar)


    if observed is not None:
        pred_plot   = np.concatenate( (observed[:,:,-1:],pred)  ,axis=1 ) # observed might include hist_arrival time
        target_plot = np.concatenate( (observed[:,:,-1:],target),axis=1 )


    if (plot is not None)  and (steps is not None) and (denorma == True):
        folder_name = os.path.join(os.getcwd() , 'experiments/pam/runs/multistep_plots' , str(WB.name))
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        trjs = np.random.randint(target.shape[0], size=num_traj)
        # print("trajectories = ",trjs)

        for traj in trjs:
            for idx in range(target.shape[2]):
                # plt.figure(figsize=(40, 8))
                fig, ax = plt.subplots(1, 1, figsize=(20, 6))
                xs = np.arange(len(target_plot[traj, :, idx]))  # added
                ys = target_plot[traj, :, idx]  # added

                ax.plot(target_plot[traj, :, idx], 'o--', label='gt' , markersize=3)
                ax.plot(pred_plot[traj, :, idx], 'o--', label='prediction' , color='black', markersize=3)
                ax.set_ylim([-2, 45])


                if observed is not None:
                    ax.axvline(x=observed.shape[1]-1, color='r', label='prediction starts')

                for x, y in zip(xs, ys):  # added  the block to add the text of the value of points
                    label = "{:.0f}".format(y)
                    ax.annotate(label,  # this is the text
                                 (x, y),  # these are the coordinates to position the label
                                 textcoords="offset points",  # how to position the text
                                 xytext=(0, 10),  # distance from text to points (x,y)
                                 ha='center' ,color='blue' ,fontsize=10)  # horizontal alignment can be left, right or center
                if key is not None:
                    ax.set_title(key + str(steps) + " Step ahead prediction_" + "traj_" + str(traj) + "_" + text)
                else:
                    ax.set_title(str(steps) + " Step ahead prediction_" + "traj_" + str(traj) + "_" + text)
                ax.grid(which='both', axis="both")
                ax.minorticks_on()
                ax.legend()


                exp_name =  "traj=" + str(traj) + "_" + str(steps) + '_steps_' + text + ".png"
                save_address = os.path.join(folder_name ,exp_name )
                #print("save_address:" , save_address)
                fig.savefig(save_address)
                #plt.show()
                image = plt.imread(save_address)
                plt.close('all')

                if WB is not None:
                    key1 = 'Trajectory:' + str(traj) + "_steps:" + str(steps)
                    # print("wandb uploading plot...")
                    WB.log({key1: wandb.Image(image)})

    # target = target[:, fromStep:, :]
    # pred = pred[:, fromStep:, :]
    numSamples = 1
    for dim in target.shape:
        numSamples = numSamples * dim
    # print('RMSE Samplesss......................................',numSamples)
    sumSquare = np.sum(np.sum(np.sum((target - pred) ** 2)))
    return np.sqrt(sumSquare / numSamples), pred, target









    sumSquare = 0
    count = 0
   #  if plot != None:
   #      for idx in range(target.shape[2]):
   #          plt.plot(target[3,:,idx],label='target')
   #          plt.plot(pred[3,:,idx],label='prediction')
   #          plt.legend()
   #          plt.show()
   #
   #  if denorma==True:
   #      pred = denorm(pred, data, tar)
   #      target = denorm(target, data, tar)
   #
   #
   #
   #  #target = target[:, fromStep:, :]
   # # pred = pred[:, fromStep:, :]
   #  numSamples = 1
   #  for dim in target.shape:
   #      numSamples = numSamples * dim
   #  #print('RMSE Samplesss......................................',numSamples)
   #  sumSquare = np.sum(np.sum(np.sum((target - pred) ** 2)))
   #  return np.sqrt(sumSquare / numSamples), pred, target

def joint_mse(pred, target, data=[], tar='observations', fromStep=0, denorma=False, plot=None):
    """
    :return: mse
    """
    if denorma==True:
        pred = denorm(pred, data, tar)
        target = denorm(target, data, tar)

    numSamples = 1
    for dim in target.shape:
        numSamples = numSamples * dim
    # print('RMSE Samplesss......................................',numSamples)
    #sumSquare = np.sum(np.sum(((target - pred)/target) ** 2,0),0)
    sumSquare = np.sum(np.sum(((target - pred)) ** 2, 0), 0)
    return sumSquare / numSamples

# loss functions
def gaussian_nll(pred,target):
    """
    gaussian nll
    :param target: ground truth positions
    :param pred_mean_var: mean and covar (as concatenated vector, as provided by model)
    :return: gaussian negative log-likelihood
    """
    pred_mean, pred_var = pred[..., :target.shape[-1]], pred[..., target.shape[-1]:]

    pred_var += 1e-8
    element_wise_nll = 0.5 * (np.log(2 * np.pi) + np.log(pred_var) + ((target - pred_mean)**2) / pred_var)
    sample_wise_error = np.sum(element_wise_nll, axis=-1)
    return np.mean(sample_wise_error)


def comparison_plot(target,pred_list=[],name_list=[],data=[], tar='observations', denorma=False):
    '''
    :param target: ground truth
    :param pred_list: list of predictions to compare
    :param name_list: names to each of predictions given as a list
    :return:
    '''
    sample = np.random.randint(target.shape[0])
    sample = 0
    print('sample number',sample)
    #fig, axes = plt.subplots(5, sharex=True, sharey=True)
    if denorma==True:
        target = denorm(target, data, tar)
        for idx,pred in enumerate(pred_list):
            pred_list[idx] = denorm(pred, data, tar)
        #plt.ylim((-1, 1))
        # plt.legend()
        # plt.show()

    fig, axs = plt.subplots(3)
    for k,idx in enumerate([0,1,4]):
        axs[k].plot(target[sample, :, idx], label='GT')
        for pred,name in zip(pred_list,name_list):
            axs[k].plot(pred[sample, :, idx], label=name)
            axs[0].title.set_text('Torque Preditctions For Joint 1, 4 and 5')
            axs[k].legend()
            axs[k].set(ylabel="Torque(Nm)")
        #plt.ylim((-1
        # , 1))
        #plt.legend()
    plt.show()

# def naive_baseline(current_obs,targets,data=[],tar_type='observations',steps=[1,3,5,10,20],denorma=False):
#     '''
#     :param current_obs: current available observations
#     :param targets: actual targets
#     :param steps: list of steps for calculating n step ahead prediction accuracy
#     :return: Nothing
#     '''
#     if type(current_obs) is not np.ndarray:
#         current_obs = current_obs.cpu().detach().numpy()
#     if type(targets) is not np.ndarray:
#         targets = targets.cpu().detach().numpy()
#
#     for step in steps:
#         if step==1:
#             pred=current_obs
#         else:
#             pred = current_obs[:,:-(step-1),:]
#         tar = targets[:,step-1:,:]
#         print('root mean square error step',step,root_mean_squared(pred,tar,data,tar=tar_type,denorma=denorma)[0])

def naive_baseline(current_obs,targets,normalizer,tar_type='targets',steps=[1,3,5,10,20],denorma=False ,WB=None):
    '''
    :param current_obs: current available observations
    :param targets: actual targets
    :param steps: list of steps for calculating n step ahead prediction accuracy
    :return: Nothing
    '''
    if type(current_obs) is not np.ndarray:
        current_obs = current_obs.cpu().detach().numpy()
    if type(targets) is not np.ndarray:
        targets = targets.cpu().detach().numpy()
    if(denorma==True):
        wb_text = "Denormalized"
    else:
        wb_text = "Normalized"
    wb_key = "NB_ERR_" + wb_text + "_Multistep:"
    for step in steps:
        if step==1:
            pred=current_obs
        else:
            pred = current_obs[:,:-(step-1),:]
        tar = targets[:,step-1:,:]
        err = root_mean_squared(pred,tar,normalizer,tar=tar_type,denorma=denorma)[0]
        print(wb_text,' root mean square error ',step,' ahead = ' ,err)

        if WB is not None:
            WB.summary[wb_key + str(step)] = err