

import sys

import matplotlib.pyplot as plt

sys.path.append('.')
from omegaconf import DictConfig, OmegaConf
import hydra
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  #import it before torch!  https://discuss.pytorch.org/t/cuda-visible-device-is-of-no-use/10018/11


import numpy as np
import torch
import wandb
import pickle

from data.mobileDataSeq import metaMobileData
# from data.mobileDataSeq_Infer import metaMobileDataInfer
# from meta_dynamic_models.neural_process_dynamics.neural_process.setFunctionContext import SetEncoder
# from meta_dynamic_models.neural_process_dynamics.neural_process_ssm.recurrentEncoderDecoder import acrknContextualDecoder
# from meta_dynamic_models.neural_process_dynamics.npDynamics import npDyn
from learning import transformer_trainer #hiprssm_dyn_trainer
from inference import transformer_inference  # hiprssm_dyn_inference
# from utils.metrics import naive_baseline
# from utils.dataProcess import split_k_m
from utils.metrics import root_mean_squared
# from utils.latentVis import plot_clustering, plot_clustering_1d
# import random

#from transformer_architecture.model_transformer_TS import  TransformerModel, GPTConfig # the older GPT model only contains decoder
from transformer_architecture.models.Transformer_Longterm import  LongTermModel, TSConfig
from transformer_architecture.ns_models.ns_Transformer import  Model, NS_TSConfig
from mlp_arcitecture.mlp_arch import MLP
from LSTM_arcitecture.LSTM_arch import CustomLSTM #TwoLayerLSTM
from  utils.dataProcess import ts2batch , get_statistics , normalize , denormalize ,run_test , get_mutual_information , linear_correl
import fnmatch

import math
import random
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
import time
import threading
import multiprocessing




# torch.manual_seed(2)
# random.seed(2)
# np.random.seed(2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


nn = torch.nn
print('hi')


def generate_from_window(train_win,test_win):
    train_targets = train_win['target']
    test_targets = test_win['target']
    #print("saleh_add: line 36 mobile_robot_hiprssm.py")


    train_obs = train_win['obs'][:,:,:]
    test_obs = test_win['obs'][:,:,:]

    train_task_idx = train_win['task_index'] # visualization
    test_task_idx = test_win['task_index']

    train_act = train_win['act'][:,:,:] #all 0 checked
    test_act = test_win['act'][:,:,:] #all 0 checked


    print("train_targets.shape",train_targets.shape)
    print("test_targets.shape" ,test_targets.shape)
    print("train_obs.shape = " ,train_obs.shape)
    print("train_act.shape = " ,train_act.shape)
    normalization = train_win['normalization']

    print("mobile_robot_hiprssm.py line 70   normalization = ",normalization)
    #print(test_act.shape, train_act.shape) #(batch_size,2*data_reader_batch_size,#num_features) e.g (1000,2*75,10)
    print("*********************************************************************************************************")
    print("******************************************** Data Loading Done ************************************")
    print("*********************************************************************************************************")
    return torch.from_numpy(train_obs).float() , torch.from_numpy(train_act).float(), torch.from_numpy(train_targets).float(), torch.from_numpy(train_task_idx).float(),\
           torch.from_numpy(test_obs).float() , torch.from_numpy(test_act).float(),  torch.from_numpy(test_targets).float(), torch.from_numpy(test_task_idx).float(), normalization





def generate_mobile_robot_data_set(data, dim): #dim is useless. it was used in some older project in order not to use all features. it was set to 111 which is much bigger than our feature size
    #print("saleh_add: line 28 mobile_robot_hiprssm.py    train_target is only the #packet I.e [:,:,-1]")
    train_windows, test_windows = data.load_data()  # normalization ham khodesh load mishe tuye data

    train_targets = train_windows['target']
    test_targets = test_windows['target']

    train_obs = train_windows['obs'][:,:,:]
    test_obs = test_windows['obs'][:,:,:]

    train_task_idx = train_windows['task_index'] # visualization
    test_task_idx = test_windows['task_index']

    train_act = train_windows['act'][:,:,:] #all 0 checked
    test_act = test_windows['act'][:,:,:] #all 0 checked

    print("train_targets.shape",train_targets.shape)
    print("test_targets.shape" ,test_targets.shape)
    print("train_obs.shape = " ,train_obs.shape)
    print("train_act.shape = " ,train_act.shape)


    print("mobile_robot_hiprssm.py line 105   data.normalization = ",data.normalization)
    print("packets_normalized:",(np.arange(12)-data.normalization['packets'][0])/data.normalization['packets'][1])
    #print(test_act.shape, train_act.shape) #(batch_size,2*data_reader_batch_size,#num_features) e.g (1000,2*75,10)
    print("*********************************************************************************************************")
    print("******************************************** Data Preprocessing Done ************************************")
    print("*********************************************************************************************************")

    return torch.from_numpy(train_obs).float(), torch.from_numpy(train_act).float(), torch.from_numpy(train_targets).float(), torch.from_numpy(train_task_idx).float(),\
           torch.from_numpy(test_obs).float(), torch.from_numpy(test_act).float(), torch.from_numpy(test_targets).float(), torch.from_numpy(test_task_idx).float()


def generate_from_dict(train_win,test_win,pack_norm):
    print("packet_norm:",pack_norm)


    train_obs = train_win['obs'][:,:,:]
    test_obs = test_win['obs'][:,:,:]

    print("train_obs.shape = " ,train_obs.shape)

    normalization = train_win['normalization']

    print("mobile_robot_hiprssm.py line 70   normalization = ",normalization)
    #print(test_act.shape, train_act.shape) #(batch_size,2*data_reader_batch_size,#num_features) e.g (1000,2*75,10)
    print("*********************************************************************************************************")
    print("******************************************** Data Loading Done ************************************")
    print("*********************************************************************************************************")
    return None, None , None , None

# def zero_initialize_weights(model):
#     for param in model.parameters():
#         if param.requires_grad:
#             nn.init.zeros_(param)
    # inp = torch.randn(350, 49,1)
    # with torch.no_grad():
    #     out1 = model(inp)
    #     print(out1)
    #     print("----------------------")
    #     zero_initialize_weights(model)
    #     out2 = model(inp)
    #     print(out2)


@hydra.main(config_path='conf',config_name='config')
def my_app(cfg)->OmegaConf:
    global config
    model_cfg = cfg.model
    print('my_app')
    exp = Experiment(model_cfg)
 #set a flag for action --->assign 0 tensor to action

class Experiment():
    def __init__(self, cfg):
        self.global_cfg = cfg
        print("experiment..")
        agg_result_dict={}

        #az inja bayad avaz beshe:
        # with open('selected_result_dict.pickle', 'rb') as file2:
        #     selected_result = pickle.load(file2)  # {'flow_331': [array([0.01563125]), array([0.17415701]), 320000]}
        #     print(selected_result.keys())
        #     selected_keys=list(selected_result.keys())
        #
        # print('selected_result_dict.pickle loaded ')
        seed = cfg.learn.seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        #self.all_flows = cfg.data_reader.flows    #list(selected_result.keys()) #cfg.data_reader.flows
        #num_keys_to_take = len(self.all_flows) // 10
        #i = cfg.data_reader.portion
        #start_index = (i) * num_keys_to_take
        #end_index = (i+1) * num_keys_to_take

        #selected_keys = self.all_flows[start_index:end_index]

        #start = selected_keys.index("flow_755")
        #self.all_flows =selected_keys[start:start+1] #["flow_800","flow_787","flow_390"] #
        # self.all_flows = selected_keys[123:]
        #self.all_flows = selected_keys[-30:-20] #laptop
        #self.all_flows = selected_keys[-1:]
        #self.all_flows=['flow_743','flow_771','flow_780','flow_787','flow_832','flow_885']
        #self.all_flows = [ 'flow_885']
        #self.all_flows = ['flow_37','flow_164','flow_167','flow_249','flow_259','flow_296','flow_412','flow_503','flow_563','flow_641','flow_642','flow_713','flow_714','flow_773','flow_859','flow_869','flow_872','flow_944','flow_1019','flow_1063','flow_1066','flow_1117']

        #Candiadets_MI_suitable for further reduction:
        #candidate_keys =         ['flow_37','flow_143', 'flow_220', 'flow_259', 'flow_296', 'flow_346', 'flow_350',  'flow_510','flow_516', 'flow_532', 'flow_541', 'flow_554', 'flow_559', 'flow_564', 'flow_567', 'flow_573', 'flow_574','flow_578', 'flow_622', 'flow_641', 'flow_667', 'flow_773','flow_1088']
        #start = candidate_keys.index("flow_574")
        #candidate_keys =   ['flow_37','flow_143','flow_220','flow_564']

        #self.all_flows =candidate_keys [start+1:]
        self.all_flows = None
        print("all_flows:",self.all_flows)

        if self.all_flows is not None: # should come from the list of selected keys:
            for flow in self.all_flows:
                partial_result= self._experiment(flow)
                #self._experiment(flow)
                agg_result_dict.update(partial_result)
                wandb.finish()
                print("Next CPU -------------------------------------------------------------------------------------------------------------------------------")
            print('finished all flows')
            print("saving final results dict")
            with open('all_of_selected_ones_result_dict_' + '.pickle', 'wb') as handle9:
                pickle.dump(agg_result_dict, handle9, protocol=pickle.HIGHEST_PROTOCOL)
                print("agg results saved successfully ")
        else:
            self._experiment()
            wandb.finish()

        # new code
        # max_computation_time = 1000 # seconds
        # processes = []
        # retries = {}
        # max_retries = 3
        # for flow in self.all_flows:
        #     retries[flow] = 0
        #     print("current_flow: ",flow)
        #     print("*****************************************************************")
        #     print("retries[flow]: ",retries[flow])
        #     print("*****************************************************************")
        #     while retries[flow] < max_retries:
        #         try:
        #             process = multiprocessing.Process(target=self._experiment, args=(flow,))
        #             processes.append(process)
        #             print('len(processes):',len(processes))
        #             process.start()
        #         # Create a new process for each computation
        #
        #
        #         #thread.start()
        #
        #             start_time = time.time()
        #             while process.is_alive():
        #                 elapsed_time = time.time() - start_time
        #                 if elapsed_time > max_computation_time:
        #                     #thread.join()  # Terminate the thread
        #                     process.terminate()  # Terminate the process
        #                     retries[flow] += 1  # Increment the retry count
        #                     print(f"Computation for element {flow} took {elapsed_time} seconds which is exceeded the maximum allowed time. Retry {retries[flow]}/{max_retries}...")
        #
        #                     break
        #             else:
        #                 # If the thread completed without exceeding the time limit, exit the while loop
        #                 #wandb.finish()
        #                 print('flow:',flow, ' is finished successfully',' Took:',elapsed_time, ' sec')
        #                 break
        #         except Exception as e:
        #             print(f"Error in starting computation for element {flow}: {str(e)}")
        #     print("Next CPU -------NoneStationary-transformer------------------------------------------------------------")
        # print('finished all flows')
        # # Wait for all processes to finish
        # for process in processes:
        #     process.join()



    def _experiment(self,flow=None):
        """Data"""
        cfg = self.global_cfg
        torch.cuda.empty_cache()
        if flow is not None:
            #print(flow)
            #key = "flow_"+str(flow)
            key = str(flow)
            print("key:",key)

        seed = cfg.learn.seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        tar_type = cfg.data_reader.tar_type  # 'delta' - if to train on differences to current states
        # 'next_state' - if to trian directly on the  next states

        #data = metaMobileData(cfg.data_reader) ##SALEH commented out because we need access to all of the config parameters
        #data = metaMobileData(cfg.data_reader) ##SALEH commented out because we need access to all of the config parameters
        data = metaMobileData(cfg) #SALEH ADDED
        # train_targets and test_targets  (based on the config) can be 1st difference
        mb = cfg.data_reader.meta_batch_size
        features_in  = cfg.learn.features_in
        features_out = cfg.learn.features_out
        # txt_train = 'train_data_window_' + str(mb) + str(features_in) + "->" + str(features_out) + '.npz'
        # txt_test  = 'test_data_window_'  + str(mb) + str(features_in) + "->" + str(features_out) + '.npz'
        # txt_train = 'train_data_window_' + str(mb) + '.npz'
        # txt_test  = 'test_data_window_'  + str(mb) + '.npz'
        #data_path = os.path.join(os.path.dirname(os.getcwd()) ,'all_data','workload')

        txt_train = 'train_dict.pickle'
        txt_test =  'test_dict.pickle'
        txt_norm =  'packet_normalizer.pickle'


        if cfg.data_reader.load:
            try: # if .npz exist:
                # with open('train_dict_normalized.pickle', 'rb') as file1:
                #     train_dict = pickle.load(file1)
                #     if flow is not None:
                #         train_dict = {key: train_dict[key]}



                ###### loading originals #########
                ctx_len = cfg.data_reader.context_size
                tar_len = cfg.data_reader.pred_len


                # with open('DE1_train_batched.pickle', 'rb') as file1:
                #     train_batched_unorm = pickle.load(file1)
                #
                # with open('DE1_test_batched.pickle', 'rb') as file2:
                #     test_batched_unorm = pickle.load(file2)

                with open('DE1_train_batched' + str(ctx_len) + '_to_' + str(tar_len) + '.pickle', 'rb') as file1:
                    train_batched_unorm = pickle.load(file1)

                with open('DE1_test_batched' + str(ctx_len) + '_to_' + str(tar_len) + '.pickle', 'rb') as file2:
                    test_batched_unorm = pickle.load(file2)

                with open('DE1_train_flat.pickle', 'rb') as file3:
                    train_flat_arr = pickle.load(file3)

                with open('DE1_test_flat.pickle', 'rb') as file4:
                    test_flat_arr = pickle.load(file4)

                with open('DE1_normalzier.pickle', 'rb') as file6:
                    normalizer = pickle.load(file6)





                #print("flowise_normalization ....")


                # train_batched_norm  =  (train_batched_unorm.copy() - normalizer['mean'])/normalizer['sigma']
                # test_batched_norm   =  (test_batched_unorm.copy() - normalizer['mean']) / normalizer['sigma']


                # train_flat_arr_norm =  (train_flat_arr.copy()- normalizer['mean'])/normalizer['sigma']
                # test_flat_arr_norm  =  (test_flat_arr.copy() - normalizer['mean'])/normalizer['sigma']
                #print("normalization done")

                # train_obs_arr = train_flat_arr_norm
                # test_obs_arr  = test_flat_arr_norm



                ####### loading residuals ######
                load_dir = cfg.learn.load_dir_residual #   'MLPL2_DE1'
                directory = os.path.join(os.getcwd(),'experiments','residuals','stacking',load_dir)
                pattern_te = "*te*"  + ".pickle"
                pattern_tr = "*tr*"  + ".pickle"
                print("loading residuals from ", directory)

                # List all files in the directory
                files = os.listdir(directory)

                # Iterate through the files and find the one that matches the pattern
                found_file_tr = None
                for filename in files:
                    if fnmatch.fnmatch(filename, pattern_tr):
                        found_file_tr = filename
                        break

                if found_file_tr is not None:
                    print(f"Found file with pattern: {found_file_tr}")
                else:
                    print(f"No file found with pattern: {pattern_tr}")
                    sys.exit(5)

                load_dir_tr = os.path.join(directory, found_file_tr)
                print('train_residual_winows', load_dir_tr)


                with open(load_dir_tr, 'rb') as file1:
                    train_dict_unorm_residual = pickle.load(file1)
                    train_dict_unorm_residual = train_dict_unorm_residual.numpy()


                files = os.listdir(directory)
                # Iterate through the files and find the one that matches the pattern
                found_file_te = None
                for filename in files:
                    if fnmatch.fnmatch(filename, pattern_te):
                        found_file_te = filename
                        break

                if found_file_te is not None:
                    print(f"Found file with pattern: {found_file_te}")
                else:
                    print(f"No file found with pattern: {pattern_te}")
                    sys.exit(5)

                load_dir_te = os.path.join(directory, found_file_te)
                print('test_residual_winows', load_dir_te)


                with open(load_dir_te, 'rb') as file2:
                    test_dict_unorm_residual = pickle.load(file2)
                    test_dict_unorm_residual = test_dict_unorm_residual.numpy()

                # with open('train_obs_arr.pickle', 'rb') as file3:
                #     train_obs_arr = pickle.load(file3)
                #
                # with open('test_obs_arr.pickle', 'rb') as file4:
                #     test_obs_arr = pickle.load(file4)



                print("flowise_normalization ....")
                train_dict = {}
                test_dict  = {}
                train_obs_arr_norm={}
                test_obs_arr_norm={}
                key="DE1"


                train_residual_batched_norm= np.zeros_like(train_dict_unorm_residual)
                train_residual_batched_norm[:, :-cfg.data_reader.pred_len, :] = (train_dict_unorm_residual.copy()[:,:-cfg.data_reader.pred_len, :] - normalizer['mean']) / normalizer['sigma']  # (context - mean) /std
                train_residual_batched_norm[:,-cfg.data_reader.pred_len:, :] = train_dict_unorm_residual.copy()[:,-cfg.data_reader.pred_len:,:]/normalizer['sigma'] #   (residual_target) /std


                test_residual_batched_norm = np.zeros_like(test_dict_unorm_residual)
                test_residual_batched_norm[:, :-cfg.data_reader.pred_len, :] = (test_dict_unorm_residual.copy()[:,:-cfg.data_reader.pred_len, :] - normalizer['mean']) / normalizer['sigma']  # (context - mean) /std
                test_residual_batched_norm[:, -cfg.data_reader.pred_len:, :] = test_dict_unorm_residual.copy()[:,-cfg.data_reader.pred_len:, :] / normalizer['sigma']  # (residual_target) /std



                train_obs_arr_norm[key] = train_flat_arr.copy()
                train_obs_arr_norm[key] = (train_obs_arr_norm[key] - normalizer['mean'])/normalizer['sigma']
                #
                test_obs_arr_norm[key] = test_flat_arr.copy()
                test_obs_arr_norm[key] = (test_obs_arr_norm[key] - normalizer['mean'])/normalizer['sigma']

                print("flowise_normalization done")


            except Exception as error: # if not exist
                print(error)
                #print(txt_train, 'couldnt be found!  need to prepare dataset')
                train_obs, train_act, train_targets, train_task_idx, test_obs, test_act, test_targets, test_task_idx = generate_mobile_robot_data_set(data, cfg.data_reader.dim)

        else:
            train_obs      , train_act        , train_targets, train_task_idx, test_obs    , test_act,    test_targets, test_task_idx = generate_mobile_robot_data_set(  data,  cfg.data_reader.dim)

        #act_dim = train_act.shape[-1]

        ####
        if cfg.learn.features_in== 'all':
            f_index = 0
        else:
            f_index = -1

        impu = cfg.data_reader.imp
        num_class = cfg.learn.num_class
        print("num_class:",num_class)



        if torch.cuda.device_count() > 0:
            size_full = True #GPU --> full-size
        else: #cpu
            size_full = False # CPU--> subsample
        if(cfg.wandb.exp_name[:3]=='tmp' and torch.cuda.device_count() > 0 ): #tmp config on gpu for sanity check
            size_full = False


        transformer_e_layers = cfg.transformer_arch.enc_layer
        transformer_d_layers = cfg.transformer_arch.dec_layer
        transformer_n_head   = cfg.transformer_arch.n_head
        transformer_d_model  = cfg.transformer_arch.d_model
        transformer_dropout = cfg.transformer_arch.dropout
        #transformer_seq_len = cfg.transformer_arch.seq_len
        transformer_seq_len = cfg.data_reader.context_size  # --> #transformer_seq_len = cfg.transformer_arch.seq_len
        #assert transformer_seq_len == cfg.data_reader.context_size
        transformer_factor  = cfg.transformer_arch.factor
        transformer_p_hidden_layers =cfg.transformer_arch.p_hidden_layers
        transformer_p_hidden_dims = cfg.transformer_arch.p_hidden_dims

        #, seq_len = transformer_seq_len, label_len = transformer_seq_len // 2, factor = transformer_factor, p_hidden_layers = transformer_p_hidden_layers

        ##### Define WandB Stuffs
        try:
            expName = cfg.wandb.exp_name +"_e-d-n-Dm-dropout:"+str(transformer_e_layers)+ "-" + str(transformer_d_layers) +"-"+ str(transformer_n_head) +"-"+ str(transformer_d_model)+"-" +str(transformer_dropout) +"_f-in:"+ str(train_residual_batched_norm.shape[-1])  + "_epoch" + str(cfg.learn.epochs)   + "_Loss-" +   str(cfg.learn.loss)+ "_ctx-"   +  str(cfg.data_reader.context_size)  + "to" + str(cfg.data_reader.pred_len)  + "_lr" + str(cfg.learn.lr)+"_seed"+str(cfg.lean.seed) +"_flow:"+str(flow)
        except:
            expName = cfg.wandb.exp_name + "_e-d-n-Dm-dropout:" + str(transformer_e_layers) + "-" + str(
            transformer_d_layers) + "-" + str(transformer_n_head) + "-" + str(transformer_d_model) + "-" + str(
            transformer_dropout) + "_f-in:" + str(train_residual_batched_norm.shape[-1]) + "_epoch" + str(
            cfg.learn.epochs) + "_Loss-" + str(cfg.learn.loss) + "_ctx-" + str(
            cfg.data_reader.context_size) + "to" + str(cfg.data_reader.pred_len) + "_lr" + str(
            cfg.learn.lr) +"_seed"+str(cfg.learn.seed) + "_flow:" + str(flow)

        if cfg.wandb.log:
            mode = "online"
        else:
            mode = "disabled"


        if cfg.learn.loss != 'cross_entropy':
            try:
                my_conf = NS_TSConfig(enc_in=train_dict['flow_1']['vals'].shape[-1], dec_in = train_dict['flow_1']['vals'].shape[-1] , pred_len=cfg.data_reader.pred_len , c_out=1 , d_model=transformer_d_model , n_heads=transformer_n_head, e_layers=transformer_e_layers , d_layers=transformer_d_layers , dropout= transformer_dropout, seq_len=transformer_seq_len ,label_len=transformer_seq_len//2 , factor=transformer_factor , p_hidden_layers=transformer_p_hidden_layers )  # ctx + target = 2*ctx

            except:
                print('here')
                my_conf = NS_TSConfig(enc_in=train_residual_batched_norm.shape[-1], dec_in = train_residual_batched_norm.shape[-1] , pred_len=cfg.data_reader.pred_len , c_out=1 , d_model=transformer_d_model , n_heads=transformer_n_head, e_layers=transformer_e_layers , d_layers=transformer_d_layers , dropout= transformer_dropout, seq_len=transformer_seq_len ,label_len=transformer_seq_len//2 , factor=transformer_factor , p_hidden_layers=transformer_p_hidden_layers, train_res=cfg.learn.train_residual )  # ctx + target = 2*ctx
                #my_conf = TSConfig(enc_in=train_dict[key]['vals'].shape[-1], dec_in = train_dict[key]['vals'].shape[-1]           , pred_len=cfg.data_reader.pred_len , c_out=1 , d_model=transformer_d_model , n_heads=transformer_n_head, e_layers=transformer_e_layers , d_layers=transformer_d_layers , dropout= transformer_dropout )  # ctx + target = 2*ctx

        else:
            #pass
            #my_conf =    TSConfig(enc_in=train_dict[key]['vals'].shape[-1], dec_in=train_dict[key]['vals'].shape[-1]   , pred_len=cfg.data_reader.pred_len, c_out=num_class, d_model=transformer_d_model   ,n_heads=transformer_n_head, e_layers=transformer_e_layers, d_layers=transformer_d_layers ,dropout=transformer_dropout)  # ctx + target = 2*ctx

            my_conf = NS_TSConfig(enc_in=train_residual_batched_norm.shape[-1], dec_in = train_residual_batched_norm.shape[-1] , pred_len=cfg.data_reader.pred_len , c_out=num_class , d_model=transformer_d_model , n_heads=transformer_n_head, e_layers=transformer_e_layers , d_layers=transformer_d_layers,dropout=transformer_dropout , seq_len=transformer_seq_len ,label_len=transformer_seq_len//2 , factor=transformer_factor , p_hidden_layers=transformer_p_hidden_layers, train_res=cfg.learn.train_residual)  # ctx + target = 2*ctx

        print("Transformer_config:",my_conf)

        #m = LongTermModel(my_conf) # Vanila transformer
        m = Model(my_conf) # NS_transformer


        #hidden_MLP_size = [80, 120, 80]
        #hidden_MLP_size = [360, 3440, 240]
        hidden_MLP_size = [360, 1020, 240]
        if cfg.learn.loss != 'cross_entropy':
            m = MLP(input_size=60, hidden_sizes=hidden_MLP_size, output_size=cfg.data_reader.pred_len) ## m = MLP(input_size=60, hidden_sizes=[120, 180, 120], output_size=cfg.data_reader.pred_len)
            expName = cfg.wandb.exp_name + "_f-in:" + str(train_residual_batched_norm.shape[-1]) + "_epoch" + str(
            cfg.learn.epochs) + "_Loss-" + str(cfg.learn.loss) + "_ctx-" + str(cfg.data_reader.context_size) + "to" + str(cfg.data_reader.pred_len) + "_lr" + str(
            cfg.learn.lr) +"_seed"+str(cfg.learn.seed) + "MLP_arch:" + str ( [cfg.data_reader.context_size] + hidden_MLP_size + [cfg.data_reader.pred_len])
        else:
            m = MLP(input_size=60, hidden_sizes=hidden_MLP_size, output_size=num_class)
            expName = cfg.wandb.exp_name + "_f-in:" + str(train_residual_batched_norm.shape[-1]) + "_epoch" + str(
            cfg.learn.epochs) + "_Loss-" + str(cfg.learn.loss) + "_ctx-" + str(
            cfg.data_reader.context_size) + "to" + str(cfg.data_reader.pred_len) + "_lr" + str(
            cfg.learn.lr) +"_seed"+str(cfg.learn.seed) + "MLP_arch:" + str ([cfg.data_reader.context_size] + hidden_MLP_size + [cfg.learn.num_class])



        n_params = sum(p.numel() for p in m.parameters())
        param_txt = str(n_params/1e6)[:6] +"M" #number of parameters in Milions
        print(param_txt)
        print("number of parameters: %.2fM" % (n_params / 1e6))
        save_path = os.getcwd() + '/experiments/saved_models/' + expName +"_"+param_txt + '.ckpt'

        if torch.cuda.device_count() > 1:
            print("We have available ", torch.cuda.device_count(), "GPUs!")
            parellel_net = nn.DataParallel(m, device_ids=[0, 1, 2, 3])
        elif (torch.cuda.device_count() ==1):
            print('only 1 GPU is avalable!')
            parellel_net = m
        else: #cpu
            parellel_net = m

        #input        = input.to(0)
        #parallel_net = parellel_net.to(0)
        parallel_net = parellel_net.to(device)

        ## Initializing wandb object and sweep object
        wandb_run = wandb.init(project=cfg.wandb.project_name, name=expName+"_"+param_txt,mode=mode)  # wandb object has a set of configs associated with it as well



        if cfg.learn.loss != 'cross_entropy':
            transformer_learn = transformer_trainer.Learn(parallel_net, loss=cfg.learn.loss, imp=impu, config=cfg, run=wandb_run,log=cfg.wandb['log'],normalizer=None  )
        else:
            transformer_learn = transformer_trainer.Learn(parallel_net, loss=cfg.learn.loss, imp=impu, config=cfg, run=wandb_run,log=cfg.wandb['log'],normalizer=normalizer , num_class=num_class )




        ############ statistical tests before training
        print("==============================initial chisq test================================")
        print("checking data = signal + noise    dependency...")


        results0, pvl0, cnt_dep0= run_test(train_residual_batched_norm.squeeze().swapaxes(0, 1),number_output_functions= cfg.data_reader.pred_len, log=False, bonfer=True)
        #print("te_batched.shape:", train_dict[key]['vals'].shape)
        # print('len_dep:', cnt_dep0)
        # print('chisq_dependent:', results0)
        # print('pv:', pvl0)


        results1, pvl1,cnt_dep1 = run_test(test_residual_batched_norm.squeeze().swapaxes(0, 1),number_output_functions= cfg.data_reader.pred_len, log=False, bonfer=True)
        # print("te_batched.shape:", test_residual_batched_norm.shape)
        # print('len_dep:', cnt_dep1)
        # print('chisq_dependent:', results1)
        # print('pv:', pvl1)




        print("=========================  initial MI_test  ===================================")


        _, SUM_MI_initial_test, init_MI_pv_test, avg_MI_initial_test_permute, SUM_MI_initial_test_permuted = get_mutual_information(test_residual_batched_norm.squeeze().swapaxes(0, 1), number_output_functions=cfg.data_reader.pred_len,perm_test_flag=True, N=100)
        # print("SUM_MI_initial_test_permuted", SUM_MI_initial_test_permuted)
        # print('SUM_MI_initial_test:', SUM_MI_initial_test)
        # print("initial MI is less than", init_MI_pv_test ,"% of the MI in a random permutations")

        #######
        _, SUM_MI_initial_train, init_MI_pv_train, avg_MI_initial_train_permute, SUM_MI_initial_train_permuted = get_mutual_information(train_residual_batched_norm.squeeze().swapaxes(0, 1),number_output_functions= cfg.data_reader.pred_len, perm_test_flag=True, N=100)
        # print('SUM_MI_initial_train:', SUM_MI_initial_train)
        # print("initial MI is less than", init_MI_pv_train ,"% of the MI in a random permutations")
        #######

        print("=========================  initial pearson_test  ===================================")
        sum_r_test_init = linear_correl(test_residual_batched_norm.squeeze().swapaxes(0, 1))

        ############ statistical tests done
        print("#################################### statistical tests done ####################################")

        if cfg.learn.load == False:
            #### Train the Model
            #transformer_learn.train(train_obs, train_act, train_targets, train_task_idx, cfg.learn.epochs, cfg.learn.batch_size, test_obs, test_act,test_targets, test_task_idx)
            transformer_learn.train(train_residual_batched_norm, cfg.learn.epochs, cfg.learn.batch_size, test_residual_batched_norm) #train_residual_batched_norm = train_dict[key]['vals']

        parallel_net.load_state_dict(torch.load(save_path, map_location=device))
        print('>>>>>>>>>>Loaded The Model From Local Folder<<<<<<<<<<<<<<<<<<<')



        ########

        if cfg.learn.loss != 'cross_entropy':
            transformer_infer = transformer_inference.Infer(parallel_net, config=cfg, run=wandb_run)
        else:
            transformer_infer = transformer_inference.Infer(parallel_net, config=cfg, run=wandb_run , normalizer=normalizer)
        batch_size = 450
        k = cfg.data_reader.context_size #=context_size=75


        print(" Test started......")
        result={}

        # the last _ is residual list
        pred_mean, _ , gt_multi, observed_part_te, residual_target_te = transformer_infer.predict_mbrl(test_residual_batched_norm,  k=k,batch_size=batch_size, tar='packets' )  # returns normalized predicted packets
        # print("observed_part_test.shape:", observed_part_te.shape)
        # print("residual_target_test.shape:", residual_target_te.shape)

        residual_ctx_and_tar_te = torch.cat([observed_part_te, residual_target_te], dim=1)
        # print("residual_ctx_and_tar_te:", residual_ctx_and_tar_te.shape)


        pred_mean_tr, _, gt_multi_tr, observed_part_tr, residual_target_tr =  transformer_infer.predict_mbrl(train_residual_batched_norm, k=k,batch_size=batch_size,tar='packets')  # returns normalized predicted packets
        residual_ctx_and_tar_tr = torch.cat([observed_part_tr, residual_target_tr], dim=1)
        # print("residual_ctx_and_tar_tr.shape:", residual_ctx_and_tar_tr.shape)



        ###### saving residuals######

        #initialize  empty arrays to use it to store denotmalized versions
        window_with_residual_te_denorm = torch.zeros_like(residual_ctx_and_tar_te)
        window_with_residual_tr_denorm = torch.zeros_like(residual_ctx_and_tar_tr)
        #[1375,50,1]
        window_with_residual_te_denorm[:,:-cfg.data_reader.pred_len,:] = (observed_part_te * torch.tensor(normalizer['sigma'])  +  torch.tensor(normalizer['mean']))    # !!!!   dont add the mean to the residuals// only multiply by std , only add mean to the context        flowise_normalizer[key][0]
        window_with_residual_te_denorm[:, -cfg.data_reader.pred_len:,:] = (residual_target_te * torch.tensor(normalizer['sigma']))

        window_with_residual_tr_denorm[:,:-cfg.data_reader.pred_len,:] = (observed_part_tr * torch.tensor(normalizer['sigma']))  +  torch.tensor(normalizer['mean'])  #!!!!        dont add the mean to the residuals // only multiply by std , only add mean to the context      flowise_normalizer[key][0]
        window_with_residual_tr_denorm[:, -cfg.data_reader.pred_len:,:] = (residual_target_tr * torch.tensor(normalizer['sigma']))

        model_name = cfg.learn.model_name
        output_dir = os.path.join(os.getcwd(),'experiments','residuals', 'stacking' , model_name + "_ctx-" + str(cfg.data_reader.context_size) + "to" + str(cfg.data_reader.pred_len))

        directory = os.path.join(output_dir)
        f_name = os.path.join(output_dir,model_name + '_tr_' + str(key) + '.pickle')
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(f_name, 'wb') as handle7:
            pickle.dump(window_with_residual_tr_denorm, handle7, protocol=pickle.HIGHEST_PROTOCOL)
            print("train_res_"+str(key)+" saved")


        f_name = os.path.join(output_dir, model_name + '_te_' + str(key) + '.pickle')
        with open(f_name, 'wb') as handle8:
            pickle.dump(window_with_residual_te_denorm, handle8, protocol=pickle.HIGHEST_PROTOCOL)
            print("test_res_"+str(key)+" saved")

        print("============================================ Residual windows saved ============================================")




        ######################### checking all statistics measures after prediction is reduced


        print(" ======================chi-square on residual_test:=============================")
        results2, pvl2 ,cnt_dep2 = run_test(residual_ctx_and_tar_te.numpy().squeeze().swapaxes(0, 1),number_output_functions= cfg.data_reader.pred_len, log=False, bonfer=True)
        # print(cnt_dep2, ':-Test-data->', results2)
        # print(pvl2)

        results3, pvl3,cnt_dep3 = run_test(residual_ctx_and_tar_tr.numpy().squeeze().swapaxes(0,1),number_output_functions= cfg.data_reader.pred_len, log=False, bonfer=True)
        # print(cnt_dep3,':-Train-data->',results3)
        # print(pvl3)



        print("=========================MI and permutation on residual_test ============================")



        _, sum_res_test , res_MI_pv_test, avg_MI_res_test_permute, sum_res_test_permuted = get_mutual_information(residual_ctx_and_tar_te.numpy().squeeze().swapaxes(0, 1), number_output_functions=cfg.data_reader.pred_len, perm_test_flag=True, N=100)
        # print("sum_res_test", sum_res_test)
        # print("sum_res_test_permuted", sum_res_test_permuted)
        # print("test_res_MI is less than ", res_MI_pv_test ," % of the MI in a random permutations")



        _, sum_res_train , res_MI_pv_train, avg_MI_res_train_permute, sum_res_train_permuted = get_mutual_information(residual_ctx_and_tar_tr.numpy().squeeze().swapaxes(0, 1), number_output_functions=cfg.data_reader.pred_len, perm_test_flag=True, N=100)
        # print("train_res_MI is less than ", res_MI_pv_train , " % of the MI in a random permutations")


        print("============================= MI and permutation on residual_test  done ==========================")

        print("==================================== Pearson on test_res ===================================")
        sum_r_test_res = linear_correl(residual_ctx_and_tar_tr.numpy().squeeze().swapaxes(0, 1))

        ### create a dict to save this wandb logs

        # ######## wandb logs
        # wb_logs = {}
        # wb_logs[key] = {}
        # wb_logs[key]['init_dep_vars_test'] = results1
        # wb_logs[key]['init_num_dep_vars_test'] = int(len(results1))
        # wb_logs[key]['init_dep_vars_test_pv'] = pvl1
        #
        # wb_logs[key]['res_dep_vars_test'] = results2
        # wb_logs[key]['res_num_dep_vars_test'] = int(len(results2))
        # wb_logs[key]['res_dep_vars_test_pv'] = pvl2
        # wb_logs[key]['dep_vars_train_res'] = results3
        # wb_logs[key]['num_dep_vars_train_res'] = int(len(results3))
        # ################# MI logs #################
        # wb_logs[key]['res_SUM_MI_test'] = sum_res_test
        #
        # wb_logs[key]['res_SUM_MI_test_perm'] = list(sum_res_test_permuted)
        # wb_logs[key]['res_SUM_MI_test_perm_min'] = np.min(np.array(sum_res_test_permuted))
        # wb_logs[key]['res_SUM_MI_test_perm_max'] = np.max(np.array(sum_res_test_permuted))
        # wb_logs[key]['res_SUM_MI_test_perm_std'] = np.std(np.array(sum_res_test_permuted))
        # wb_logs[key]['res_SUM_MI_test_perm_avg'] = np.mean(np.array(sum_res_test_permuted))
        #
        # wb_logs[key]['SUM_MI_train_res_perm'] = list(sum_res_train_permuted)
        #
        # wb_logs[key]['init_SUM_MI_test'] = SUM_MI_initial_test
        # wb_logs[key]['init_SUM_MI_train'] = SUM_MI_initial_train
        #
        # wb_logs[key]['init_SUM_MI_test_perm'] = list(SUM_MI_initial_test_permuted)
        # wb_logs[key]['init_SUM_MI_test_perm_min'] = np.min(np.array(SUM_MI_initial_test_permuted))
        # wb_logs[key]['init_SUM_MI_test_perm_max'] = np.max(np.array(SUM_MI_initial_test_permuted))
        # wb_logs[key]['init_SUM_MI_test_perm_std'] = np.std(np.array(SUM_MI_initial_test_permuted))
        # wb_logs[key]['init_SUM_MI_test_perm_avg'] = np.mean(np.array(SUM_MI_initial_test_permuted))
        # wb_logs[key]['init_SUM_MI_train_perm'] = list(SUM_MI_initial_train_permuted)
        #
        # #### MI logs p-values
        # wb_logs[key]['init_MI_pval_train '] = init_MI_pv_train
        # wb_logs[key]['init_MI_pval_test '] = init_MI_pv_test
        # wb_logs[key]['res_MI_pval_train '] = res_MI_pv_train
        # wb_logs[key]['res_MI_pval_test '] = res_MI_pv_test
        #
        # ####################### pearson logs ##########################
        # wb_logs[key]['init_pearson_test'] = sum_r_test_init
        # wb_logs[key]['res_pearson_test'] = sum_r_test_res
        #
        # directory = os.path.join(os.getcwd(), 'experiment_wb_logs', 'wb_logs_' + str(model_name))
        # saved_dict_file_name = os.path.join(directory, str(key) + ".pickle")
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # with open(saved_dict_file_name, 'wb') as handle:
        #     pickle.dump(wb_logs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print("dictionary saved ...................... in ",saved_dict_file_name)








        ################# chi-square logs #################
        wandb.run.summary['init_chisq_dep_vars_train'] = results0
        wandb.run.summary['init_chisq_num_dep_vars_train'] = int(cnt_dep0)
        wandb.run.summary['init_chisq_dep_vars_train_pv'] = pvl0

        wandb.run.summary['init_chisq_dep_vars_test_list'] = results1
        wandb.run.summary['init_chisq_num_dep_vars_test'] = int(cnt_dep1)
        wandb.run.summary['init_chisq_dep_vars_test_pv'] = pvl1

        wandb.run.summary['res_chisq_dep_vars_test_list'] = results2
        wandb.run.summary['res_chisq_num_dep_vars_test'] = int(cnt_dep2)
        wandb.run.summary['res_chisq_dep_vars_test_pv'] = pvl2

        wandb.run.summary['res_chisq_dep_vars_train_list'] = results3
        wandb.run.summary['res_chisq_num_dep_vars_train'] = int(cnt_dep3)
        wandb.run.summary['res_chisq_dep_vars_pv']    = pvl3

        ################# MI logs #################
        wandb.run.summary['res_SUM_MI_test'] = sum_res_test
        wandb.run.summary['res_SUM_MI_train'] = sum_res_train

        wandb.run.summary['res_SUM_MI_test_perm'] = list(sum_res_test_permuted)
        wandb.run.summary['res_SUM_MI_test_perm_min'] = np.min(np.array(sum_res_test_permuted))
        wandb.run.summary['res_SUM_MI_test_perm_max'] = np.max(np.array(sum_res_test_permuted))
        wandb.run.summary['res_SUM_MI_test_perm_std'] = np.std(np.array(sum_res_test_permuted))
        wandb.run.summary['res_SUM_MI_test_perm_avg'] = np.mean(np.array(sum_res_test_permuted))

        wandb.run.summary['res_SUM_MI_train_perm_avg'] = np.mean(np.array(sum_res_train_permuted))

        wandb.run.summary['res_SUM_MI_train_minus_perm_avg'] = sum_res_train - np.mean(np.array(sum_res_train_permuted))
        wandb.run.summary['init_SUM_MI_train_minus_perm_avg']= SUM_MI_initial_train - np.mean(np.array(SUM_MI_initial_train_permuted))

        wandb.run.summary['res_SUM_MI_test_minus_perm_avg'] = sum_res_test - np.mean(np.array(sum_res_test_permuted))
        wandb.run.summary['init_SUM_MI_test_minus_perm_avg'] = SUM_MI_initial_test - np.mean(np.array(SUM_MI_initial_test_permuted))



        # wandb.run.summary['SUM_MI_train_res_perm'] = list(sum_res_train_permuted)

        wandb.run.summary['init_SUM_MI_test'] = SUM_MI_initial_test
        wandb.run.summary['init_SUM_MI_train'] = SUM_MI_initial_train

        wandb.run.summary['init_SUM_MI_test_perm']     = list(SUM_MI_initial_test_permuted)
        wandb.run.summary['init_SUM_MI_test_perm_min'] = np.min(np.array(SUM_MI_initial_test_permuted))
        wandb.run.summary['init_SUM_MI_test_perm_max'] = np.max(np.array(SUM_MI_initial_test_permuted))
        wandb.run.summary['init_SUM_MI_test_perm_std'] = np.std(np.array(SUM_MI_initial_test_permuted))
        wandb.run.summary['init_SUM_MI_test_perm_avg'] = np.mean(np.array(SUM_MI_initial_test_permuted))

        wandb.run.summary['init_SUM_MI_train_perm_avg'] = np.mean(np.array(SUM_MI_initial_train_permuted))
                #### MI logs p-values
        wandb.run.summary['init_MI_pval_train '] = init_MI_pv_train
        wandb.run.summary['init_MI_pval_test '] = init_MI_pv_test
        wandb.run.summary['res_MI_pval_train '] = res_MI_pv_train
        wandb.run.summary['res_MI_pval_test '] = res_MI_pv_test




        ####################### pearson logs ##########################
        wandb.run.summary['init_pearson_test'] = sum_r_test_init
        wandb.run.summary['res_pearson_test'] = sum_r_test_res

        ####################### initial values ##############
        #wandb.run.summary['sin_pow'] = math.sqrt(sin_pow)
        #wandb.run.summary['init_rel_noise_pow_std'] = math.sqrt(relative_noise_power)
        try:
            wandb.run.summary['random_seed'] = int(cfg.lean.seed)
            #wb_logs[key]['random_seed'] = int(cfg.lean.seed)
        except:
            pass
        # wandb.run.summary['noise_pow'] = math.sqrt(noise_pow)
        # wandb.run.summary['noise_to_sig'] = math.sqrt(relative_noise_power)




        if cfg.data_reader.pred_len == 1: # plot the predictions on the whole data using running window

            print("loading predictions and residuals of previous model .....")
            old_pred_name = cfg.learn.load_dir_residual   ## MLPL2_candidates
            load_flattend_dir = os.path.join(os.getcwd(), 'experiments', 'flattened_pred', 'stack_0_', old_pred_name) # the address  where we load the flattened pred

            f_name = os.path.join(load_flattend_dir, old_pred_name + '_pred_tr_flat_' + str(key) + '.pickle')
            with open(f_name, 'rb') as file1:
                pred_tr_flat_old = pickle.load(file1)

            f_name = os.path.join(load_flattend_dir, old_pred_name + '_pred_te_flat_' + str(key) + '.pickle')
            with open(f_name, 'rb') as file2:
                pred_te_flat_old = pickle.load(file2)

            f_name = os.path.join(load_flattend_dir, old_pred_name + '_gt_tr_flat_' + str(key) + '.pickle')
            with open(f_name, 'rb') as file3:
                gt_tr_flat_old = pickle.load(file3)

            f_name = os.path.join(load_flattend_dir, old_pred_name+ '_gt_te_flat_' + str(key) + '.pickle')
            with open(f_name, 'rb') as file4:
                gt_te_flat_old = pickle.load(file4)


            f_name = os.path.join(load_flattend_dir, old_pred_name + '_res_tr_flat_' + str(key) + '.pickle')
            with open(f_name, 'rb') as file5:
                res_tr_flat_old = pickle.load(file5)

            f_name = os.path.join(load_flattend_dir, old_pred_name+ '_res_te_flat_' + str(key) + '.pickle')
            with open(f_name, 'rb') as file6:
                res_te_flat_old = pickle.load(file6)




            # Running window prediction on Test
            pred_mean_te_rw, _, gt_multi_te_rw, residual_te_rw = transformer_infer.predict_rw(torch.from_numpy(test_obs_arr_norm[key]), k=k, batch_size=batch_size,tar='packets')  # returns normalized predicted packets

            pred_test_new = pred_mean_te_rw
            gt_test_old = gt_multi_te_rw  # it  is old gt (or original gt) becuase they are calculated as a last point of a window in the original arr
            residual_test_new = residual_te_rw # it is not sensible becuase it is calculated based on the difference of old_gt and new_prediction

            ##Running window prediction on Train & plot
            pred_mean_tr_rw, _, gt_multi_tr_rw, residual_tr_rw = transformer_infer.predict_rw(torch.from_numpy(train_obs_arr_norm[key]), k=k, batch_size=batch_size,tar='packets')  # returns normalized predicted packets
            pred_train_new = pred_mean_tr_rw
            gt_train_old = gt_multi_tr_rw      # it  is old gt (or original gt) becuase they are calculated as a last point of a window in the original arr
            residual_train_new = residual_tr_rw # it is not sensible becuase it is calculated based on the difference of old_gt and new_prediction

            print("make sure gt_train_old==gt_tr_flat_old  !!!!!!!!!!")
            # trivial plot (only based on the new model)

            # ## Train and test  plot
            #
            fig1, axs = plt.subplots(2, 1, figsize=(45, 15*2))
            fig1.suptitle(key + "_1step ahead prediction on residuals", fontsize=16)
            axs[0].plot(pred_test_new, label="new_prediction")
            axs[0].plot(res_te_flat_old, label="new_groundtruth") # old residual is now the new ground truth
            axs[0].plot(res_te_flat_old - pred_test_new , label="new_residual")
            # axs[0].plot(residual_test, label="residual")
            axs[0].legend(prop={'size': 25})
            axs[0].set_title('Test' , fontsize=25)
            axs[0].tick_params(axis='both', labelsize=20)
            #
            axs[1].plot(pred_train_new, label="new_prediction")
            axs[1].plot(res_tr_flat_old, label="groundtruth")
            axs[1].plot(res_tr_flat_old - pred_train_new, label="new_residual")
            axs[1].legend(prop={'size': 25})
            axs[1].set_title('Train', fontsize=25)
            axs[1].tick_params(axis='both', labelsize=20)

            plot_name2 = key + "_1step_ahead_prediction" + str(cfg.wandb.exp_name) + ".pdf"
            fig1.savefig(plot_name2)
            plot_name = key + "_1step_ahead_prediction"+ str(cfg.wandb.exp_name) +".png"
            fig1.savefig(plot_name)
            plt.show()
            plt.close('all')

            if wandb_run is not None:
                keyy = str(key) + "_Qualitative-Results"
                print("wandb uploading Qualitative-result plot...")
                image = plt.imread(plot_name)
                wandb_run.log({keyy: wandb.Image(image)})



            # some of two models plot

            fig1, axs = plt.subplots(2, 1, figsize=(45, 15*2))
            fig1.suptitle(key + "_1step ahead prediction on combination", fontsize=16)

            axs[0].plot(pred_te_flat_old, label="prediction_1")  # old residual is now the new ground truth
            axs[0].plot(pred_test_new, label="prediction_2")
            axs[0].plot(pred_te_flat_old + pred_test_new , label="all_prediction")
            axs[0].plot(gt_test_old , label="original_gt")
            #axs[0].plot(gt_te_flat_old, label="original_gt_double_ckeck") # for the sake of double check only

            # axs[0].plot(residual_test, label="residual")
            axs[0].legend(prop={'size': 25})
            axs[0].set_title('Test' , fontsize=25)
            axs[0].tick_params(axis='both', labelsize=20)
            #


            axs[1].plot(pred_tr_flat_old, label="prediction_1")  # old residual is now the new ground truth
            axs[1].plot(pred_train_new, label="prediction_2")
            axs[1].plot(pred_tr_flat_old + pred_train_new , label="all_prediction")
            axs[1].plot(gt_train_old , label="original_gt")
            #axs[1].plot(gt_tr_flat_old, label="original_gt_double_ckeck") # for the sake of double check only

            axs[1].legend(prop={'size': 25})
            axs[1].set_title('Train', fontsize=25)
            axs[1].tick_params(axis='both', labelsize=20)

            plot_name2 = key + "_combined_1step_ahead_prediction" + str(cfg.wandb.exp_name) + ".pdf"
            fig1.savefig(plot_name2)
            plot_name = key + "_combined_1step_ahead_prediction"+ str(cfg.wandb.exp_name) +".png"
            fig1.savefig(plot_name)
            plt.show()
            plt.close('all')

            if wandb_run is not None:
                keyy = str(key) + "combined_Qualitative-Results"
                print("wandb uploading Qualitative-result plot...")
                image = plt.imread(plot_name)
                wandb_run.log({keyy: wandb.Image(image)})





        print("finished ... ")


        plt.close('all')
        return result


def main():
    my_app()



## https://stackoverflow.com/questions/32761999/how-to-pass-an-entire-list-as-command-line-argument-in-python/32763023
if __name__ == '__main__':
    main()