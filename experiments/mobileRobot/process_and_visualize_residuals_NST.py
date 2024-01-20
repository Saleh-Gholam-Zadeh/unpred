

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
#from transformer_architecture.models.Transformer_Longterm import  LongTermModel, TSConfig
from transformer_architecture.ns_models.ns_Transformer import  Model, NS_TSConfig
#from mlp_arcitecture.mlp_arch import MLP
#from synthetic_data_gen import *
from  utils.dataProcess import ts2batch , get_statistics , normalize , denormalize
import fnmatch


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
number_of_class =20
print(number_of_class)

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
        print('trying to load the file  selected_result_dict.pickle ')
        with open('selected_result_dict.pickle', 'rb') as file2:
            selected_result = pickle.load(file2)  # {'flow_331': [array([0.01563125]), array([0.17415701]), 320000]}
            print(selected_result.keys())
            selected_keys=list(selected_result.keys())

        print('selected_result_dict.pickle loaded ')

        self.all_flows = cfg.data_reader.flows    #list(selected_result.keys()) #cfg.data_reader.flows
        #num_keys_to_take = len(self.all_flows) // 10
        #i = cfg.data_reader.portion
        #start_index = (i) * num_keys_to_take
        #end_index = (i+1) * num_keys_to_take

        #selected_keys = self.all_flows[start_index:end_index]
        #start = selected_keys.index("flow_994")
        #self.all_flows = selected_keys[start:start+1]
        #self.all_flows = selected_keys[26:27]
        #self.all_flows = selected_keys[-30:-20] #laptop
        #self.all_flows = selected_keys[-1:]
        print("all_flows:",self.all_flows)

        # old code
        if self.all_flows != 'None': # should come from the list of selected keys:
            for flow in self.all_flows:
                partial_result= self._experiment(flow)
                #self._experiment(flow)
                agg_result_dict.update(partial_result)
                wandb.finish()
                print("Next CPU - ------------------------------------------------------------------------------------------------------------------------------")
            print('finished all flows')
            # print("saving final results dict")
            # with open('MLP_all_of_selected_ones_result_dict_' + '.pickle', 'wb') as handle9:
            #     pickle.dump(agg_result_dict, handle9, protocol=pickle.HIGHEST_PROTOCOL)
            #     print("agg results saved successfully ")
        else:
            self._experiment()
            wandb.finish()
        #/ old code





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

        model_name = 'NST_mae_stack0'

        tar_type = cfg.data_reader.tar_type  # 'delta' - if to train on differences to current states
        # 'next_state' - if to trian directly on the  next states

        #data = metaMobileData(cfg.data_reader) ##SALEH commented out because we need access to all of the config parameters
        data = metaMobileData(cfg) #SALEH ADDED
        # train_targets and test_targets  (based on the config) can be 1st difference
        mb = cfg.data_reader.meta_batch_size
        features_in  = cfg.learn.features_in
        features_out = cfg.learn.features_out


        txt_train = 'train_dict.pickle'
        txt_test =  'test_dict.pickle'
        txt_norm =  'packet_normalizer.pickle'


        if cfg.data_reader.load:

            # with open('train_dict_normalized.pickle', 'rb') as file1:
            #     train_dict = pickle.load(file1)
            #     if flow is not None:
            #         train_dict = {key: train_dict[key]}
            with open('train_dict_cpu.pickle', 'rb') as file1:
                train_dict_unorm = pickle.load(file1)
                if flow is not None:
                    train_dict_unorm = {key: train_dict_unorm[key]}

            with open('test_dict_cpu.pickle', 'rb') as file2:
                test_dict_unorm = pickle.load(file2)
                if flow is not None:
                    test_dict_unorm = {key:test_dict_unorm[key]}

            with open('train_dict_normalized.pickle', 'rb') as file3:
                train_dict_norm = pickle.load(file3)
                if flow is not None:
                    train_dict_norm = {key: train_dict_norm[key]}

            with open('test_dict_normalized.pickle', 'rb') as file4:
                test_dict_norm = pickle.load(file4)
                if flow is not None:
                    test_dict_norm= {key:test_dict_norm[key]}

            with open('train_obs_arr.pickle', 'rb') as file5:
                train_obs_arr = pickle.load(file5)
                if flow is not None:
                    num=int(key.split('_')[-1])
                    train_obs_arr = {key:train_obs_arr[num]}

            with open('test_obs_arr.pickle', 'rb') as file6:
                test_obs_arr = pickle.load(file6)
                if flow is not None:
                    num=int(key.split('_')[-1])
                    test_obs_arr = {key:test_obs_arr[num]}

            #
            # with open('global_packet_normalizer.pickle', 'rb') as file3:
            #     packet_norm = pickle.load(file3)

            with open('chi_square_original_dict.pickle', 'rb') as file7: # contains only selected keys
                chi_square_original_dict = pickle.load(file7)
                if flow is not None: # chi_square_original_dict['flow_0']['train_windows'].shape = [1374,50,1] ---> Not Normalized
                    chi_square_original_dict = {key:chi_square_original_dict[key]}


            with open('chi_square_residual_dict'+'_' + str(model_name)+'.pickle', 'rb') as file8:
                chi_square_residual_dict = pickle.load(file8)
                if flow is not None:
                    chi_square_residual_dict = {key:chi_square_residual_dict[key]}

            with open('cpuwise_normalizer.pickle', 'rb') as file9:
                flowise_normalizer = pickle.load(file9)  # {'flow_331': [array([0.01563125]), array([0.17415701]), 320000]}
                if flow is not None:
                    flowise_normalizer = {key: flowise_normalizer[key]}

            print("flowise_normalization of obs_arr ....")
            train_dict = {}
            test_dict  = {}
            train_obs_arr_norm={}
            test_obs_arr_norm={}
            for i, (key, inner_dict) in enumerate(chi_square_original_dict.items()):
                num_key = int(key.split('_')[-1])
                print('key:',key, 'num:',num_key)
                train_dict[key]= train_dict_unorm[key].copy()
                train_dict[key]['vals'] = (train_dict[key]['vals'] - flowise_normalizer[key][0])/flowise_normalizer[key][1]

                test_dict[key] = test_dict_unorm[key].copy()
                test_dict[key]['vals'] = (test_dict[key]['vals'] - flowise_normalizer[key][0]) / flowise_normalizer[key][1]

                num_key = int(key.split('_')[-1])
                train_obs_arr_norm[key] = train_obs_arr[num_key,:,:].copy()
                train_obs_arr_norm[key] = (train_obs_arr_norm[key] - flowise_normalizer[key][0])/flowise_normalizer[key][1]

                test_obs_arr_norm[key] = test_obs_arr[num_key,:,:].copy()
                test_obs_arr_norm[key] = (test_obs_arr_norm[key] - flowise_normalizer[key][0])/flowise_normalizer[key][1]

                # to double check
                # plt.plot(train_obs_arr[num_key, :50, :])
                # plt.plot(train_dict_unorm[key]['vals'][0, :, :])  # [1374,50,1]
                #
                # plt.show()



            print("obs_arr normalization done")


        else:
            print('================================================')
            pass

        print('..................filtering the dicts (based on intersection )..............')
        right_keys = []
        left_keys = []

        #chi_square_residual_dict
        for i,(key , inner_dict) in enumerate(chi_square_residual_dict.items()):
            print("key:",key)
            #print(len(chi_square_residual_dict[key]['dep_intersect']))
            if (len(chi_square_residual_dict[key]['dep_intersect']) >= 45):
                right_keys.append(key)
            if (len(chi_square_residual_dict[key]['dep_intersect']) <= 5):
                left_keys.append(key)
        print("len(right_keys):",len(right_keys))
        print("len(left_keys):", len(left_keys))
        print(left_keys)
        print(right_keys)
        print('..................filtering done .................')

        # print('..................filtering the dicts (based on test only )..............')
        # right_keys = []
        # left_keys = []
        #
        # # chi_square_residual_dict
        # for i, (key, inner_dict) in enumerate(chi_square_original_dict.items()):
        #     print("key:", key)
        #     # print(len(chi_square_residual_dict[key]['dep_intersect']))
        #     if (len(chi_square_residual_dict[key]['dep_test']) >= 45):
        #         right_keys.append(key)
        #     if (len(chi_square_residual_dict[key]['dep_test']) <= 5):
        #         left_keys.append(key)
        # print("len(right_keys):", len(right_keys))
        # print("len(left_keys):", len(left_keys))
        # print(left_keys)
        # print(right_keys)
        # print('..................filtering done .................')
        #
        #
        # sys.exit(5)


        transformer_e_layers = cfg.transformer_arch.enc_layer
        transformer_d_layers = cfg.transformer_arch.dec_layer
        transformer_n_head   = cfg.transformer_arch.n_head
        transformer_d_model  = cfg.transformer_arch.d_model
        transformer_dropout = cfg.transformer_arch.dropout
        transformer_seq_len = cfg.transformer_arch.seq_len
        transformer_factor  = cfg.transformer_arch.factor
        transformer_p_hidden_layers =cfg.transformer_arch.p_hidden_layers
        transformer_p_hidden_dims = cfg.transformer_arch.p_hidden_dims

        expName = cfg.wandb.exp_name + "_" + str(model_name)
        try:
            expName = expName +"_e-d-n-Dm-dropout:"+str(transformer_e_layers)+ "-" + str(transformer_d_layers) +"-"+ str(transformer_n_head) +"-"+ str(transformer_d_model)+"-" +str(transformer_dropout) +"_f-in:"+ str(train_dict['flow_1']['vals'].shape[-1])  + "_epoch" + str(cfg.learn.epochs)   + "_Loss-" +   str(cfg.learn.loss)+ "_ctx-"   +  str(cfg.data_reader.context_size)  + "to" + str(cfg.data_reader.pred_len)  + "_lr" + str(cfg.learn.lr)+"_seed"+str(cfg.lean.seed) +"_flow:"+str(flow)
        except:
            expName = expName + "_e-d-n-Dm-dropout:" + str(transformer_e_layers) + "-" + str(
            transformer_d_layers) + "-" + str(transformer_n_head) + "-" + str(transformer_d_model) + "-" + str(
            transformer_dropout) + "_f-in:" + str(train_dict[key]['vals'].shape[-1]) + "_epoch" + str(
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
                 #my_conf = TSConfig(enc_in=train_dict['flow_1']['vals'].shape[-1], dec_in = train_dict['flow_1']['vals'].shape[-1] , pred_len=cfg.data_reader.pred_len , c_out=1 , d_model=transformer_d_model , n_heads=transformer_n_head, e_layers=transformer_e_layers , d_layers=transformer_d_layers , dropout= transformer_dropout)  # ctx + target = 2*ctx

            except:
                print('here')
                my_conf = NS_TSConfig(enc_in=train_dict[key]['vals'].shape[-1], dec_in = train_dict[key]['vals'].shape[-1]           , pred_len=cfg.data_reader.pred_len , c_out=1 , d_model=transformer_d_model , n_heads=transformer_n_head, e_layers=transformer_e_layers , d_layers=transformer_d_layers , dropout= transformer_dropout, seq_len=transformer_seq_len ,label_len=transformer_seq_len//2 , factor=transformer_factor , p_hidden_layers=transformer_p_hidden_layers )  # ctx + target = 2*ctx
                #my_conf = TSConfig(enc_in=train_dict[key]['vals'].shape[-1], dec_in = train_dict[key]['vals'].shape[-1]           , pred_len=cfg.data_reader.pred_len , c_out=1 , d_model=transformer_d_model , n_heads=transformer_n_head, e_layers=transformer_e_layers , d_layers=transformer_d_layers , dropout= transformer_dropout )  # ctx + target = 2*ctx

        else:
            pass
            #my_conf = NS_TSConfig(enc_in=train_obs.shape[-1], dec_in = train_obs.shape[-1] , pred_len=cfg.data_reader.pred_len , c_out=number_of_class , d_model=transformer_d_model , n_heads=transformer_n_head, e_layers=transformer_e_layers , d_layers=transformer_d_layers ,       dropout=transformer_dropout , seq_len=transformer_seq_len ,label_len=transformer_seq_len//2 , factor=transformer_factor , p_hidden_layers=transformer_p_hidden_layers)  # ctx + target = 2*ctx

        print("Transformer_config:",my_conf)


        #print("Transformer_config:", my_conf)

        # m = LongTermModel(my_conf) # Vanila transformer
        m = Model(my_conf) # NS_transformer
        #m = MLP(input_size=49, hidden_sizes=[490, 700, 490], output_size=1)
        # m = MLP(input_size=49,hidden_sizes=[500,500],output_size=1)

        n_params = sum(p.numel() for p in m.parameters())
        param_txt = str(n_params / 1e6)[:5] + "M"  # number of parameters in Milions
        print(param_txt)
        print("number of parameters: %.2fM" % (n_params / 1e6))

        #save_path = os.getcwd() + '/experiments/saved_models/' + expName + "_" + param_txt + '.ckpt'

        if torch.cuda.device_count() > 1:
            print("We have available ", torch.cuda.device_count(), "GPUs!")
            parellel_net = nn.DataParallel(m, device_ids=[0, 1, 2, 3])
        elif (torch.cuda.device_count() == 1):
            print('only 1 GPU is avalable!')
            parellel_net = m
        else:  # cpu
            parellel_net = m

        # input        = input.to(0)
        # parallel_net = parellel_net.to(0)
        parallel_net = parellel_net.to(device)

        ## Initializing wandb object and sweep object
        wandb_run = wandb.init(project=cfg.wandb.project_name, name=expName + "_" + param_txt, mode=mode)  # wandb object has a set of configs associated with it as well



        #
        #
        for i,key in enumerate(right_keys):
            print('..................loading the relevant Model for:'+str(key)+'....'+str(i)+'/'+str(len(right_keys)) +'.............')
            #
            load_dir = os.path.join(os.getcwd(),'experiments','saved_models_' + str(model_name))
            pattern = "NST*"+str(key) +"_0.702M.ckpt"

            # List all files in the directory
            files = os.listdir(load_dir)

            # Iterate through the files and find the one that matches the pattern
            found_file = None
            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    found_file = filename
                    break

            if found_file is not None:
                print(f"Found file with pattern: {found_file}")
            else:
                print(f"No file found with pattern: {pattern}")

            load_dir = os.path.join(load_dir,found_file)
            print('Model:',load_dir)




            parallel_net.load_state_dict(torch.load(load_dir, map_location=device))
            print('>>>>>>>>>>Model_'+str(key)+' is loaded successfully<<<<<<<<<<<<<<<<<<<')
            # print("packet_norm:",packet_norm)

            transformer_infer = transformer_inference.Infer(parallel_net, config=cfg, run=wandb_run)
            batch_size = 450
            k = cfg.data_reader.context_size  # =context_size=75

            #print(" Test started......")
            result = {}
            result_train = {}


            result[key] = {}
            result_train[key] = {}
            inner_dict_train = train_dict[key]

            # pred_mean, _, gt_multi, observed_part, window_with_residual_te = transformer_infer.predict_mbrl(inner_dict, k=k, batch_size=batch_size, tar='packets')  # returns normalized predicted packets
            # pred_mean_tr, _, gt_multi_tr, observed_part_tr, window_with_residual_tr = transformer_infer.predict_mbrl(inner_dict_train, k=k, batch_size=batch_size, tar='packets')  # returns normalized predicted packets

            # Running window prediction on Test
            pred_mean_te_rw, _, gt_multi_te_rw, residual_te_rw = transformer_infer.predict_rw(torch.from_numpy(test_obs_arr_norm[key]), k=k, batch_size=batch_size,tar='packets')  # returns normalized predicted packets

            pred_test = pred_mean_te_rw
            gt_test = gt_multi_te_rw
            residual_test = residual_te_rw

            ##Running window prediction on Train & plot
            pred_mean_tr_rw, _, gt_multi_tr_rw, residual_tr_rw = transformer_infer.predict_rw(torch.from_numpy(train_obs_arr_norm[key]), k=k, batch_size=batch_size,tar='packets')  # returns normalized predicted packets

            pred_train = pred_mean_tr_rw
            gt_train = gt_multi_tr_rw
            residual_train = residual_tr_rw

            fig1, axs = plt.subplots(2, 1, figsize=(45, 15 * 2))
            fig1.suptitle(key + "_1step ahead prediction", fontsize=16)
            axs[0].plot(pred_test[:, :], label="prediction")
            axs[0].plot(gt_test[:, :], label="groundtruth")
            axs[0].plot(residual_test[:, :], label="residual")
            axs[0].legend()
            axs[0].set_title('Test')

            axs[1].plot(pred_train[:, :], label="prediction")
            axs[1].plot(gt_train[:, :], label="groundtruth")
            axs[1].plot(residual_train[:, :], label="residual")
            axs[1].legend()
            axs[1].set_title('Train')


            plot_name = key + "_NST_1step_ahead_prediction_on_remained_residual.png"
            directory = os.path.join(os.getcwd(), 'gt_pred_residual_plots',str(model_name),'right_side')
            address = os.path.join(directory,plot_name)
            if not(os.path.exists(directory)):
                os.makedirs(directory)
            fig1.savefig(address)
            #plt.show()

            try:
                if wandb_run is not None:
                    keyy = str(key) + "_Qualitative-Results"
                    #print("wandb uploading Qualitative-result plot...")
                    image = plt.imread(address)
                    wandb_run.log({keyy: wandb.Image(image)})
            except Exception as e:
                print(e)





        for i,key in enumerate(left_keys):
            print('..................loading the relevant Model for:'+str(key)+'....'+str(i)+'/'+str(len(left_keys)) +'.............')
            #
            load_dir = os.path.join(os.getcwd(),'experiments','saved_models_' + str(model_name))
            pattern = "NST*"+str(key) +"_0.702M.ckpt"

            # List all files in the directory
            files = os.listdir(load_dir)

            # Iterate through the files and find the one that matches the pattern
            found_file = None
            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    found_file = filename
                    break

            if found_file is not None:
                print(f"Found file with pattern: {found_file}")
            else:
                print(f"No file found with pattern: {pattern}")

            load_dir = os.path.join(load_dir,found_file)
            print('Model:',load_dir)






            parallel_net.load_state_dict(torch.load(load_dir, map_location=device))
            print('>>>>>>>>>>Model_'+str(key)+' is loaded successfully<<<<<<<<<<<<<<<<<<<')
            # print("packet_norm:",packet_norm)

            transformer_infer = transformer_inference.Infer(parallel_net, config=cfg, run=wandb_run)
            batch_size = 450
            k = cfg.data_reader.context_size  # =context_size=75

            #print(" Test started......")
            result = {}
            result_train = {}
            multiSteps = [1]
            # test target is not used at all in the next line
            # fig, axs = plt.subplots(4, figsize=(20, 40))

            #for key, inner_dict in test_dict.items():
            result[key] = {}
            result_train[key] = {}
            inner_dict_train = train_dict[key]

            # pred_mean, _, gt_multi, observed_part, window_with_residual_te = transformer_infer.predict_mbrl(inner_dict, k=k, batch_size=batch_size, tar='packets')  # returns normalized predicted packets
            # pred_mean_tr, _, gt_multi_tr, observed_part_tr, window_with_residual_tr = transformer_infer.predict_mbrl(inner_dict_train, k=k, batch_size=batch_size, tar='packets')  # returns normalized predicted packets

            # Running window prediction on Test
            pred_mean_te_rw, _, gt_multi_te_rw, residual_te_rw = transformer_infer.predict_rw(torch.from_numpy(test_obs_arr_norm[key]), k=k, batch_size=batch_size,tar='packets')  # returns normalized predicted packets

            pred_test = pred_mean_te_rw
            gt_test = gt_multi_te_rw
            residual_test = residual_te_rw

            ##Running window prediction on Train & plot
            pred_mean_tr_rw, _, gt_multi_tr_rw, residual_tr_rw = transformer_infer.predict_rw(torch.from_numpy(train_obs_arr_norm[key]), k=k, batch_size=batch_size,tar='packets')  # returns normalized predicted packets

            pred_train = pred_mean_tr_rw
            gt_train = gt_multi_tr_rw
            residual_train = residual_tr_rw

            fig1, axs = plt.subplots(2, 1, figsize=(45, 15 * 2))
            fig1.suptitle(key + "_1step ahead prediction", fontsize=16)
            axs[0].plot(pred_test[:, :], label="prediction")
            axs[0].plot(gt_test[:, :], label="groundtruth")
            axs[0].plot(residual_test[:, :], label="residual")
            axs[0].legend()
            axs[0].set_title('Test')

            axs[1].plot(pred_train[:, :], label="prediction")
            axs[1].plot(gt_train[:, :], label="groundtruth")
            axs[1].plot(residual_train[:, :], label="residual")
            axs[1].legend()
            axs[1].set_title('Train')


            plot_name = key + "_NST_1step_ahead_prediction_on_vanished_residual.png"
            directory = os.path.join(os.getcwd(), 'gt_pred_residual_plots', str(model_name), 'left_side')
            address = os.path.join(directory,plot_name)
            if not(os.path.exists(directory)):
                os.makedirs(directory)
            fig1.savefig(address)

            try:
                if wandb_run is not None:
                    keyy = str(key) + "_Qualitative-Results"
                    #print("wandb uploading Qualitative-result plot...")
                    image = plt.imread(address)
                    wandb_run.log({keyy: wandb.Image(image)})
            except Exception as e:
                print(e)
        #sys.exit(3)








        print("finished ... ")


        plt.close('all')
        wandb.finish()
        return result


def main():
    my_app()



## https://stackoverflow.com/questions/32761999/how-to-pass-an-entire-list-as-command-line-argument-in-python/32763023
if __name__ == '__main__':
    main()