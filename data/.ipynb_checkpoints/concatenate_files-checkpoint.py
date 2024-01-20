import sys
import os
sys.path.append('.')
from typing import List, Any

import numpy as np
import json
import pandas as pd


class frankaArm():
    """
    The data relates to musculo-skeltal robot arm from Max Plank Tubingen. Here data appended with -3 at tail.
    """

    name = 'frankaarm'

    def __init__(self,standardize=True,imputation=0.0,targets='delta'):

        self.datapath = os.getcwd() + '/data/franka_attach_data/test_changing/'
        #self.datapath2 = os.getcwd() + '/data/franka_attach_data/single/test/'
        self.downsample = 1
        self.standardize = standardize
        self.tar_type = targets
        self.normalization = None
        self.percentage_imputation = imputation
        self._load_data(self.datapath)


    def _preprocessFranka(self,observation, action, time_diff, flag=False):
        if flag:
            obs = list();
            act = list();
            obs_valid = list()
            for idx, num_steps in enumerate(time_diff):
                obs.append(observation[idx]);
                act.append(action[idx]);
                obs_valid.append(True)

                for t in range(num_steps - 1):
                    obs.append(np.zeros_like(observation[idx]));
                    act.append(action[idx]);
                    obs_valid.append(False)
        else:
            obs = observation; act= action; obs_valid=observation.shape[0]*[True]
        return np.array(obs), np.array(act), np.array(obs_valid)

    def normalize(self, data, mean, std):
        dim = data.shape[-1]
        return (data - mean[:dim]) / (std[:dim] + 1e-10)

    def denormalize(self, data, mean, std):
        dim = data.shape[-1]
        return data * (std[:dim] + 1e-10) + mean[:dim]

    def compute_normalization(self,data):
        """
        Function to take in a dataset and compute the means, and stds.
        Return 6 elements: mean of s_t, std of s_t, mean of (s_t+1 - s_t), std of (s_t+1 - s_t), mean of actions, std of actions
        """
        mean_obs=np.mean(data['observations'],axis=0)
        std_obs=np.std(data['observations'],axis=0)
        mean_deltas=np.mean(data['delta'],axis=0)
        std_deltas=np.std(data['delta'],axis=0)
        mean_actions=np.mean(data['actions'],axis=0)
        std_actions=np.std(data['actions'],axis=0)
        return mean_obs, std_obs, mean_deltas, std_deltas, mean_actions, std_actions



    def _loop_data(self, path):
        path = path
        firstFlag = True
        i = 0
        list_dir = sorted([subdir.lower() for subdir in os.listdir(path)])
        for subdir in list_dir:

            subpath = path+subdir+'/'
            for f in os.listdir(subpath):
                print(subpath,i)
                data_in, data_out, data_valid = self._load_file(subpath,f)
                task_idx = i*np.ones((data_in.shape[0]))
                if firstFlag:
                    full_data_in = data_in
                    full_data_out = data_out
                    full_data_valid = data_valid
                    full_task_idx = task_idx
                    firstFlag = False
                else:
                    print(full_data_in.shape, data_in.shape)
                    full_data_in = np.concatenate((full_data_in, data_in))
                    full_data_out = np.concatenate((full_data_out, data_out))
                    full_data_valid = np.concatenate((full_data_valid, data_valid))
                    full_task_idx = np.concatenate((full_task_idx, task_idx))
            i = i+1

        data_in = full_data_in
        data_out = full_data_out
        data_valid = full_data_valid
        task_idx = full_task_idx

        H = data_in.shape[0]
        self.episode_length = 300  # changed

        data_ctx_in = np.array([data_in[ind:ind + self.episode_length, :] for ind in range(0, H, self.episode_length) if
                                (ind + self.episode_length < H - 1)])
        data_tar_in = data_ctx_in

        data_ep_in = np.concatenate((data_ctx_in[:-1, :, :], data_tar_in[1:, :, :]),axis=1)

        data_ctx_out = np.array(
            [data_out[ind:ind + self.episode_length, :] for ind in range(0, H, self.episode_length) if
             (ind + self.episode_length < H - 1)])
        data_tar_out = data_ctx_out

        data_ep_out = np.concatenate((data_ctx_out[:-1, :, :], data_tar_out[1:, :, :]), axis=1)

        data_valid_ctx = np.array(
            [data_valid[ind:ind + self.episode_length] for ind in range(0, H, self.episode_length) if
             (ind + self.episode_length < H - 1)])

        data_valid_tar = data_valid_ctx

        data_ep_valid = np.concatenate((data_valid_ctx[:-1, :], data_valid_tar[1:, :]), axis=1)

        data_task_ctx = np.array(
            [task_idx[ind:ind + self.episode_length] for ind in range(0, H, self.episode_length) if
             (ind + self.episode_length < H - 1)])

        data_task_tar = data_task_ctx

        data_ep_task = np.concatenate((data_task_ctx[:-1, :], data_task_tar[1:, :]), axis=1)

        return data_ep_in, data_ep_out, data_ep_valid, data_ep_task

    def _load_data(self,split):

        if split==False:
            data_in, data_out, data_valid, task_idx =  self._loop_data(self.datapath)
            data_in = np.array([data[::self.downsample, :] for data in data_in])
            data_out = np.array([data[::self.downsample, :] for data in data_out])
            data_valid = np.expand_dims(np.array([data[::self.downsample] for data in data_valid]),-1)

            #randomize
            dim = 14
            # arr = np.arange(data_in.shape[0])
            # np.random.seed(seed=122)
            # np.random.shuffle(arr)
            # data_in = data_in[arr, :, :dim]
            # data_out = data_out[arr, :,:dim]
            # data_valid = data_valid[arr,:,:dim]

            #train_test_split
            numData = data_in.shape[0]
            self.num_Train = int(0.7* numData)
            self.num_Test = numData - self.num_Train
            data_train_in = data_in[:self.num_Train,:,:]
            data_train_out = data_out[:self.num_Train,:,:]
            data_test_in = data_in[self.num_Train:,:,:]
            data_test_out = data_out[self.num_Train:,:,:]
            data_valid_train = data_valid[:self.num_Train]
            data_valid_test = data_valid[self.num_Train:]
            self.train_task_idx = task_idx[:self.num_Train]
            self.test_task_idx = task_idx[self.num_Train:]
        else:
            print(os.getcwd())
            data_train_in, data_train_out, data_valid_train, self.train_task_idx = self._loop_data(self.datapath)
            data_test_in, data_test_out, data_valid_test, self.test_task_idx = self._loop_data(self.datapath)
            data_train_in = np.array([data[::self.downsample, :] for data in data_train_in])[:,:,:7]
            data_train_out = np.array([data[::self.downsample, :] for data in data_train_out])
            data_valid_train = np.expand_dims(np.array([data[::self.downsample] for data in data_valid_train]), -1)

            data_test_in = np.array([data[::self.downsample, :] for data in data_test_in])[:,:,:7]
            data_test_out = np.array([data[::self.downsample, :] for data in data_test_out])
            data_valid_test = np.expand_dims(np.array([data[::self.downsample] for data in data_valid_test]), -1)

            self.num_Train = data_train_in.shape[0]
            self.num_Test = data_test_in.shape[0]
        #create (state,action,next_state)
        train_obs = data_train_in[:, 1:-1, :]; test_obs = data_test_in[:, 1:-1, :]
        train_prev_act = data_train_out[:, :-2, :]; test_prev_act = data_test_out[:, :-2, :]

        train_target_act = data_train_out[:, 1:-1, :]; test_target_act = data_test_out[:, 1:-1, :]
        train_targets = data_train_in[:, 2:, :];test_targets = data_test_in[:, 2:, :]
        train_valid = data_valid_train[:, 1:-1, :]; test_valid = data_valid_test[:, 1:-1, :]

        rs = np.random.RandomState(seed=42)

        train_obs_valid = rs.rand(train_targets.shape[0], train_targets.shape[1], 1) < 1 - self.percentage_imputation
        train_obs_valid[:, :5] = True
        train_obs_valid = np.logical_and(train_obs_valid,train_valid)
        print("Fraction of Valid Train Observations:",
              np.count_nonzero(train_obs_valid) / np.prod(train_obs_valid.shape))
        rs = np.random.RandomState(seed=23541)
        test_obs_valid = rs.rand(test_targets.shape[0], test_targets.shape[1], 1) < 1 - self.percentage_imputation
        test_obs_valid[:, :5] = True
        test_obs_valid = np.logical_and(test_obs_valid, test_valid)
        print("Fraction of Valid Test Observations:", np.count_nonzero(test_obs_valid) / np.prod(test_obs_valid.shape))


        # self.seqToArray
        self.seqToArray = lambda x: np.reshape(x, (x.shape[0] * x.shape[1], -1))


        self.train_targets = self.seqToArray(train_targets);
        self.train_obs = self.seqToArray(train_obs);
        self.train_prev_acts = self.seqToArray(train_prev_act);
        self.train_valid = self.seqToArray(train_valid)
        self.train_obs_valid = self.seqToArray(train_obs_valid);
        self.test_targets = self.seqToArray(test_targets);
        self.test_obs = self.seqToArray(test_obs);
        self.test_prev_acts = self.seqToArray(test_prev_act);
        self.train_target_acts = self.seqToArray(train_target_act)
        self.test_target_acts = self.seqToArray(test_target_act)
        self.test_valid = self.seqToArray(test_valid)
        self.test_obs_valid = self.seqToArray(test_obs_valid);

        # compute delta
        if self.tar_type == 'delta':
            self.train_targets = (self.train_targets - self.train_obs)
            self.test_targets = (self.test_targets - self.test_obs)

        self.train_act_diff = self.train_target_acts - self.train_prev_acts
        self.test_act_diff = self.test_target_acts - self.test_prev_acts

        # Standardize
        if self.standardize:
            print("Standardizingg.....................")
            dataTrain = dict()
            dataTrain = {'observations': self.train_obs, 'delta': self.train_targets, 'actions': self.train_target_acts}
            dataTest = {'observations': self.test_obs, 'delta': self.test_targets, 'actions': self.test_target_acts}

            mean_obs, std_obs, mean_deltas, std_deltas, mean_actions, std_actions = self.compute_normalization(
                dataTrain)

            # compute mean and std of delta separately

            train_diff = (self.train_targets - self.train_obs)[:,:7]
            mean_diff = np.mean(self.train_act_diff, axis=0)
            std_diff = np.std(self.train_act_diff, axis=0)


            self.normalization = dict()
            self.normalization['observations'] = [mean_obs, std_obs]
            self.normalization['actions'] = [mean_actions, std_actions]
            self.normalization['diff'] = [mean_deltas, std_deltas]
            self.normalization['act_diff'] = [mean_diff, std_diff]
            if self.tar_type == 'delta':
                self.normalization['targets'] = [mean_deltas, std_deltas]
            else:
                self.normalization['targets'] = [mean_obs, std_obs]


            self.train_obs = self.normalize(self.train_obs, self.normalization["observations"][0],
                                            self.normalization["observations"][1])
            self.train_acts = self.normalize(self.train_target_acts, self.normalization["actions"][0],
                                             self.normalization["actions"][1])
            self.train_targets = self.normalize(self.train_targets, self.normalization["targets"][0],
                                                self.normalization["targets"][1])
            self.test_obs = self.normalize(self.test_obs, self.normalization["observations"][0],
                                           self.normalization["observations"][1])
            self.test_acts = self.normalize(self.test_target_acts, self.normalization["actions"][0],
                                            self.normalization["actions"][1])
            self.test_targets = self.normalize(self.test_targets, self.normalization["targets"][0],
                                               self.normalization["targets"][1])
            self.train_act_diff = self.normalize(self.train_act_diff, self.normalization["act_diff"][0],
                                             self.normalization["act_diff"][1])
            self.test_act_diff = self.normalize(self.test_act_diff, self.normalization["act_diff"][0],
                                             self.normalization["act_diff"][1])

        return True



    def _load_file(self, path, f):
        # Load each file.
        with open(path + f) as json_file:
            data = json.load(json_file)
        data = pd.DataFrame(data=data['data'], columns=data['columns'])
        dfInput = data[
            ['joint_pos0', 'joint_pos1', 'joint_pos2', 'joint_pos3', 'joint_pos4', 'joint_pos5', 'joint_pos6',
             'joint_vel0', 'joint_vel1', 'joint_vel2', 'joint_vel3', 'joint_vel4', 'joint_vel5', 'joint_vel6',

        "current_load0",
        "current_load1", "current_load2", "current_load3", "current_load4", "current_load5", "current_load6"
        ]]

        dfAction = data[['command0', 'command1', 'command2', 'command3', 'command4', 'command5', 'command6']]
        #dfAction = data[['last_cmd0', 'last_cmd1', 'last_cmd2', 'last_cmd3', 'last_cmd4', 'last_cmd5', 'last_cmd6']]
        #dfAction = data[['uff0', 'uff1', 'uff2', 'uff3', 'uff4', 'uff5', 'uff6']]
        dfTime = data[["time_stamp"]].values
        diff = dfTime[1:] - dfTime[:-1]
        diff = np.array(diff)
        diff = np.insert(diff, 0, 0.001)
        diff = (np.round(diff, 3) / 0.001).astype(int)

        data_in,data_out,data_valid = self._preprocessFranka(dfInput.values,dfAction.values,diff,flag=False)

        #print(np.max(np.max(np.max(data_in))),np.min(np.min(np.min(data_in))))

        return np.array(data_in), np.array(data_out), np.array(data_valid)

if __name__ == "__main__":
    arm = frankaArm()
