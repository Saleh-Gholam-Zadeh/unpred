import torch
import numpy as np
import scipy.stats
import pandas as pd
import math


def get_statistics(data):


    re_shape = lambda x: np.reshape(x, (x.shape[0] * x.shape[1], -1))
    data = re_shape(data);
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    # print("saleh_added: mean=",mean, "  std=",std)
    return mean, std



def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def squeeze_list(nested_list):
    '''
    :param nested_list: nested_list = [[[cpu1_1], [cpu1_2], ..., [cpu1_1400]], [[cpu2_1], [cpu2_2], ..., [cpu2_1400]], ..., [[cpu1145_1], [cpu1145_2], ..., [cpu1145_1400]]] len=1400 ,  cpui_j.shape =[24,1] ==> cpui : 1400 ta [24,1]
    :return:  abstrac_lsit: [[cpu_1], [cpu2], ..., [cpu1145]] #len =1145 , cpui: 3d arr [1400,24,1]
    all minutes are mixed but flows will be kept
    '''
    squeezed_list=[]
    abstract_list = []
    cnt=0

    for sublist in nested_list: # nested list contains all 1145 cpus ---> sublist contains 1cpu  a list with 1400 elements, each element is  24*1 array

        abstract_list.append([])

        if (len(sublist)>0): # if data for this cpu exists:
            abstract_list[-1].append( np.stack(sublist,axis=0) )
        else:
            #print("flow_",cnt,"is empty")
            abstract_list[-1] = []
        cnt=cnt+1
        #print("squeeze_list_cnt:",cnt,'/',len(nested_list))

    return  abstract_list

def ts2batch(data: np.ndarray , n_batch: int=10 ,len_hlaf_batch: int=16 , cent =None):

    if (len(np.squeeze(data).shape) >1):
        raise ValueError( 'Data should be 1D not' + str(len(np.squeeze(data).shape))  )



    data = np.squeeze(data)
    start = len_hlaf_batch  # 16
    end=  len(data) - len_hlaf_batch #17

    if cent is None:
        centers= np.random.randint(start,end,n_batch)  # (16,17)
    else:
        centers = cent

    trajs = np.zeros((n_batch,2*len_hlaf_batch))

    # print("centers:",centers)
    for i , center in enumerate(centers):
        trajs[i,:]=data[center-len_hlaf_batch:center+len_hlaf_batch]


    return trajs , centers


def ts2batch_ctx_tar(data: np.ndarray , n_batch: int=10 ,len_ctx: int=49,len_tar: int=1 , cent =None):

    if (len(np.squeeze(data).shape) >1):
        raise ValueError( 'Data should be 1D not' + str(len(np.squeeze(data).shape))  )



    data = np.squeeze(data)
    start = len_ctx  # 16
    end=  len(data) - len_tar #17

    if cent is None:
        centers= np.random.randint(start,end,n_batch)  # (16,17)
    else:
        centers = cent
    unique_centers = np.unique(centers)
    new_n_batch = len(unique_centers)
    trajs = np.zeros((new_n_batch,len_ctx+len_tar))

    # print("centers:",centers)
    for i , cent in enumerate(unique_centers):
        trajs[i,:]=data[cent-len_ctx:cent+len_tar]


    return trajs , unique_centers


def circular_shift_features(data, t):
    # m = data.shape[0]  # number of features
    # n = data.shape[1]  # number of data points

    # Perform circular shift on the features
    shifted_data = np.roll(data, t, axis=0)

    return shifted_data

def permute_1d_array(arr):
    assert len(arr.shape) < 3
    arr = np.array(arr).flatten()  # Ensure array is flattened to 1D
    permuted_arr = np.random.permutation(arr)
    return permuted_arr

def count_elements_greater_than_MI(MI , B):
    """
    MI = current MI
    B = list of permutations
    """
    return sum(1 for element in B if element > MI)


def get_mutual_information(data, number_output_functions=1, min_n_datapoints_a_bin = None, perm_test_flag=True, N=10):
    # Calulate the Shannon mutual information
    def make_summand_from_frequencies(x, y):
        if x == 0:
            return 0
        else:
            return x * math.log2(x / y) / math.log2(basis_log_mutual_information)

    data = circular_shift_features(data, number_output_functions)

    # Insert the the indices of the rows where the components of the output functions are stored
    # for i in range(0, number_output_functions):
    #     list_variables.insert(i, i)
    sum_list = []
    m = data.shape[0]
    n = data.shape[1]
    l = [0 for i in range(0, m)]
    freq_data = [0 for i in range(0, m)]
    left_features = [i for i in range(0, m)]
    list_variables = [i for i in range(0, m)]
    data = data[list_variables, :]

    if (min_n_datapoints_a_bin is None):
        min_n_datapoints_a_bin = 0.05*n


    constant_features = []

    for i in range(0, m):
        mindata = min(data[i, :])
        maxdata = max(data[i, :])
        if maxdata <= mindata:
            print("Feature #"f"{list_variables[i]}" " has only constant values")
            left_features.remove(i)
            constant_features.append(list_variables[i])
        else:
            # start the binning by sorting the data points
            list_points_of_support = []
            datapoints = data[i, :].copy()
            datapoints.sort()
            last_index = 0
            # go through the data points and bin them
            for point in range(0, datapoints.size):
                if point >= (datapoints.size - 1):  # if end of the data points leave the for-loop
                    break
                # close a bin if there are at least min_n_datapoints_a_bin and the next value is bigger
                if datapoints[last_index:point + 1].size >= min_n_datapoints_a_bin and datapoints[point] < datapoints[
                    point + 1]:
                    list_points_of_support.append(datapoints[point + 1])
                    last_index = point + 1
            if len(list_points_of_support) > 0:  # test that there is at least one point of support (it can be if there are only constant value up to the first ones which are less than min_n_datapoints_a_bin
                if list_points_of_support[0] > datapoints[
                    0]:  # add the first value as a point of support if it does not exist (less than min_n_datapoints_a_bin at the beginning)
                    list_points_of_support.insert(0, datapoints[0])
            else:
                list_points_of_support.append(datapoints[0])
            list_points_of_support.append(datapoints[
                                              -1] + 0.1)  # Add last point of support such that last data point is included (half open interals in Python!)
            if datapoints[datapoints >= list_points_of_support[
                -2]].size < min_n_datapoints_a_bin:  # if last bin has not at least min_n_datapoints_a_bin fuse it with the one before the last bin
                if len(list_points_of_support) > 2:  # Test if there are at least 3 points of support (only two can happen if there only constant values at the beginning and only less than n_min_datapoints_a_bin in the end)
                    list_points_of_support.pop(-2)
            l[i] = list_points_of_support
            freq_data[i] = np.histogram(data[i, :], bins=l[i])[0]

    # Check for constant features
    if constant_features != []:
        print("List of features with constant values:")
        print(constant_features)
    for id_output in range(0, number_output_functions):
        if id_output in constant_features or len(
                freq_data[id_output]) < 2:  # Warn if the output function is constant e.g. due to an unsuitable binning
            print("Warning: Output function " + str(id_output) + " is constant!")


    # Calculate the mutual information for each feature with the corresponding component of the output function
    list_of_data_frames = []
    mutual_info = np.ones((1,len(left_features) - number_output_functions + 1))  # number of featuers plus one component of the output-function


# N times for loop here to shuffle N times and store the sum_MI
    if perm_test_flag:
        N=N+1

    total_MI_for_each_permutation = []
    for cnt in range(N):
        print("cnt: ",cnt,'/',N)
        sum_list =[]
        for i in range(0, number_output_functions):
            basis_log_mutual_information = len(freq_data[i])
            # shuffle (data [i,:] ) ---> data[i,:]

            if perm_test_flag and cnt>0:
                perm_labels = permute_1d_array(data [i,:].copy())
                data[i,:] = perm_labels

            list_of_features = list(range(number_output_functions, len(left_features)))
            list_of_features.insert(0, i)
            id_features = np.array(list_variables)[list_of_features]

            for j in list_of_features:
                freq_data_product = ((
                np.histogram2d(data[i, :], data[left_features[j], :], bins=(l[i], l[left_features[j]]))[0])) / n
                expfreq = (np.outer(freq_data[i], freq_data[left_features[j]])) / (n * n)
                if j < number_output_functions:
                    mutual_info[0, 0] = np.sum(np.array(list(
                        map(make_summand_from_frequencies, freq_data_product.flatten().tolist(),
                            expfreq.flatten().tolist()))))
                else:
                    mutual_info[0, j - number_output_functions + 1] = np.sum(np.array(list(
                        map(make_summand_from_frequencies, freq_data_product.flatten().tolist(),
                            expfreq.flatten().tolist()))))

            sum_mi = np.sum(mutual_info[0,1:]) # the sum over all features for each output
            sum_list.append(sum_mi)
            pd_mutual_information = pd.DataFrame({"index feature": id_features, "mutual information": mutual_info.tolist()[0]})
            pd_mutual_information['index feature'] = pd_mutual_information['index feature'].astype(int)

            list_of_data_frames.append(pd_mutual_information)

        if cnt==0:
            actual_total_MI = sum(sum_list) # sum over all outputs (previously done on the features)
            actual_list_of_df = list_of_data_frames  # we can return this instead of None
        else: # permutation test ---> values come to the list to make a distribution:
            total_MI_for_each_permutation.append(sum(sum_list))


    if perm_test_flag==False:
        return list_of_data_frames, actual_total_MI,None   ,     None       ,         None
    else:
        avg_MI_permute = sum(total_MI_for_each_permutation) / len(total_MI_for_each_permutation)
        pvalue = np.sum(np.array(total_MI_for_each_permutation) > actual_total_MI)/len(total_MI_for_each_permutation)
        return None,                actual_total_MI, pvalue, avg_MI_permute  ,total_MI_for_each_permutation


def run_test(data: np.ndarray, min_n_datapoints_a_bin = None, number_output_functions: int=1 , alpha=0.01 ,log: bool=True ,bonfer: bool= True):

    '''
        Implementation of MI code
        data: np.array m_row ---> ctx (features) + target : so each sample or datapoints is stored in a column
    /// n_col---> samples : each sample stored in one column
        i.e
        m = data.shape[0]  # number features (plus targets)
        n = data.shape[1]  # number of data points
    the whole length of target + context = m ---> target comes first
     each column of the data: [t1, t2, ..., t_num_output_function, ctx1, ctx2, ... m ]

     outputs:
     cnt_dep = calculate how many of targets are correlated with how many of context (max = ctx*tar)
     pval_list= list of all pvalues: [ctx1-tar1 ctx1-tar2 ... ctx1-tarm ctx2-tar1 ctx2-tar2 ... ctxk-tarm]

    '''

    data = circular_shift_features(data, number_output_functions)


    counter_bins_less_than5_relevant_principal_features=0 # number of chi-square tests with less than 5 datapoints a bin
    counter_bins_less_than1_relevant_principal_features=0 # number of chi-square tests with less than 1 datapoint a bin
    counter_number_chi_square_tests_relevant_principal_features=0 # nu



    #data = data.to_numpy()
    m = data.shape[0]  # number features
    n = data.shape[1]  # number of data points
    if (min_n_datapoints_a_bin is None):
        min_n_datapoints_a_bin = 0.05*n

    #alpha=0.01/m
    if bonfer:
        #print("old_alpha:",alpha)
        #((m-number_output_functions)*number_output_functions) --> number of experiments based on which we return a number or make a decision
        alpha = alpha/((m-number_output_functions)*number_output_functions)
        #print("bonfer_alpha after correction:",alpha)

    l = [0 for i in range(0, m)]  # list of lists with the points of support for the binning
    freq_data = [0 for i in range(0, m)]  # list of histograms
    left_features = [i for i in range(0, m)]  # list of features that is step by step reduced to the relevant ones
    constant_features = []

    # remove constant features and binning (discretizing the continuous values of our features)
    for i in range(0, m):
        mindata = min(data[i, :])
        maxdata = max(data[i, :])
        if maxdata <= mindata:
            print("Feature #"f"{i}" " has only constant values")
            left_features.remove(i)
            constant_features.append(i)
            raise ValueError('WTF') #added by saleh
        else:
            # start the binning by sorting the data points
            list_points_of_support = []
            datapoints = data[i, :].copy()
            datapoints.sort()
            last_index = 0
            # go through the data points and bin them
            for point in range(0, datapoints.size):
                if point >= (datapoints.size - 1):  # if end of the data points leave the for-loop
                    break
                # close a bin if there are at least min_n_datapoints_a_bin and the next value is bigger
                if datapoints[last_index:point + 1].size >= min_n_datapoints_a_bin and datapoints[point] < datapoints[
                    point + 1]:
                    list_points_of_support.append(datapoints[point + 1])
                    last_index = point + 1
            if len(list_points_of_support) > 0:  # test that there is at least one point of support (it can be if there are only constant value up to the first ones which are less than min_n_datapoints_a_bin
                if list_points_of_support[0] > datapoints[
                    0]:  # add the first value as a point of support if it does not exist (less than min_n_datapoints_a_bin at the beginning)
                    list_points_of_support.insert(0, datapoints[0])
            else:
                list_points_of_support.append(datapoints[0])
            list_points_of_support.append(datapoints[
                                              -1] + 0.1)  # Add last point of support such that last data point is included (half open interals in Python!)
            if datapoints[datapoints >= list_points_of_support[
                -2]].size < min_n_datapoints_a_bin:  # if last bin has not at least min_n_datapoints_a_bin fuse it with the one before the last bin
                if len(list_points_of_support) > 2:  # Test if there are at least 3 points of support (only two can happen if there only constant values at the beginning and only less than n_min_datapoints_a_bin in the end)
                    list_points_of_support.pop(-2)
            l[i] = list_points_of_support
            freq_data[i] = np.histogram(data[i, :], bins=l[i])[0]
    #print("Binning done!")
    #print("List of features with constant values:")
    #print(constant_features)

    for id_output in range(0, number_output_functions):
        if id_output in constant_features or len(freq_data[id_output]) < 2:  # Warn if the output function is constant e.g. due to an unsuitable binning
            print("Warning: System state " + str(id_output) + " is constant!")
            raise ValueError('WTF') #added by saleh

    intermediate_list_depending_on_system_state=[]
    intermediate_list_not_depending_on_system_state=[]
    pval_list = []
    indices_principal_feature_values=np.zeros((1, 2))
    cnt_dep = 0

    for j in range(number_output_functions,m):
        if len(freq_data[j]) > 1:
            dependent = 0 # Flag for the input feature j if there is a relation to one output-function
            for id_output in range(0,number_output_functions):
                counter_number_chi_square_tests_relevant_principal_features +=1
                freq_data_product = np.histogram2d(data[id_output, :], data[j, :],
                                            bins=(l[id_output], l[j]))[0]
                expfreq = np.outer(freq_data[id_output], freq_data[j]) / n
                if sum(expfreq.flatten() < 5) > 0:
                    counter_bins_less_than5_relevant_principal_features += 1
                if sum(expfreq.flatten() < 1) > 0:
                    counter_bins_less_than1_relevant_principal_features += 1
                pv = scipy.stats.chisquare(freq_data_product.flatten(), expfreq.flatten(),ddof=(freq_data_product.shape[0]-1)+(freq_data_product.shape[1]-1))[1]
                pval_list.append(pv)
                # According to the documentation of scipy.stats.chisquare, the degrees of freedom is k-1 - ddof where ddof=0 by default and k=freq_data_product.shape[0]*freq_data_product.shape[0].
                # According to literatur, the chi square test statistic for a test of independence (r x m contingency table) is approximately chi square distributed (under some assumptions) with degrees of freedom equal
                # freq_data_product.shape[0]-1)*(freq_data_product.shape[1]-1) = freq_data_product.shape[0]*freq_data_product.shape[1] - freq_data_product.shape[0] - freq_data_product.shape[1] + 1.
                # Consequently, ddof is set equal freq_data_product.shape[0]-1+freq_data_product.shape[1]-1 to adjust the degrees of freedom accordingly.

                # if p-value pv is less than alpha the hypothesis that j is independent of the output function is rejected
                if pv <= alpha:
                    dependent=1 # if the current feature is related to any of the outputs then it would become 1
                    cnt_dep += 1 # it counts the current feature is related to how many of the outputs. it is integer between 0 to num_output
                    #break
            if dependent==1:
                intermediate_list_depending_on_system_state.append(j)
            else:
                intermediate_list_not_depending_on_system_state.append(j)
        else:
            intermediate_list_not_depending_on_system_state.append(j)
            pv=1.1
        #indices_principal_feature_values= np.concatenate((indices_principal_feature_values, np.array([j, pv]).reshape((1, 2))), axis=0)
        indices_principal_feature_values = np.concatenate((indices_principal_feature_values, np.array([j, pv]).reshape((1, 2))), axis=0)


    return intermediate_list_depending_on_system_state,pval_list,cnt_dep


def linear_correl(arr ,tresh: float = 0.01, number_output_functions: int=1,bonfer = True):


    '''
    implementation of PearsonR correlation
    m = data.shape[0]  # number of features (or more precisely features + outputs )
    n = data.shape[1]  # number of data points (windows)
    return p-value and r va
    '''
    assert isinstance(tresh, float), "Variable is not of type float"
    assert len(arr.shape)==2
    m,n = arr.shape
    if bonfer :
        tresh = tresh/(number_output_functions*(m-number_output_functions))

    X_mat = arr[:-number_output_functions,:] # features
    Y = arr[-number_output_functions:,:] # labels
    res = []
    sum_r = 0
    print(tresh)
    for i in range(number_output_functions): # for loop on y
        for j in range(m-number_output_functions): # for loop on x --> all corelations between xi and yj will be considered
            r , pv = (scipy.stats.pearsonr(X_mat[j,:],Y[i,:]))
            res.append((pv,r))
            #print(pv,tresh)
            if pv <tresh:
                #print(i," pv:",pv,"  r:", r)
                sum_r = sum_r + np.abs(r)
    return sum_r





def split_k_m(sequence,k,burn_in=5):
    '''
    TODO: K and M as argument instead of split by half
    '''
    #print("saleh added line 9 dataProcess.py  k=",k , 'burn_in=',burn_in)
    if k==0:
        context_seq, target_seq = None, sequence
    else:

        context_seq, target_seq = sequence[:, :k, :], sequence[:, k-burn_in:, :] #with the default setting  k= data_reader_batch_size e.g 75  --> context has the len of 75 and target has the len of 5 .  burn_in=0
    #print("context_seq.shape,target_seq.shape,k,burn_in ",context_seq.shape,target_seq.shape,k,burn_in  )
    return context_seq, target_seq

def get_ctx_target_impute(obs, act, target, k, num_context=None, test_gt_known=True, tar_imp=0.0, ctx_burn_in=0, tar_burn_in=5, random_seed=True):
    '''
    :param obs: observations
    :param act: actions
    :param target: targets (might be difference(delta) )
    :param k: how many timesteps to have for context sequence after splitting the sequence
    :param num_context: if we want a context size less than k, in paper None by default
    :param test_gt_known: is test ground truth available
    :param tar_imp: percentage imputation in target sequence
    :param ctx_burn_in: how much context to burnin
    :param tar_burn_in: how much of target should be used as burnin
    :param random_seed:
    :return:
    '''
    if random_seed:
        seed = np.random.randint(1, 1000)
    else:
        seed = 42

    seed=42 #saleh_added

    if ctx_burn_in is None:
        ctx_burn_in = k

    rs = np.random.RandomState(seed=seed)
    ctx_obs, tar_obs = split_k_m(obs, k, burn_in=ctx_burn_in)
    ctx_act, tar_act = split_k_m(act, k, burn_in=ctx_burn_in)
    if test_gt_known:
        ctx_tar, tar_tar = split_k_m(target, k, burn_in=ctx_burn_in)
    else:
        ctx_tar, tar_tar = target, None
    # Sample locations of context points
    if num_context is not None:
        num_context = num_context
        locations = np.random.choice(k,
                                     size=num_context,
                                     replace=False)
        #TODO: change this randomization per episode too
        ctx_obs = ctx_obs[:, locations[:k], :]
        ctx_act = ctx_act[:, locations[:k], :]
        ctx_tar = ctx_tar[:, locations[:k], :]

    if tar_imp is not None:
        tar_obs_valid = rs.rand(tar_obs.shape[0], tar_obs.shape[1], 1) < 1 - tar_imp
        tar_obs_valid[:, :tar_burn_in] = True
    else:
        tar_obs_valid = rs.rand(tar_obs.shape[0], tar_obs.shape[1], 1) < 0
        tar_obs_valid[:, :tar_burn_in] = True
    return ctx_obs, ctx_act, ctx_tar, tar_obs, tar_act, tar_tar, tar_obs_valid


def get_ctx_target_multistep(obs, act, target, k, num_context=None, test_gt_known=True, ctx_burn_in=0, tar_burn_in=5, random_seed=True):
    '''
    :param obs: observations
    :param act: actions
    :param target: targets
    :param k: how many timesteps to have for context sequence after splitting the sequence
    :param num_context: if we want a context size less than k, in paper None by default
    :param test_gt_known: is test ground truth available
    :param tar_imp: percentage imputation in target sequence
    :param ctx_burn_in: how much context to burnin
    :param tar_burn_in: how much of target should be used as burnin
    :param random_seed:
    :return:
    '''
    if random_seed:
        seed = np.random.randint(1, 1000)
    else:
        seed = 0

    seed = 0 #saleh_added
    if ctx_burn_in is None:
        ctx_burn_in = k

    rs = np.random.RandomState(seed=seed)

    ctx_obs, tar_obs = split_k_m(obs, k, burn_in=ctx_burn_in) #ctx_burn_in=0 & tar_burn_in=5 &   #obs.shape =[700,81,25] (generated in sliding window fashion and then stacked)
    ctx_act, tar_act = split_k_m(act, k, burn_in=ctx_burn_in)
    #ctx_obs.shape [700,75,25] , tar_obs.shape = [700,6,25]
    if test_gt_known: #True at least when predict_mbrl() is called
        ctx_tar, tar_tar = split_k_m(target, k, burn_in=ctx_burn_in) #[710,75,25] , [710,5,25]
    else:
        ctx_tar = target
        tar_tar = None
    # Sample locations of context points
    if num_context is not None: #False, inja nemiad
        num_context = num_context
        locations = np.random.choice(k,size=num_context,replace=False)
        #TODO: change this randomization per episode too and add test_gt_stuff
        ctx_obs = ctx_obs[:, locations[:k], :]
        ctx_act = ctx_act[:, locations[:k], :]
        ctx_tar = ctx_tar[:, locations[:k], :]

    #                      rs.rand(700,6,1) ---> choose from uniform distribution [0,1]
    tar_obs_valid = rs.rand(tar_obs.shape[0], tar_obs.shape[1], 1) < 0 #Everything False
    tar_obs_valid[:, :tar_burn_in] = True   #the first 5 become observable


    return ctx_obs,      ctx_act,      ctx_tar,     tar_obs,     tar_act,       tar_tar,       tar_obs_valid
#          [700,75,25]   [700,75,25]   [700,75,1]   [700,5,25]   [700,5,25]   [700,5,1]    [700,5,1]

def get_sliding_context_batch_mbrl(obs, act, target, k, steps=1, tar_burn_in=5):
    '''
    Given say N episodes, it creates context/target windows in the ration k:(step+tar_length).
    The window centers are ordered as opposed to random.
    :param obs:                                                                  e.g [10 ,150(2*contex_12size) ,25(features)]
    :param act:                                                                  e.g [10 ,150(2*contex_size) ,25(features)]
    :param target:                                                               the last feature in the  obs(t-1) is equal to target
    :param k: context size
    :param steps: multi step ahead prediction to make
    :return:
    '''
    #tar_length = steps + tar_burn_in   # Multisteps+5    #bayad -1 beshe
    tar_length = steps + tar_burn_in -1  # for both case target =  'delta' and target ='observation'
    H = obs.shape[1] # 2*contex_size
    #Creating ordered window centres
    window_centres = np.arange(k,H-tar_length+1) #  1d array[k,k+1,...,2k-multistep-5+1 (e.g 144 for step=1)]
    #using above centers, Creates windows contex_target windows in sliding window fashion
    obs_hyper_batch = [obs[:, ind - k:ind + tar_length, :] for ind in window_centres]
    act_hyper_batch = [act[:, ind - k:ind + tar_length, :] for ind in window_centres] #a list of 70(k-burn_in) elements. each elements' shape=[10,k+step+burn_in(e.g 81),25(#feature)]
    target_hyper_batch = [target[:, ind - k:ind + tar_length, :] for ind in window_centres] #a list of 70(k-burn_in) elements. each elements' shape=[10,k+step+burn_in(e.g 81),1(#target_dim)]

    return torch.cat(obs_hyper_batch,dim=0), torch.cat(act_hyper_batch,dim=0), torch.cat(target_hyper_batch, dim=0) # returned target shape=[10*len(list)e.g 700, k+step+burn_in(e.g 81),1(#target_dim)]

def squeeze_sw_batch(pred_mean_hyper_batch, pred_var_hyper_batch, target_hyper_batch, num_episodes):
                    #[700,6,1]            , [700,6,1]            , [700,6,1]         ,[10]
    '''
    :param pred_hyper_batch:
    :param target_hyper_batch:
    :return: predicted and ground truth sequence and has number of episodes = num_episodes
    '''
    if type(pred_mean_hyper_batch) is np.ndarray:
        pred_mean_hyper_batch = torch.from_numpy(pred_mean_hyper_batch).float()
    # if type(pred_var_hyper_batch) is np.ndarray:
    #     pred_var_hyper_batch = torch.from_numpy(pred_var_hyper_batch).float()
    if type(target_hyper_batch) is np.ndarray:
        target_hyper_batch = torch.from_numpy(target_hyper_batch).float()
    hyper_episodes = pred_mean_hyper_batch.shape[0] #[710]
    hyper_windows_per_episode = int(hyper_episodes/num_episodes) #71
    assert num_episodes*hyper_windows_per_episode == hyper_episodes
    for ind in range(hyper_windows_per_episode): #(0to70)
        if ind==0: # it get the first 10 episodes
            squeezed_pred_mean = pred_mean_hyper_batch[ind*num_episodes:(ind+1)*num_episodes,:,:]  #first time we consider burin_in as well
            #squeezed_pred_var = pred_var_hyper_batch[ind * num_episodes:(ind + 1) * num_episodes, :, :]
            squeezed_gt = target_hyper_batch[ind*num_episodes:(ind+1)*num_episodes,:,:]
        else:
            squeezed_pred_mean = torch.cat((squeezed_pred_mean, pred_mean_hyper_batch[ind * num_episodes:(ind + 1) * num_episodes, -1:, :]),dim=1)
            #squeezed_pred_var = torch.cat((squeezed_pred_var,pred_var_hyper_batch[ind * num_episodes:(ind + 1) * num_episodes, -1:, :]),dim=1)

            squeezed_gt = torch.cat((squeezed_gt,torch.unsqueeze(target_hyper_batch[ind*num_episodes:(ind+1)*num_episodes,-1,:], dim=1)), dim=1)
    return squeezed_pred_mean, np.nan, squeezed_gt





seqToArray = lambda x: np.reshape(x, (x.shape[0] * x.shape[1], -1))
def arrayToSeq(x, numEp, epLen):
    if type(x) is not np.ndarray:
        x = x.cpu().detach().numpy()
    return np.reshape(x, (numEp, epLen, -1))

def normalize(data, mean, std):
    dim = data.shape[-1]
    return (data - mean[:dim]) / (std[:dim] + 1e-10)


def denormalize(data, mean, std):
    #dim = data.shape[-1]
    return data * (std + 1e-10) + mean

def norm(x,normalizer,tar_type='targets'):
    if type(x) is not np.ndarray:
        x = x.cpu().detach().numpy()
    if tar_type=='observations':
        return normalize(x, normalizer["observations"][0][:x.shape[-1]],normalizer["observations"][1][:x.shape[-1]])

    if tar_type=='label':
        #print("++++++++++Denorm with label++++++++++++++++")
        return normalize(x, normalizer["label"][0][:x.shape[-1]],normalizer["label"][1][:x.shape[-1]])


    if tar_type == 'label_diff':
        return normalize(x, normalizer["label_diff"][0][:x.shape[-1]],normalizer["label_diff"][1][:x.shape[-1]])



    if tar_type == 'actions':
        return normalize(x, normalizer["actions"][0][:x.shape[-1]],normalizer["actions"][1][:x.shape[-1]])

    if tar_type == 'targets':
        return normalize(x, normalizer["targets"][0][:x.shape[-1]],normalizer["targets"][1][:x.shape[-1]])

    else:
        raise ValueError(' norm key is not specified.')


#def denorm(x, data, tar_type='targets'):
def denorm(x, normalizer, tar_type='targets'):
    #print("tar_type = ",tar_type)
    if type(x) is not np.ndarray:
        x = x.cpu().detach().numpy()

    if tar_type=='observations':
        raise ValueError(' Warning: this shouldnt be used')
        return denormalize(x, normalizer["observations"][0][:x.shape[-1]],normalizer["observations"][1][:x.shape[-1]])

    if tar_type=='packets':
        #print("++++++++++Denorm with label++++++++++++++++")
        return denormalize(x, normalizer['mu'],normalizer["std"])


    if tar_type == 'label_diff':
        #print("!!!!!'Denorm with label_diff'!!!!!!!!")
        return denormalize(x, normalizer["label_diff"][0][:x.shape[-1]],normalizer["label_diff"][1][:x.shape[-1]])

    if tar_type == 'targets': #target is either label or label_diff
        #print("!!!!!Denorm with tareget !!!!!!!!")
        return denormalize(x, normalizer["targets"][0][:x.shape[-1]],normalizer["targets"][1][:x.shape[-1]])


    if tar_type == 'actions':
        return denormalize(x, normalizer["actions"][0][:x.shape[-1]],normalizer["actions"][1][:x.shape[-1]])

    if tar_type == 'act_diff':
        return denormalize(x, normalizer["act_diff"][0][:x.shape[-1]],normalizer["act_diff"][1][:x.shape[-1]])


    else:
        raise ValueError(' Denorm key is not specified.')


def diffToState(diff,current,normalizer,standardize=True ,gt_flag=None):
    '''
    :param diff: difference between next and current state
    :param current: current state
    :param data: data object
    :return: normalized next state
    '''
    assert (diff.shape[-1]==current.shape[-1]), "  mismatch in number of features " + str (diff.shape[-1]) +" != " + str(current.shape[-1])
    if type(diff) is not np.ndarray:
        diff = diff.cpu().detach().numpy()
    if type(current) is not np.ndarray:
        current = current.cpu().detach().numpy()

    if standardize:
        current = denorm(current, normalizer, 'label')
        diff = denorm(diff, normalizer, "label_diff")

        next = norm(current + diff, normalizer, "label")
    else:
        next = current + diff
        if gt_flag is not None:
            assert (all((next >= 0))) , "  converted packet numbers should be positive but " + str(next)

    return next,diff

def diffToStateMultiStep(diff, current, valid_flag, normalizer, standardize=True):
    '''
    :param diff: difference between next and current state
    :param current: current state
    :param data: data object
    :return: normalized next state, diff
    '''
    assert (diff.shape[-1] == current.shape[-1]), "  mismatch in number of features " + str(diff.shape[-1]) + " != " + str(current.shape[-1])
    if type(diff) is not np.ndarray:
        diff = diff.cpu().detach().numpy()
    if type(current) is not np.ndarray:
        current = current.cpu().detach().numpy()



    if standardize:
        current = denorm(current, normalizer, 'label')
        diff = denorm(diff, normalizer, "label_diff")

    next_state = np.zeros(current.shape)

    for t in range(current.shape[1]):
        '''
        Loop over the differences and check is valid flag is true or false
        '''
        if valid_flag[0,t,0] == False and t>0:
            next_state[:,t] = next_state[:,t-1] + diff[:,t]
        else:
            next_state[:,t] = current[:,t] + diff[:,t]

    next_norm =  norm(next_state, normalizer, 'label')
    diff_norm =  norm(diff, normalizer, 'label_diff')
    return next_norm, diff_norm


def diffToStateImpute(diff, current, valid_flag, normalizer, standardize=True):
    '''
    :param diff: difference between next and current state
    :param current: current state
    :param data: data object
    :return: normalized next state, diff
    '''

    assert (diff.shape[-1] == current.shape[-1]), "  mismatch in number of features " + str(diff.shape[-1]) + " != " + str(current.shape[-1])
    if type(diff) is not np.ndarray:
        diff = diff.cpu().detach().numpy()
    if type(current) is not np.ndarray:
        current = current.cpu().detach().numpy()


    if standardize:
        current = denorm(current, normalizer, 'label')
        diff = denorm(diff, normalizer, "label_diff")
    next_state = np.zeros(current.shape)

    for idx in range(current.shape[0]):
        for t in range(current.shape[1]):
            '''
            Loop over the differences and check is valid flag is true or false
            '''
            if valid_flag[0, t, 0] == False and t > 0:
                next_state[:, t] = next_state[:, t - 1] + diff[:, t]
            else:
                next_state[idx, t] = current[idx, t] + diff[idx, t]

    next_norm = norm(next_state, normalizer, 'label')
    diff_norm = norm(diff, normalizer, 'label_diff')

    #assert (all((next_norm >= 0))), "  converted packet numbers should be positive but " + str(next)   --->it is prediction which can be negative if not trained properly
    return next_norm, diff_norm

def diffToAct(diff,prev,data,standardize=True):
    '''
    :param diff: difference between next and current state
    :param current: current state
    :param data: data object
    :return: normalized next state
    '''
    print("diffToAct was called!!")
    if type(diff) is not np.ndarray:
        diff = diff.cpu().detach().numpy()
    if type(prev) is not np.ndarray:
        prev = prev.cpu().detach().numpy()

    if standardize:
        prev = denorm(prev, data, 'actions')
        diff = denorm(diff, data, "act_diff")

        current = norm(prev + diff, data, "actions")

    return current,diff



