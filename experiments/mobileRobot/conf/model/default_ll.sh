wandb:
  log: True
  project_name: 'baselines-mobile-diff-----'
  exp_name: 'mobile-acrkn'
  sweep: False
  sweep_id: null

learn:
  load: False
  gpu: '1'
  epochs: 1500
  batch_size: 350
  latent_vis: False
  plot_traj: True
  lr: 3e-4
  save_model: True
  loss: 'mse'

np:
  clip_gradients: True
  latent_obs_dim: 30
  agg_dim: 60


set_encoder:
  encoder_hidden_units: [240]
  aggregator: 'BA'
  time_embed: {'type':None, 'dim': 0}
  enc_out_norm: 'post'
  variance_act: 'softplus'

ssm_decoder:
  num_basis: 15
  bandwidth: 3
  enc_net_hidden_units: [ 120 ]
  dec_net_hidden_units: [ 240 ]
  trans_net_hidden_units: [ ]
  control_net_hidden_units: [ 120 ]
  task_net_hidden_units: [ 120 ]
  process_noise_hidden_units: [ 30 ]
  trans_net_hidden_activation: "Tanh"
  control_net_hidden_activation: 'ReLU'
  process_noise_hidden_activation: 'ReLU'
  task_net_hidden_activation: 'ReLU'
  learn_trans_covar: True
  decoder_conditioning: False
  intuitive_conditioning: False
  context_mu_flag_coeff: False
  context_mu_flag_control: False
  context_mu_flag_noise: False
  context_var_flag_coeff: False
  context_var_flag_control: False
  context_var_flag_noise: False
  additive_linear_task: False
  additive_linear_task_factorized: False
  additive_l_linear_task_factorized: True
  additive_nl_task: False
  nl_diagonal: True
  additive_ll_task: False
  hyper_transition_matrix: False
  multi_gaussian_l_transform: False
  trans_covar: 0.1
  learn_initial_state_covar: False
  initial_state_covar: 10
  enc_out_norm: 'post'
  clip_gradients: True
  never_invalid: False
  variance_act: 'softplus'
  num_heads: 1

data_reader:
  imp: 0.50
  terrain: 'sin2'
  frequency: '500'
  meta_batch_size: 3000
  batch_size: 150
  tar_type: 'delta'
  load: null
  save: 1
  dim: 111
  standardize: True
  split:
    - [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,41,42,43,44,45,46,47,48,49]
    - [30,31,32,33,34,35,36,37,38,39,40]
  shuffle_split: null
  file_name: 'AntWindows_123_0'
  trajPerTask: 10

submitit:
  folder: './experiments/mujoco_meta/Ant/slurm_output/'
  name: 'jobtrial'
  timeout_min: 10
  mem_gb: 10
  nodes: 1
  cpus_per_task: 3
  gpus_per_node: 1
  tasks_per_node: 1
  slurm_partition: gpu
