import numpy as np
import matplotlib.pyplot as plt

# setting = 'demand_90_30_PatchTST_custom_ftM_sl90_ll15_pl30_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0'
# preds = np.load('./results/' + setting + '/pred.npy')
# trues = np.load('./results/' + setting + '/true.npy')
# plt.figure()
# plt.plot(trues[0, :, -1], label='GroundTruth')
# plt.plot(preds[0, :, -1], label='Prediction')
# plt.legend()
# plt.show()

import argparse
import random

import torch

from exp.exp_main import Exp_Main

if __name__ == '__main__':

    args = argparse.Namespace()

    # random seed
    args.random_seed = 2021

    # basic config
    args.is_training = 1
    args.model_id = 'demand_90_30'
    args.model = 'PatchTST'

    # data loader
    args.data = 'custom'
    args.root_path = '../dataset'
    args.data_path = 'demand.csv'
    args.features = 'M'
    args.target = 'OT'
    args.freq = 'd'
    args.checkpoints = './checkpoints/'

    # forecasting task
    args.seq_len = 90
    args.label_len = 15
    args.pred_len = 30

    # PatchTST
    args.fc_dropout = 0.2
    args.head_dropout = 0.0
    args.patch_len = 16
    args.stride = 8
    args.padding_patch = 'end'
    args.revin = 1
    args.affine = 0
    args.subtract_last = 0
    args.decomposition = 1
    args.kernel_size = 25
    args.individual = 0

    # Formers
    args.embed_type = 0
    args.enc_in = 21
    args.dec_in = 7
    args.c_out = 7
    args.d_model = 128
    args.n_heads = 16
    args.e_layers = 3
    args.d_layers = 1
    args.d_ff = 256
    args.moving_avg = 25
    args.factor = 1
    args.distil = True
    args.dropout = 0.2
    args.embed = 'timeF'
    args.activation = 'gelu'
    args.output_attention = False
    args.do_predict = False

    # optimization
    args.num_workers = 10
    args.itr = 1
    args.train_epochs = 100
    args.batch_size = 128
    args.patience = 20
    args.learning_rate = 0.0001
    args.des = 'Exp'
    args.loss = 'mse'
    args.lradj = 'type3'
    args.pct_start = 0.3
    args.use_amp = False

    # GPU
    args.use_gpu = True
    args.gpu = 0
    args.use_multi_gpu = False
    args.devices = '0,1,2,3'
    args.test_flop = False

    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des, ii)

    Exp = Exp_Main
    exp = Exp(args)

    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)

    torch.cuda.empty_cache()
