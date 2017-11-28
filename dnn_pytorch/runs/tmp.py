import os
import subprocess

gpu_device = '0'

# train = '/nas/data/m1/zhangb8/lorelei/data/reference/eval17/il5/0808/reflex_train.bio'
train = '/nas/data/m1/zhangb8/lorelei/data/reference/eval17/il5/0808/ann_tool.rm_sents.train+Tigrinya.clean.train.bio'
dev = '/nas/data/m1/zhangb8/lorelei/data/reference/ldc/tir/unsequestered/unsequestered.bio'
test = '/nas/data/m1/zhangb8/lorelei/data/reference/ldc/tir/unsequestered/unsequestered.bio'

model_dir = '/nas/data/m1/zhangb8/ml/model/pytorch/tmp'

# pre_emb = ''
pre_emb = '/nas/data/m1/zhangb8/lorelei/data/word_embedding/tir/set0+set1+setE+voa+reflex.emb'

# features
cluster = ''
# cluster = '/nas/data/m1/zhangb8/lorelei/data/word_clusters/tir/set0+set1+setE+voa+reflex/paths'

# run command
script = '../../dnn_pytorch/seq_labeling/train.py'
cmd = [
    '/nas/data/m1/zhangb8/tools/anaconda3/bin/python3',
    # 'python3',
    script,
    # data settings
    '--train', train,
    '--dev', dev,
    '--test', test,
    '--model_dp', model_dir,
    '--tag_scheme', 'iobes',
    # '--tag_scheme', 'classification',

    # parameter settings
    '--lower', '0',
    '--char_dim', '25',
    '--char_lstm_dim', '25',
    '--char_conv_channel', '25',
    '--word_dim', '100',
    '--word_lstm_dim', '100',
    '--pre_emb', pre_emb,
    '--all_emb', '0',
    '--cap_dim', '0',
    '--crf', '1',
    '--dropout', '0.5',
    '--lr_method', 'sgd-init_lr=.005-lr_decay_epoch=50',
    '--batch_size', '20',

    # external feature settings
    '--feat_dim', '0',
    '--upenn_stem', '',
    '--pos_model', '',
    '--cluster', cluster,
    '--ying_stem', ''
]

os.environ.update({'OMP_NUM_THREADS': '1'})
os.environ.update({'CUDA_VISIBLE_DEVICES': gpu_device})
print(' '.join(cmd))
subprocess.call(cmd, env=os.environ)