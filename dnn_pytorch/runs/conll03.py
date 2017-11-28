import os
import subprocess

gpu_device = '0'

train = '/nas/data/m1/zhangb8/ml/data/ner/CoNLL/eng03/eng.train.bio'
dev = '/nas/data/m1/zhangb8/ml/data/ner/CoNLL/eng03/eng.testa.bio'
test = '/nas/data/m1/zhangb8/ml/data/ner/CoNLL/eng03/eng.testb.bio'

model_dir = '/nas/data/m1/zhangb8/ml/model/pytorch/conll03'

# pre_emb = ''
pre_emb = '/nas/data/m1/zhangb8/ml/data/embeddings/lample_pretrained/eng.Skip100'

# cluster = ''
cluster = '/nas/data/m1/zhangb8/kbp_edl/data/word_clusters/eng/paths'


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
    '--zeros', '1',
    '--char_dim', '25',
    '--char_lstm_dim', '25',
    '--char_conv_channel', '25',
    '--word_dim', '100',
    '--word_lstm_dim', '100',
    '--pre_emb', pre_emb,
    '--all_emb', '1',
    '--cap_dim', '0',
    '--crf', '1',
    '--dropout', '0.5',
    '--lr_method', 'sgd-init_lr=.005-lr_decay_epoch=100',
    '--batch_size', '20',

    # external feature settings
    '--feat_dim', '5',
    '--upenn_stem', '',
    '--pos_model', '',
    '--cluster', cluster,
    '--ying_stem', ''
]

os.environ.update({'OMP_NUM_THREADS': '1'})
os.environ.update({'CUDA_VISIBLE_DEVICES': gpu_device})
print(' '.join(cmd))
subprocess.call(cmd, env=os.environ)