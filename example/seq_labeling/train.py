import os
import subprocess


train = 'data/eng.train.bio'
dev = 'data/eng.testa.bio'
test = 'data/eng.testb.bio'

model_dir = 'model/'

# use word2vec to generate pre-trained embeddings.
# tutorial: https://code.google.com/archive/p/word2vec/
pre_emb = ''
# pre_emb = '/nas/data/m1/zhangb8/ml/data/embeddings/lample_pretrained/eng.Skip100'

#
# run command
#
script = '../../dnn_pytorch/seq_labeling/train.py'
cmd = [
    'python3',
    script,
    # data settings
    '--train', train,
    '--dev', dev,
    '--test', test,
    '--model_dp', model_dir,
    '--tag_scheme', 'iobes',

    # parameter settings
    '--lower', '0',
    '--zeros', '1',
    '--char_dim', '25',
    '--char_lstm_dim', '25',
    '--char_conv_channel', '25',
    '--word_dim', '100',
    '--word_lstm_dim', '100',
    '--pre_emb', pre_emb,
    '--all_emb', '0',
    '--cap_dim', '0',
    '--feat_dim', '5',
    '--feat_column', '1',
    '--crf', '1',
    '--dropout', '0.5',
    '--lr_method', 'sgd-init_lr=.005-lr_decay_epoch=100',
    '--batch_size', '20',
    '--gpu', '0',
]

# set OMP threads to 1
os.environ.update({'OMP_NUM_THREADS': '1'})
# set which gpu to use if gpu option is turned on
gpu_device = '0'
os.environ.update({'CUDA_VISIBLE_DEVICES': gpu_device})

print(' '.join(cmd))
subprocess.call(cmd, env=os.environ)
