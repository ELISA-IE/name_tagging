import os
import sys
import argparse
import time
import timeit
import random
import torch
import itertools
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from collections import OrderedDict

from ml_pytorch.seq_labeling.nn import SeqLabeling
from ml_pytorch.seq_labeling.utils import create_input, Tee
from ml_pytorch.seq_labeling.utils import evaluate, eval_script
from ml_pytorch.seq_labeling.loader import word_mapping, char_mapping, tag_mapping, feats_mapping
from ml_pytorch.seq_labeling.loader import update_tag_scheme, prepare_dataset, load_sentences
from ml_pytorch.seq_labeling.loader import augment_with_pretrained
# external features
from ml_pytorch.seq_labeling.generate_features import generate_features
from ml_pytorch.ml_utils import exp_lr_scheduler


# Read parameters from command line
parser = argparse.ArgumentParser()
parser.add_argument(
    "-T", "--train", default="",
    help="Train set location"
)
parser.add_argument(
    "-d", "--dev", default="",
    help="Dev set location"
)
parser.add_argument(
    "-t", "--test", default="",
    help="Test set location"
)
parser.add_argument(
    "-m", "--model_dp", default="",
    help="model directory path"
)
parser.add_argument(
    "-s", "--tag_scheme", default="iobes",
    help="Tagging scheme (IOB or IOBES)"
)
parser.add_argument(
    "-l", "--lower", default='0',
    type=int, help="Lowercase words (this will not affect character inputs)"
)
parser.add_argument(
    "-z", "--zeros", default="0",
    type=int, help="Replace digits with 0"
)
parser.add_argument(
    "-c", "--char_dim", default="25",
    type=int, help="Char embedding dimension"
)
parser.add_argument(
    "-C", "--char_lstm_dim", default="25",
    type=int, help="Char LSTM hidden layer size"
)
parser.add_argument(
    "-V", "--char_conv_channel", default="1",
    type=int, help="Use CNN to generate character embeddings. (0 to disable)"
)
parser.add_argument(
    "-w", "--word_dim", default="100",
    type=int, help="Token embedding dimension"
)
parser.add_argument(
    "-W", "--word_lstm_dim", default="100",
    type=int, help="Token LSTM hidden layer size"
)
parser.add_argument(
    "-p", "--pre_emb", default="",
    help="Location of pretrained embeddings"
)
parser.add_argument(
    "-A", "--all_emb", default="0",
    type=int, help="Load all embeddings"
)
parser.add_argument(
    "-a", "--cap_dim", default="0",
    type=int, help="Capitalization feature dimension (0 to disable)"
)
parser.add_argument(
    "-f", "--crf", default="1",
    type=int, help="Use CRF (0 to disable)"
)
parser.add_argument(
    "-D", "--dropout", default="0.5",
    type=float, help="Droupout on the input (0 = no dropout)"
)
parser.add_argument(
    "-L", "--lr_method", default="sgd-lr_.005",
    help="Learning method (SGD, Adadelta, Adam..)"
)
parser.add_argument(
    "-r", "--reload", default="0",
    type=int, help="Reload the last saved model"
)
parser.add_argument(
    "--num_epochs", default="100",
    type=int, help="Number of training epochs"
)
parser.add_argument(
    "--batch_size", default="5",
    type=int, help="Batch size."
)
#
# external features
#
parser.add_argument(
    "--feat_dim", default="0",
    type=int, help="dimension for each feature."
)
parser.add_argument(
    "--comb_method", default="0",
    type=int, help="combination method. (1, 2, 3 or 4)"
)
parser.add_argument(
    "--upenn_stem", default="",
    help="path of upenn morphology analysis result."
)
parser.add_argument(
    "--pos_model", default="",
    help="path of pos tagger model."
)
parser.add_argument(
    "--cluster", default="",
    help="path of brown cluster paths."
)
parser.add_argument(
    "--ying_stem", default="",
    help="path of Ying's stemming result."
)
parser.add_argument(
    "--gaz", default="", nargs="+",
    help="gazetteers paths."
)

args = parser.parse_args()

# parse model dir
model_dir = args.model_dp

# Parse parameters
parameters = OrderedDict()
parameters['tag_scheme'] = args.tag_scheme
parameters['lower'] = args.lower == 1
parameters['zeros'] = args.zeros == 1
parameters['char_dim'] = args.char_dim
parameters['char_lstm_dim'] = args.char_lstm_dim
parameters['char_conv_channel'] = args.char_conv_channel
parameters['word_dim'] = args.word_dim
parameters['word_lstm_dim'] = args.word_lstm_dim
parameters['pre_emb'] = args.pre_emb
parameters['all_emb'] = args.all_emb == 1
parameters['cap_dim'] = args.cap_dim
parameters['crf'] = args.crf == 1
parameters['dropout'] = args.dropout
parameters['lr_method'] = args.lr_method
parameters['num_epochs'] = args.num_epochs
parameters['batch_size'] = args.batch_size
# external features
parameters['feat_dim'] = args.feat_dim
parameters['comb_method'] = args.comb_method
parameters['upenn_stem'] = args.upenn_stem
parameters['pos_model'] = args.pos_model
parameters['cluster'] = args.cluster
parameters['ying_stem'] = args.ying_stem
parameters['gaz'] = args.gaz

# generate model name
model_name = []
for k, v in parameters.items():
    if not v:
        continue
    if k in ['pre_emb', 'pos_model', 'cluster']:
        v = os.path.basename(v)
    model_name.append('='.join((k, str(v))))
model_dir = os.path.join(model_dir, ','.join(model_name[:-5]))

# Check parameters validity
assert os.path.isfile(args.train)
assert os.path.isfile(args.dev)
assert os.path.isfile(args.test)
assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0 or parameters['exp_feat_dim'] > 0
assert 0. <= parameters['dropout'] < 1.0
assert parameters['tag_scheme'] in ['iob', 'iobes', 'classification']
assert not parameters['all_emb'] or parameters['pre_emb']
assert not parameters['pre_emb'] or parameters['word_dim'] > 0
assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])
if parameters['upenn_stem']:
    assert os.path.exists(parameters['upenn_stem']) and \
           parameters['comb_method'] != 0
if parameters['pos_model']:
    assert os.path.exists(parameters['pos_model']) and \
           parameters['comb_method'] != 0
if parameters['cluster']:
    assert os.path.exists(parameters['cluster']) and \
           parameters['comb_method'] != 0
if parameters['ying_stem']:
    assert os.path.exists(parameters['ying_stem']) and \
           parameters['comb_method'] != 0

# Check evaluation script / folders
if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
os.makedirs(model_dir, exist_ok=True)

# register logger to save print(messages to both stdout and disk)
training_log_path = os.path.join(model_dir, 'training_log.txt')
if os.path.exists(training_log_path):
    os.remove(training_log_path)
f = open(training_log_path, 'w')
sys.stdout = Tee(sys.stdout, f)

print('Training data: %s' % args.train)
print('Dev data: %s' % args.dev)
print('Test data: %s' % args.test)
print("Model location: %s" % model_dir)
print('Model parameters:')
for k, v in parameters.items():
    print('%s=%s' % (k, v))

# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

# Load sentences
train_sentences = load_sentences(args.train, lower, zeros)
dev_sentences = load_sentences(args.dev, lower, zeros)
test_sentences = load_sentences(args.test, lower, zeros)

# train_sentences = train_sentences[:50]
# dev_sentences = dev_sentences[:50]
# test_sentences = test_sentences[:50]

# Use selected tagging scheme (IOB / IOBES), also check tagging scheme
update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)

#
# generate external features
#
train_feats, train_stem = generate_features(train_sentences, parameters)
dev_feats, dev_stem = generate_features(dev_sentences, parameters)
test_feats, test_stem = generate_features(test_sentences, parameters)

# Create a dictionary / mapping of words
# If we use pretrained embeddings, we add them to the dictionary.
if parameters['pre_emb']:
    dico_words_train = word_mapping(train_sentences, lower)[0]
    dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        parameters['pre_emb'],
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in dev_sentences + test_sentences])
        ) if not parameters['all_emb'] else None
    )
else:
    dico_words, word_to_id, id_to_word = word_mapping(train_sentences, lower)
    dico_words_train = dico_words

# Create a dictionary and a mapping for words / POS tags / tags
dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)
# create a dictionary and a mapping for each feature
dico_feats_list, feat_to_id_list, id_to_feat_list = feats_mapping(train_feats)

parameters['label_size'] = len(id_to_tag)
parameters['word_vocab_size'] = len(id_to_word)
parameters['char_vocab_size'] = len(id_to_char)
parameters['feat_vocab_size'] = [len(item) for item in id_to_feat_list]

# Index data
dataset = dict()
dataset['train'] = prepare_dataset(
    train_sentences,
    train_feats, train_stem,
    word_to_id, char_to_id, tag_to_id, feat_to_id_list, lower
)
dataset['dev'] = prepare_dataset(
    dev_sentences,
    dev_feats, dev_stem,
    word_to_id, char_to_id, tag_to_id, feat_to_id_list, lower
)
dataset['test'] = prepare_dataset(
    test_sentences,
    test_feats, test_stem,
    word_to_id, char_to_id, tag_to_id, feat_to_id_list, lower
)

print("%i / %i / %i sentences in train / dev / test." % (
    len(dataset['train']), len(dataset['dev']), len(dataset['test'])))

# initialize model
print('model initializing...')
model = SeqLabeling(parameters)
model.load_pretrained(id_to_word, **parameters)

# Parse optimization method parameters
lr_method = parameters['lr_method']
if "-" in lr_method:
    lr_method_name = lr_method[:lr_method.find('-')]
    lr_method_parameters = {}
    for x in lr_method[lr_method.find('-') + 1:].split('-'):
        split = x.split('=')
        assert len(split) == 2
        lr_method_parameters[split[0]] = float(split[1])
else:
    lr_method_name = lr_method
    lr_method_parameters = {}
# initialize optimizer function
if lr_method_name == 'sgd':
    optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
else:
    print('unknown optimization method.')

since = time.time()
best_model = model
best_dev = 0.0
best_test = 0.0
metric = 'f1'  # use metric 'f1' or 'acc'
num_epochs = args.num_epochs
batch_size = args.batch_size

for epoch in range(num_epochs):
    print('-' * 10)
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    time_epoch_start = time.time()  # epoch start time

    # Each epoch has a training and validation phase
    for phase in ['train', 'dev', 'test'][:]:
        if phase == 'train':
            optimizer = exp_lr_scheduler(optimizer_ft, epoch,
                                         **lr_method_parameters)
            model.train(True)  # Set model to training mode
            random.shuffle(dataset[phase])
        else:
            model.train(False)  # Set model to evaluate mode

        epoch_loss = []

        # Iterate over data.
        preds = []
        for i in range(0, len(dataset[phase]), batch_size):
            inputs, seq_index_mapping, char_index_mapping, seq_len, char_len = create_input(dataset[phase][i:i+batch_size], parameters)

            # forward
            outputs, loss = model.forward(inputs, seq_len, char_len, char_index_mapping)
            try:
                epoch_loss.append(loss.data[0])
            except AttributeError:
                pass

            # backward + optimize only if in training phase
            if phase == 'train':
                # zero the parameter gradients
                optimizer.zero_grad()

                loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem
                # in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm(model.parameters(), 5)

                optimizer.step()

                sys.stdout.write(
                    '%d instances processed. current batch loss: %f\r' %
                    (i, np.mean(epoch_loss))
                )
                sys.stdout.flush()
            else:
                if parameters['crf']:
                    preds += [outputs[seq_index_mapping[j]].data
                              for j in range(len(outputs))]
                else:
                    _, _preds = torch.max(outputs.data, 2)

                    preds += [
                        _preds[seq_index_mapping[j]][:seq_len[seq_index_mapping[j]]]
                        for j in range(len(seq_index_mapping))
                        ]

        if phase == 'train':
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            print('{} Loss: {:.4f}\n'.format(phase, epoch_loss))
        else:
            epoch_f1, epoch_acc, predicted_bio = evaluate(preds, dataset[phase], id_to_tag)
            if metric == 'f1':
                epoch_score = epoch_f1
            elif metric == 'acc':
                epoch_score = epoch_acc
            print(
                '{} F1: {:.4f} Acc: {:.4f}\n'.format(phase, epoch_f1, epoch_acc)
            )

        # deep copy the model
        if phase == 'dev' and epoch_score > best_dev:
            best_dev = epoch_score
            print('new best score on dev: %.4f' % best_dev)
            print('saving the current model to disk...')

            state = {
                'epoch': epoch + 1,
                'parameters': parameters,
                'mappings': {
                    'id_to_word': id_to_word,
                    'id_to_char': id_to_char,
                    'id_to_tag': id_to_tag,
                    'id_to_feat_list': id_to_feat_list  # boliang
                },
                'state_dict': model.state_dict(),
                'best_prec1': best_dev,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, os.path.join(model_dir, 'best_model.pth.tar'))
            with open(os.path.join(model_dir, 'best_dev.ner.bio'), 'w') as f:
                f.write(predicted_bio)

        if phase == 'test' and epoch_score > best_test:
            best_test = epoch_score
            print('new best score on test: %.4f' % best_test)
            with open(os.path.join(model_dir, 'best_test.ner.bio'), 'w') as f:
                f.write(predicted_bio)

    time_epoch_end = time.time()  # epoch end time
    print('epoch training time: %f seconds' % round(
        (time_epoch_end - time_epoch_start), 2))

