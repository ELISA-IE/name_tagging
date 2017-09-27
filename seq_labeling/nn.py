import torch
import re
import torch.nn as nn
import numpy as np
import _pickle as cPickle
from torch.autograd import Variable
from seq_labeling.loader import load_embedding


class SeqLabeling(nn.Module):
    def __init__(self,
                 word_vocab_size,
                 word_dim,
                 word_lstm_dim,
                 word_bidirect,
                 label_size,
                 lr_method,
                 pre_emb,
                 **kwargs
                 ):
        super(SeqLabeling, self).__init__()
        self.word_emb = nn.Embedding(word_vocab_size, word_dim)

        self.word_lstm = nn.LSTM(word_dim, word_lstm_dim, 1, bidirectional=word_bidirect, batch_first=True)
        if word_bidirect:
            linear_input_dim = 2 * word_lstm_dim
        else:
            linear_input_dim = word_lstm_dim
        self.linear = nn.Linear(linear_input_dim, label_size)
        self.softmax = nn.Softmax()
        self.criterion = self.cross_entropy_loss

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.word_emb.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-initrange, initrange)

    def load_pretrained(self, id_to_word, pre_emb, word_dim, **kwargs):
        if not pre_emb:
            return

        # Initialize with pretrained embeddings
        new_weights = self.word_emb.weight.data
        print('Loading pretrained embeddings from %s...' % pre_emb)
        pretrained = {}
        emb_invalid = 0
        for i, line in enumerate(load_embedding(pre_emb)):
            if type(line) == bytes:
                try:
                    line = str(line, 'utf-8')
                except UnicodeDecodeError:
                    continue
            line = line.rstrip().split()
            if len(line) == word_dim + 1:
                pretrained[line[0]] = np.array(
                    [float(x) for x in line[1:]]
                ).astype(np.float32)
            else:
                emb_invalid += 1
        if emb_invalid > 0:
            print('WARNING: %i invalid lines' % emb_invalid)
        c_found = 0
        c_lower = 0
        c_zeros = 0
        # Lookup table initialization
        for i in range(len(id_to_word)):
            word = id_to_word[i]
            if word in pretrained:
                new_weights[i] = torch.from_numpy(pretrained[word])
                c_found += 1
            elif word.lower() in pretrained:
                new_weights[i] = torch.from_numpy(pretrained[word.lower()])
                c_lower += 1
            elif re.sub('\d', '0', word.lower()) in pretrained:
                new_weights[i] = torch.from_numpy(pretrained[
                    re.sub('\d', '0', word.lower())
                ])
                c_zeros += 1
        self.word_emb.weight = nn.Parameter(new_weights)

        print('Loaded %i pretrained embeddings.' % len(pretrained))
        print('%i / %i (%.4f%%) words have been initialized with '
              'pretrained embeddings.' % (
                  c_found + c_lower + c_zeros, len(id_to_word),
                  100. * (c_found + c_lower + c_zeros) / len(id_to_word)
              ))
        print('%i found directly, %i after lowercasing, '
              '%i after lowercasing + zero.' % (
                  c_found, c_lower, c_zeros
              ))

    def cross_entropy_loss(self, pred, target):
        onehot_target = torch.zeros(pred.size())
        onehot_target[range(len(target)), target.data] = 1
        onehot_target = Variable(onehot_target)

        loss = torch.sum(- onehot_target * torch.log(pred))

        return loss

    def forward(self, inputs, batch_len):
        words = inputs['words']

        word_emb = self.word_emb(words)

        word_emb = torch.nn.utils.rnn.pack_padded_sequence(word_emb, batch_len,
                                                           batch_first=True)
        word_lstm_out, word_lstm_h = self.word_lstm(word_emb)
        word_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(word_lstm_out,
                                                                  batch_first=True)

        linear_out = self.linear(word_lstm_out)

        softmax_out = torch.stack([self.softmax(linear_out[i]) for i in range(len(batch_len))], 0)

        return softmax_out

    def loss(self, preds, reference, batch_len):
        loss = 0

        batch_len = np.array(batch_len)
        max_length = max(batch_len)
        for step in range(max_length):
            if np.sum(batch_len > step) == 0:
                break
            mask_vector = torch.from_numpy(
                (batch_len > step).astype(np.int32)).byte()
            index_vector = Variable(
                torch.masked_select(torch.arange(0, len(batch_len)),
                                    mask_vector).long())
            valid_output = torch.index_select(preds[:, step], 0, index_vector)
            valid_target = torch.index_select(reference[:, step], 0,
                                              index_vector)

            step_loss = self.criterion(valid_output, valid_target)
            loss += step_loss

        return loss / torch.sum(torch.from_numpy(batch_len))

    def save_mappings(self, id_to_word, id_to_char, id_to_tag, id_to_feat_list):
        """
        We need to save the mappings if we want to use the model later.
        """
        self.id_to_word = id_to_word
        self.id_to_char = id_to_char
        self.id_to_tag = id_to_tag
        self.id_to_feat_list = id_to_feat_list  # boliang
        with open(self.mappings_path, 'wb') as f:
            mappings = {
                'id_to_word': self.id_to_word,
                'id_to_char': self.id_to_char,
                'id_to_tag': self.id_to_tag,
                'id_to_feat_list': self.id_to_feat_list  # boliang
            }
            cPickle.dump(mappings, f)



