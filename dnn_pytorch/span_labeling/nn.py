import torch
import re
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from dnn_pytorch.seq_labeling.nn import MultiLeNetConv2dLayer
from dnn_pytorch.seq_labeling.loader import load_embedding

from dnn_pytorch.dnn_utils import init_param, init_variable, log_sum_exp, sequence_mask
from dnn_pytorch import LongTensor, FloatTensor


class SpanLabeling(nn.Module):
    def __init__(self, model_param):
        super(SpanLabeling, self).__init__()

        #
        # model parameters
        #
        self.model_param = model_param
        word_vocab_size = model_param['word_vocab_size']
        char_vocab_size = model_param['char_vocab_size']
        feat_vocab_size = model_param['feat_vocab_size']
        word_dim = model_param['word_dim']
        word_lstm_dim = model_param['word_lstm_dim']
        char_dim = model_param['char_dim']
        char_lstm_dim = model_param['char_lstm_dim']
        feat_dim = model_param['feat_dim']
        dropout = model_param['dropout']
        char_conv = model_param['char_conv']
        label_size = model_param['label_size']
        label_weights = model_param['label_weights']

        # initialize word lstm input dim to 0
        word_lstm_input_dim = 0

        #
        # word embedding layer
        #
        self.word_emb = nn.Embedding(word_vocab_size, word_dim)
        word_lstm_input_dim += word_dim

        #
        # char embedding layer
        #
        if char_dim:
            self.char_dim = char_dim
            self.char_emb = nn.Embedding(char_vocab_size, char_dim)

        #
        # bi-lstm char layer
        #
        if char_lstm_dim:
            self.char_lstm_init_hidden = (
                Parameter(torch.randn(2, 1, char_lstm_dim).type(
                    FloatTensor)),
                Parameter(torch.randn(2, 1, char_lstm_dim).type(
                    FloatTensor))
            )
            self.char_lstm_dim = char_lstm_dim
            self.char_lstm = nn.LSTM(char_dim, char_lstm_dim, 1,
                                     bidirectional=True, batch_first=True)
            word_lstm_input_dim += 2 * char_lstm_dim

        # cnn char layer
        if char_conv:
            max_length = 25
            out_channel = char_conv
            kernel_sizes = [(2, char_dim), (3, char_dim), (4, char_dim)]
            kernel_shape = []
            for i in range(len(kernel_sizes)):
                kernel_shape.append([1, out_channel, kernel_sizes[i]])
            pool_sizes = [(max_length - 2 + 1, 1),
                          (max_length - 3 + 1, 1),
                          (max_length - 4 + 1, 1)]
            self.multi_convs = MultiLeNetConv2dLayer(kernel_shape, pool_sizes)
            word_lstm_input_dim += out_channel * len(kernel_sizes)

        #
        # feat dim
        #
        if feat_dim:
            self.feat_emb = [nn.Embedding(v, feat_dim) for v in feat_vocab_size]
            word_lstm_input_dim += len(self.feat_emb) * feat_dim

        #
        # dropout for word bi-lstm layer
        #
        if dropout:
            self.word_lstm_dropout = nn.Dropout(p=dropout)

        #
        # word bi-lstm layer
        #
        self.word_lstm_init_hidden = (
            Parameter(torch.randn(2, 1, word_lstm_dim).type(FloatTensor)),
            Parameter(torch.randn(2, 1, word_lstm_dim).type(FloatTensor)),
        )
        self.word_lstm_dim = word_lstm_dim
        self.word_lstm = nn.LSTM(word_lstm_input_dim, word_lstm_dim, 1,
                                 bidirectional=True, batch_first=True)

        #
        # tanh layer
        #
        tanh_layer_input_dim = 2 * word_lstm_dim
        self.tanh_linear = nn.Linear(tanh_layer_input_dim,
                                     word_lstm_dim)

        #
        # attention mechanism to generate span
        #
        self.attention = Attention(word_lstm_dim)

        #
        # linear layer before loss
        #
        self.linear = nn.Linear(3 * word_lstm_dim, label_size)

        #
        # loss
        #
        self.softmax = nn.Softmax()
        self.criterion = BalancedCrossEntropyLoss(label_weights)

        #
        # initialize weights of each layer
        #
        self.init_weights()

    def init_weights(self):
        init_param(self.word_emb)

        if self.model_param['char_dim']:
            init_param(self.char_emb)
        if self.model_param['char_lstm_dim']:
            init_param(self.char_lstm)
            self.char_lstm.flatten_parameters()
        if self.model_param['char_conv']:
            init_param(self.multi_convs)
        if self.model_param['feat_dim']:
            for f_e in self.feat_emb:
                init_param(f_e)

        init_param(self.word_lstm)
        self.word_lstm.flatten_parameters()

        init_param(self.tanh_linear)

        init_param(self.attention)

        init_param(self.linear)

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
                new_weights[i] = torch.from_numpy(
                    pretrained[re.sub('\d', '0', word.lower())]
                )
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

    def forward(self, inputs):
        seq_len = inputs['seq_len']
        char_len = inputs['char_len']
        char_index_mapping = inputs['char_index_mapping']

        seq_len = np.array(seq_len)
        char_len = np.array(char_len)
        batch_size = len(seq_len)

        word_lstm_input = []
        #
        # word embeddings
        #
        words = inputs['words']

        word_emb = self.word_emb(words.type(LongTensor))
        word_lstm_input.append(word_emb)

        #
        # char embeddings
        #
        char_repr = []
        if self.model_param['char_dim']:
            chars = inputs['chars']
            char_emb = self.char_emb(chars.type(LongTensor))

        #
        # char bi-lstm embeddings
        #
        if self.model_param['char_lstm_dim']:
            lstm_char_emb = char_emb[:, :char_len[0]]
            char_lstm_dim = self.model_param['char_lstm_dim']
            char_lstm_init_hidden = (
                self.char_lstm_init_hidden[0].expand(2, len(char_len), char_lstm_dim).contiguous(),
                self.char_lstm_init_hidden[1].expand(2, len(char_len), char_lstm_dim).contiguous(),
            )
            lstm_char_emb = torch.nn.utils.rnn.pack_padded_sequence(
                lstm_char_emb, char_len, batch_first=True
            )
            char_lstm_out, char_lstm_h = self.char_lstm(
                lstm_char_emb, char_lstm_init_hidden
            )
            char_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                char_lstm_out, batch_first=True
            )
            char_lstm_h = char_lstm_h[0].permute(1, 0, 2).contiguous().view(len(char_len), 2*char_lstm_dim)
            char_repr.append(char_lstm_h)

        #
        # char CNN embeddings
        #
        if self.model_param['char_conv']:
            cnn_char_emb = char_emb[:, :25]
            char_cnn_out = self.multi_convs(cnn_char_emb)
            char_repr += char_cnn_out

        if char_repr:
            char_repr = torch.cat(char_repr, dim=1)

            char_index_mapping = LongTensor([char_index_mapping[k] for k in range(len(char_len))])
            char_repr = char_repr[char_index_mapping]

            char_repr_padded_seq = nn.utils.rnn.PackedSequence(data=char_repr, batch_sizes=seq_len.tolist())
            char_repr, _ = nn.utils.rnn.pad_packed_sequence(
                char_repr_padded_seq
            )
            word_lstm_input.append(char_repr)

        #
        # feat input
        #
        if self.model_param['feat_dim']:
            feat_emb = []
            for i, f_e in enumerate(self.feat_emb):
                feat = inputs['feats'][:, :, i]
                feat_emb.append(f_e(feat.type(LongTensor)))
            word_lstm_input += feat_emb

        #
        # bi-directional lstm
        #
        word_lstm_dim = self.model_param['word_lstm_dim']
        word_lstm_input = torch.cat(word_lstm_input, dim=2)
        # dropout
        if self.model_param['dropout']:
            word_lstm_input = self.word_lstm_dropout(word_lstm_input)

        word_lstm_init_hidden = (
            self.word_lstm_init_hidden[0].expand(2, batch_size, word_lstm_dim).contiguous(),
            self.word_lstm_init_hidden[1].expand(2, batch_size, word_lstm_dim).contiguous()
        )

        word_lstm_input = torch.nn.utils.rnn.pack_padded_sequence(
            word_lstm_input, seq_len, batch_first=True
        )
        word_lstm_out, word_lstm_h = self.word_lstm(
            word_lstm_input, word_lstm_init_hidden
        )
        word_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            word_lstm_out, batch_first=True
        )

        #
        # tanh layer
        #
        tanh_out = nn.Tanh()(self.tanh_linear(word_lstm_out))

        #
        # generate spans using span index
        #
        spans = inputs['spans']
        span_len = inputs['span_len']
        span_tags = inputs['span_tags']
        span_pos = inputs['span_pos']

        span_repr = []
        for i, s in enumerate(spans):
            s_pos = span_pos[i][0]
            s_repr = tanh_out[s_pos][s]
            span_repr.append(s_repr)
        span_repr = torch.stack(span_repr)

        # compute span representation using attention machanism
        attentive_span_repr = self.attention(span_repr, span_len)

        # combine attentive span representation with head and tail embeddings.
        head = span_repr[:, 0]
        tail = span_repr[torch.from_numpy(np.arange(len(spans))).type(LongTensor),
                         torch.from_numpy(np.array(span_len) - 1).type(LongTensor)]
        final_span_repr = torch.cat([head, attentive_span_repr, tail], dim=1)

        # fully connected layer
        linear_out = self.linear(final_span_repr)

        span_outputs = self.softmax(linear_out)
        # print(span_outputs)

        # cross entropy loss
        loss = 0
        if self.training:
            reference = span_tags

            loss = self.criterion(span_outputs, reference) / len(spans)

        return span_outputs, loss


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()

        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.tanh = torch.nn.Tanh()
        self.v = Parameter(
            torch.from_numpy(init_variable((hidden_dim))).type(FloatTensor)
        )
        self.softmax = torch.nn.Softmax()

    def forward(self, inputs, span_len):
        e = torch.matmul(self.tanh(self.linear(inputs)), self.v)

        zero_one_mask = Variable(FloatTensor(sequence_mask(span_len)))

        mask = (zero_one_mask - 1) * 1000

        masked_e = e + mask

        alpha = torch.unsqueeze(self.softmax(masked_e) * zero_one_mask, 2)

        x = torch.sum(inputs * alpha, dim=1)

        return x


class BalancedCrossEntropyLoss(nn.Module):
    def __init__(self, label_weights):
        super(BalancedCrossEntropyLoss, self).__init__()

        self.label_weights = Variable(FloatTensor(label_weights))

    def forward(self, pred, ref):
        # compute cross entropy loss
        loss = - (torch.log(pred)*self.label_weights)[torch.from_numpy(np.arange(pred.size()[0])).type(LongTensor), ref.data]
        loss = torch.sum(loss)

        return loss

