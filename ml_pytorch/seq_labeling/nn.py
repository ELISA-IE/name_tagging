import torch
import re
import torch.nn as nn
import numpy as np
import _pickle as cPickle
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from ml_pytorch.seq_labeling.loader import load_embedding
from ml_pytorch.ml_utils import init_param, log_sum_exp
from ml_pytorch import LongTensor, FloatTensor


class SeqLabeling(nn.Module):
    def __init__(self,
                 word_vocab_size,
                 word_dim,
                 word_lstm_dim,
                 word_bidirect,
                 crf,
                 label_size,
                 lr_method,
                 pre_emb,
                 **kwargs
                 ):
        super(SeqLabeling, self).__init__()

        self.word_emb = nn.Embedding(word_vocab_size, word_dim)

        self.word_lstm = nn.LSTM(word_dim, word_lstm_dim, 1,
                                 bidirectional=word_bidirect, batch_first=True)
        if word_bidirect:
            linear_input_dim = 2 * word_lstm_dim
        else:
            linear_input_dim = word_lstm_dim
        self.linear = nn.Linear(linear_input_dim, label_size)

        self.softmax = nn.Softmax()

        if crf:
            self.criterion = CRFLoss(label_size)
        else:
            self.criterion = CrossEntropyLoss()

        self.init_weights()

    def init_weights(self):
        init_param(self.word_emb)
        init_param(self.word_lstm)
        init_param(self.linear)
        init_param(self.criterion)

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

    def forward(self, inputs, batch_len):
        #
        # word embeddings
        #
        words = inputs['words']

        word_emb = self.word_emb(words.type(LongTensor))

        word_emb = torch.nn.utils.rnn.pack_padded_sequence(word_emb, batch_len,
                                                           batch_first=True)

        #
        # bi-directional lstm
        #
        word_lstm_out, word_lstm_h = self.word_lstm(word_emb)
        word_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(word_lstm_out,
                                                                  batch_first=True)

        #
        # fully connected layer
        #
        linear_out = self.linear(word_lstm_out)

        if type(self.criterion) == CrossEntropyLoss:
            outputs = torch.stack([self.softmax(linear_out[i]) for i in range(len(batch_len))], 0)
        elif type(self.criterion) == CRFLoss and not self.training:
            preds = linear_out
            outputs = []
            for i in range(len(batch_len)):
                valid_output = preds[i][:batch_len[i]]
                _output = self.criterion(valid_output, None,
                                         viterbi=True,
                                         return_best_sequence=True)
                outputs.append(_output)
        else:
            outputs = None

        #
        # compute batch loss
        #
        loss = 0
        batch_len = np.array(batch_len)
        if self.training:
            if type(self.criterion) == CrossEntropyLoss:
                preds = outputs
            elif type(self.criterion) == CRFLoss:
                preds = linear_out
            reference = inputs['tags']

            # for i in range(len(batch_len)):
            #     valid_output = preds[i][:batch_len[i]]
            #     valid_target = reference[i][:batch_len[i]]
            #
            #     step_loss = self.criterion(valid_output, valid_target)
            #
            #     loss += step_loss
            #
            # loss = loss / torch.sum(torch.from_numpy(batch_len))

            loss = self.criterion(preds, reference, batch_len)
        return outputs, loss

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


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred, ref, batch_len):
        mask = np.zeros((len(batch_len), max(batch_len)))
        for i in range(len(batch_len)):
            mask[i, range(batch_len[i])] = 1
        mask = Variable(torch.from_numpy(mask).type(FloatTensor))
        reshaped_pred = - torch.log(pred).view(-1, pred.size()[2])
        loss = reshaped_pred[torch.from_numpy(np.arange(reshaped_pred.size()[0])).type(LongTensor), ref.view(ref.numel()).data]

        loss = torch.sum(loss * mask.view(mask.numel()))

        loss = loss / torch.sum(torch.from_numpy(batch_len))

        return loss


class CRFLoss(nn.Module):
    def __init__(self, num_labels):
        super(CRFLoss, self).__init__()

        self.num_labels = num_labels
        self.transitions = Parameter(
            FloatTensor(num_labels+2, num_labels+2)
        )

    def forward(self, pred, ref, viterbi=False, return_best_sequence=False):
        seq_len = pred.size(0)

        small = -1000
        b_s = Variable(torch.from_numpy(
            np.array([[small] * self.num_labels + [0, small]]).astype(np.float32)
        ).type(FloatTensor))
        e_s = Variable(torch.from_numpy(
            np.array([[small] * self.num_labels + [small, 0]]).astype(np.float32)
        ).type(FloatTensor))
        observations = torch.cat(
            [pred, Variable(small * torch.ones((seq_len, 2)).type(FloatTensor))],
            dim=1
        )
        observations = torch.cat(
            [b_s, observations, e_s],
            dim=0
        )

        # compute all path scores
        paths_scores = []
        previous = observations[0]
        for obs in observations[1:]:
            _previous = torch.unsqueeze(previous, 1)
            _obs = torch.unsqueeze(obs, 0)
            if viterbi:
                scores = _previous + _obs + self.transitions
                out, out_indices = scores.max(dim=0)
                if return_best_sequence:
                    paths_scores.append((out, out_indices))
                else:
                    paths_scores.append(out)
                previous = out
            else:
                previous = log_sum_exp(_previous + _obs + self.transitions,
                                       dim=0)
                paths_scores.append(previous)

        if return_best_sequence:
            _, previous = paths_scores[-1][1].max(dim=0)
            sequence = []
            for s in paths_scores[::-1]:
                previous = s[1][previous]
                sequence.append(previous)

            sequence = torch.cat(sequence[::-1]+[paths_scores[-1][0].max(dim=0)[1]])

            return sequence[1:-1]

        all_paths_scores = log_sum_exp(paths_scores[-1], dim=0)

        # compute real path score if reference is provided
        if ref is not None:
            # Score from tags
            real_path_score = pred[torch.from_numpy(np.arange(seq_len)).type(LongTensor), ref.data].sum()

            # Score from transitions
            b_id = Variable(torch.from_numpy(np.array([self.num_labels], dtype=np.long)).type(LongTensor))
            e_id = Variable(torch.from_numpy(np.array([self.num_labels + 1], dtype=np.long)).type(LongTensor))
            padded_tags_ids = torch.cat([b_id, ref, e_id], dim=0)
            real_path_score += self.transitions[
                padded_tags_ids[torch.from_numpy(np.arange(seq_len + 1)).type(LongTensor)].data,
                padded_tags_ids[torch.from_numpy(np.arange(seq_len + 1) + 1).type(LongTensor)].data
            ].sum()

            # compute loss
            loss = all_paths_scores - real_path_score

            return loss


