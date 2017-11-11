import torch
import re
import torch.nn as nn
import numpy as np
import _pickle as cPickle
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from ml_pytorch.seq_labeling.loader import load_embedding
from ml_pytorch.ml_utils import init_param, log_sum_exp, sequence_mask
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
                                 bidirectional=word_bidirect,
                                 batch_first=True)
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

    def forward(self, inputs, batch_len):
        #
        # word embeddings
        #
        words = inputs['words']

        word_emb = self.word_emb(words.type(LongTensor))

        word_emb = torch.nn.utils.rnn.pack_padded_sequence(
            word_emb, batch_len, batch_first=True
        )

        #
        # bi-directional lstm
        #
        word_lstm_out, word_lstm_h = self.word_lstm(word_emb)
        word_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            word_lstm_out, batch_first=True
        )

        #
        # fully connected layer
        #
        linear_out = self.linear(word_lstm_out)

        if type(self.criterion) == CrossEntropyLoss:
            outputs = torch.stack(
                [self.softmax(linear_out[i]) for i in range(len(batch_len))], 0
            )
        elif type(self.criterion) == CRFLoss and not self.training:
            preds = linear_out
            outputs = self.criterion(
                preds, None, batch_len, viterbi=True, return_best_sequence=True
            )
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

            loss = self.criterion(preds, reference, batch_len)
            loss = loss / torch.sum(torch.from_numpy(batch_len))

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
        batch_size = pred.size(0)
        seq_len = pred.size(1)

        mask = sequence_mask(batch_len)
        mask = Variable(torch.from_numpy(mask).type(FloatTensor))

        loss = - torch.log(pred)[
            torch.from_numpy(
                np.array([np.arange(batch_size)] * seq_len).transpose()
            ).type(LongTensor),
            torch.from_numpy(
                np.array([np.arange(seq_len)] * batch_size)
            ).type(LongTensor),
            ref.data
        ]
        loss = torch.sum(loss * mask)

        return loss


class CRFLoss(nn.Module):
    def __init__(self, num_labels):
        super(CRFLoss, self).__init__()

        self.num_labels = num_labels
        self.transitions = Parameter(
            FloatTensor(num_labels+2, num_labels+2)
        )

    def forward(self, pred, ref, batch_len,
                viterbi=False, return_best_sequence=False):
        batch_size = pred.size(0)
        seq_len = pred.size(1)
        label_size = pred.size(2)

        small = -1000
        b_s_array = np.array(
            [[[small] * self.num_labels + [0, small]]] * batch_size
        ).astype(np.float32)
        b_s = Variable(torch.from_numpy(b_s_array)).type(FloatTensor)
        right_padding_array = np.array(
            [[[0] * self.num_labels + [small, small]]] * batch_size
        ).astype(np.float32)
        right_padding = Variable(
            torch.from_numpy(right_padding_array)
        ).type(FloatTensor)
        observations = torch.cat(
            [pred, Variable(
                small * torch.ones((batch_size, seq_len, 2))
            ).type(FloatTensor)],
            dim=2
        )
        observations = torch.cat(
            [b_s, observations, right_padding],
            dim=1
        )

        e_s = np.array([small] * self.num_labels + [0, 1000]).astype(np.float32)
        e_s_mask = np.zeros(observations.size())
        for i in range(batch_size):
            e_s_mask[i][batch_len[i]+1] = e_s

        observations += Variable(torch.from_numpy(e_s_mask)).type(FloatTensor)

        # compute all path scores
        paths_scores = Variable(
            FloatTensor(seq_len+1, batch_size, label_size+2)
        )
        paths_indices = Variable(
            LongTensor(seq_len+1, batch_size, label_size+2)
        )
        previous = observations[:, 0]
        for i in range(1, observations.size(1)):
            obs = observations[:, i]
            _previous = torch.unsqueeze(previous, 2)
            _obs = torch.unsqueeze(obs, 1)
            if viterbi:
                scores = _previous + _obs + self.transitions
                out, out_indices = scores.max(dim=1)
                if return_best_sequence:
                    paths_indices[i-1] = out_indices
                paths_scores[i-1] = out
                previous = out
            else:
                previous = log_sum_exp(_previous + _obs + self.transitions,
                                       dim=1)
                paths_scores[i-1] = previous

        paths_scores = paths_scores.permute(1, 0, 2)
        paths_indices = paths_indices.permute(1, 0, 2)

        if return_best_sequence:
            sequence = []
            for i in range(len(paths_indices)):
                p_indices = paths_indices[i][:batch_len[i]+1]
                p_score = paths_scores[i][:batch_len[i]+1]
                _, previous = p_indices[-1].max(dim=0)
                seq = []
                for j in reversed(range(len(p_score))):
                    s = p_indices[j]
                    previous = s[previous]
                    seq.append(previous)

                seq = torch.cat(seq[::-1]+[p_score[-1].max(dim=0)[1]])

                sequence.append(seq[1:-1])

            return sequence

        all_paths_scores = log_sum_exp(
            paths_scores[
                torch.from_numpy(np.arange(batch_size)).type(LongTensor),
                torch.from_numpy(batch_len).type(LongTensor)
            ],
            dim=1
        ).sum()

        # compute real path score if reference is provided
        if ref is not None:
            # Score from tags
            real_path_mask = Variable(
                torch.from_numpy(sequence_mask(batch_len))
            ).type(FloatTensor)
            real_path_score = pred[
                torch.from_numpy(
                    np.array([np.arange(batch_size)]*seq_len).transpose()
                ).type(LongTensor),
                torch.from_numpy(
                    np.array([np.arange(seq_len)]*batch_size)
                ).type(LongTensor),
                ref.data
            ]
            real_path_score = torch.sum(real_path_score * real_path_mask)

            # Score from transitions
            b_id = Variable(
                torch.from_numpy(
                    np.array([[self.num_labels]] * batch_size)
                ).type(LongTensor)
            )
            right_padding = Variable(torch.zeros(b_id.size())).type(LongTensor)

            padded_tags_ids = torch.cat([b_id, ref, right_padding], dim=1)

            e_id = np.array([self.num_labels+1])
            e_id_mask = np.zeros(padded_tags_ids.size())
            for i in range(batch_size):
                e_id_mask[i][batch_len[i] + 1] = e_id

            padded_tags_ids += Variable(
                torch.from_numpy(e_id_mask)
            ).type(LongTensor)

            transition_score_mask = Variable(
                torch.from_numpy(sequence_mask(batch_len+1))
            ).type(FloatTensor)
            real_transition_score = self.transitions[
                padded_tags_ids[
                :, torch.from_numpy(np.arange(seq_len + 1)).type(LongTensor)
                ].data,
                padded_tags_ids[
                :, torch.from_numpy(np.arange(seq_len + 1) + 1).type(LongTensor)
                ].data
            ]
            real_path_score += torch.sum(
                real_transition_score * transition_score_mask
            )

            # compute loss
            loss = all_paths_scores - real_path_score

            return loss

