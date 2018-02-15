import itertools
import torch
import numpy as np
from collections import defaultdict

from torch.autograd import Variable
from dnn_pytorch import LongTensor

from dnn_pytorch.seq_labeling import utils

try:
    import _pickle as cPickle
except ImportError:
    import cPickle


def generate_spans(seq, max_len=0):
    rtn = []
    for i in range(len(seq)):
        for j in range(len(seq)-i):
            span = seq[i:i+j+1]
            if max_len and len(span) <= max_len:
                rtn.append(span)
    return rtn


def create_input(data, parameters, add_label=True):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    inputs = utils.create_input(data, parameters, add_label)

    # inputs are sorted by sequence length in order to be processed by pytorch
    # lstm
    reversed_seq_index_mapping = {v: i for i, v in inputs['seq_index_mapping'].items()}
    sorted_span_index = [
        data[reversed_seq_index_mapping[i]]['span_index']
        for i in range(len(data))
        ]
    sorted_span_tag = [
        data[reversed_seq_index_mapping[i]]['span_tag']
        for i in range(len(data))
        ]
    sorted_span_entity_tag = [
        data[reversed_seq_index_mapping[i]]['span_entity_tag']
        for i in range(len(data))
        ]
    sorted_span_candidates = [
        data[reversed_seq_index_mapping[i]]['span_candidates']
        for i in range(len(data))
        ]

    batch_spans = [
        [span for j, span in enumerate(seq)]
        for i, seq in enumerate(sorted_span_index)
        ]
    batch_spans = list(itertools.chain.from_iterable(batch_spans))
    batch_span_len = [len(s) for s in batch_spans]
    # pad spans
    max_span_len = max(batch_span_len)
    padded_batch_spans = [list(s) + [0]*(max_span_len-len(s)) for s in batch_spans]

    batch_tags = [
        [sorted_span_tag[i][j] for j, span in enumerate(seq)]
        for i, seq in enumerate(sorted_span_index)
        ]
    batch_tags = list(itertools.chain.from_iterable(batch_tags))

    batch_entity_tags = [
        [sorted_span_entity_tag[i][j] for j, span in enumerate(seq)]
        for i, seq in enumerate(sorted_span_index)
        ]
    batch_entity_tags = list(itertools.chain.from_iterable(batch_entity_tags))

    batch_candidates = [
        [span for j, span in enumerate(seq)]
        for i, seq in enumerate(sorted_span_candidates)
        ]
    batch_candidates = list(itertools.chain.from_iterable(batch_candidates))
    batch_candidate_len = [len(s) for s in batch_candidates]
    # pad candidates
    max_candidate_len = max(batch_candidate_len)
    padded_batch_candidates = [
        list(c) + [0] * (max_candidate_len - len(c)) for c in batch_candidates
        ]

    span_position = [
        [[i, j] for j, span in enumerate(seq)]
        for i, seq in enumerate(sorted_span_index)
        ]
    span_position = list(itertools.chain.from_iterable(span_position))

    inputs['spans'] = np.array(padded_batch_spans)
    inputs['span_len'] = np.array(batch_span_len)
    inputs['span_tags'] = np.array(batch_tags)
    inputs['span_pos'] = np.array(span_position)
    inputs['span_entity_tags'] = np.array(batch_entity_tags)
    inputs['span_candidates'] = np.array(padded_batch_candidates)
    inputs['span_candidate_len'] = np.array(batch_candidate_len)

    return inputs


def process_preds(ner_prob, linking_prob, inputs, id_to_span_tag, tag_to_id, entity_to_id):
    span_preds = defaultdict(list)
    spans = inputs['spans']
    span_pos = inputs['span_pos']
    span_len = inputs['span_len']
    for i in range(len(ner_prob)):
        _, ner_pred = torch.max(ner_prob[i], dim=0)
        ner_pred = ner_pred[0]  # to int
        ner_pred_label = id_to_span_tag[ner_pred]
        ner_confidence = ner_prob[i][ner_pred]

        _, linking_pred = torch.max(linking_prob[i], dim=0)  # computes the pred index among candidates
        linking_pred = linking_pred[0]  # to int
        linking_confidence = linking_prob[i][linking_pred]
        linking_pred = inputs['span_candidates'][i][linking_pred]  # computes the acutal entity index in entity embeddings

        s_pos = span_pos[i]
        if ner_pred_label is not 'O':
            span_preds[s_pos[0]].append(
                (
                    set(spans[i][:span_len[i]].data),
                    ner_pred, ner_confidence,
                    linking_pred, linking_confidence
                )
            )

    # clean overlapped mentions, choose mentions with highest conf
    # score
    ner_pred_rtn = []
    linking_pred_rtn = []
    seq_len = inputs['seq_len']
    for j in range(len(seq_len)):
        span_pred = span_preds[j]

        # remove spans that exceed sequence length before padding
        span_pred = [s for s in span_pred if max(s[0]) < seq_len[j]]

        # sort span prediction by confidence
        sorted_span_pred = sorted(span_pred, key=lambda x: x[2], reverse=True)

        # choose span with top confidence and remove conflicted span
        conflict_table = defaultdict(set)
        for k, s in enumerate(sorted_span_pred):
            index, _, _, _, _ = s
            for l in index:
                conflict_table[l].add(k)
        unique_span_pred = []
        span_to_ignore = set()
        for k, s in enumerate(sorted_span_pred):
            index, _, _, _, _ = s
            if k not in span_to_ignore:
                unique_span_pred.append(s)
                for l in index:
                    span_to_ignore |= conflict_table[l]

        # convert span labels to bio labels
        p = dict()
        for s in unique_span_pred:
            index, ner_pred, _, linking_pred, __ = s
            for k, l in enumerate(sorted(index)):
                if len(index) == 1:
                    n_p = tag_to_id['S-' + id_to_span_tag[ner_pred]]
                elif k == 0:
                    n_p = tag_to_id['B-' + id_to_span_tag[ner_pred]]
                elif k == len(index) - 1:
                    n_p = tag_to_id['E-' + id_to_span_tag[ner_pred]]
                else:
                    n_p = tag_to_id['I-' + id_to_span_tag[ner_pred]]
                p[l] = (n_p, linking_pred)

        ner_pred_rtn.append(
            [p[k][0] if k in p else tag_to_id['O'] for k in range(seq_len[j])]
        )
        linking_pred_rtn.append(
            [p[k][1] if k in p else entity_to_id['<NIL>'] for k in range(seq_len[j])]
        )

    return ner_pred_rtn, linking_pred_rtn


def evaluate_linking(preds, dataset):
    num_perfect_mtn = 0  # number of linkable perfect mentions
    num_correct_pred = 0
    for i, sentence in enumerate(dataset):
        span_index = sentence['span_index']
        span_entity_tag = sentence['span_entity_tag']
        span_candidates = sentence['span_candidates']
        span_tag = sentence['span_tag']
        linkable_mtn = []
        for j, index in enumerate(span_index):
            if span_tag[j] != 0:
                # only evaluate non-nil entities
                if span_entity_tag[j] == 0:
                    continue
                entity_index = span_candidates[j][span_entity_tag[j]]
                linkable_mtn.append((index, entity_index))
                num_perfect_mtn += 1

        sentence_linking_pred = preds[i]
        for mtn in linkable_mtn:
            if sentence_linking_pred[mtn[0][0]] == mtn[1]:
                num_correct_pred += 1

    acc = num_correct_pred / num_perfect_mtn

    return num_perfect_mtn, num_correct_pred, acc
