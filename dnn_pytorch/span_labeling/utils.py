import itertools
import torch
import numpy as np
from dnn_pytorch.seq_labeling import utils
from torch.autograd import Variable
from dnn_pytorch import LongTensor

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

    # inputs are sorted by sequence length.
    reversed_seq_index_mapping = {v: i for i, v in inputs['seq_index_mapping'].items()}
    sorted_span_index = [
        data[reversed_seq_index_mapping[i]]['span_index']
        for i in range(len(data))
        ]
    sorted_span_tag = [
        data[reversed_seq_index_mapping[i]]['span_tag']
        for i in range(len(data))
        ]
    sorted_span_kbid = [
        data[reversed_seq_index_mapping[i]]['span_kbid']
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

    span_position = [
        [[i, j] for j, span in enumerate(seq)]
        for i, seq in enumerate(sorted_span_index)
        ]
    span_position = list(itertools.chain.from_iterable(span_position))

    inputs['spans'] = Variable(torch.from_numpy(np.array(padded_batch_spans))).type(LongTensor)
    inputs['span_len'] = batch_span_len
    inputs['span_tags'] = Variable(torch.from_numpy(np.array(batch_tags))).type(LongTensor)
    inputs['span_pos'] = span_position

    # sort spans by length in order to process in batches. need to keep index
    # spans_to_sort = [
    #     [(span, (i, j), sorted_span_tag[i][j]) for j, span in enumerate(seq)]
    #     for i, seq in enumerate(sorted_span_index)
    #     ]
    # spans_to_sort = list(itertools.chain.from_iterable(spans_to_sort))
    #
    # sorted_spans = sorted(spans_to_sort, key=lambda x: len(x[0]))
    #
    # # group spans by span length
    # sorted_span_groups = []
    # group = []
    # group_len = 0
    # for i, s in enumerate(sorted_spans):
    #     if not group:
    #         group.append(s)
    #         group_len = len(s[0])
    #     elif len(s[0]) != group_len:
    #         sorted_span_groups.append(group)
    #         group = [s]
    #         group_len = len(s[0])
    #     elif len(s[0]) == group_len:
    #         group.append(s)
    #     if i == len(sorted_spans) - 1 and group:
    #         sorted_span_groups.append(group)
    #         group = []
    #
    # inputs['span_groups'] = sorted_span_groups

    # inputs['span_index'] = [
    #     data[reversed_seq_index_mapping[i]]['span_index']
    #     for i in range(len(data))
    #     ]
    # inputs['span_tag'] = [
    #     data[reversed_seq_index_mapping[i]]['span_tag']
    #     for i in range(len(data))
    #     ]

    return inputs
