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

    # sort data by sequence length
    seq_index_mapping, data = zip(
        *[item for item in sorted(
            enumerate(data), key=lambda x: len(x[1]['words']), reverse=True
        )]
    )

    batch_spans = list(itertools.chain.from_iterable([s['span_index'] for s in data]))

    batch_span_len = [len(s) for s in batch_spans]
    # pad spans
    max_span_len = max(batch_span_len) if batch_span_len else 0
    padded_batch_spans = [list(s) + [0]*(max_span_len-len(s)) for s in batch_spans]

    batch_candidates = list(
        itertools.chain.from_iterable([s['span_candidates'] for s in data]))
    batch_candidate_len = [len(s) for s in batch_candidates]
    # pad candidates
    max_candidate_len = max(batch_candidate_len) if batch_candidate_len else 0
    padded_batch_candidates = [
        list(c) + [0] * (max_candidate_len - len(c)) for c in batch_candidates
        ]

    batch_candidate_type = list(
        itertools.chain.from_iterable([s['span_candidate_type'] for s in data]))
    # pad candidates type
    padded_batch_candidates_type = [
        list(c) + [0] * (max_candidate_len - len(c)) for c in
        batch_candidate_type
        ]

    batch_candidate_conf = list(
        itertools.chain.from_iterable([s['span_candidate_conf'] for s in data]))
    # pad candidates type
    padded_batch_candidates_conf = [
        list(c) + [0] * (max_candidate_len - len(c)) for c in
        batch_candidate_conf
        ]

    span_seq_index = [
        [i] * len(seq) for i, seq in enumerate([s['span_index'] for s in data])
        ]
    span_seq_index = list(itertools.chain.from_iterable(span_seq_index))

    # add label for training phase
    if add_label:
        batch_tags = list(itertools.chain.from_iterable([s['span_tag'] for s in data]))
        batch_gold_candidates = list(itertools.chain.from_iterable([s['span_gold_candidate'] for s in data]))
        inputs['span_tags'] = np.array(batch_tags)
        inputs['span_gold_candidates'] = np.array(batch_gold_candidates)

    inputs['spans'] = np.array(padded_batch_spans)
    inputs['span_len'] = np.array(batch_span_len)
    inputs['span_seq_index'] = np.array(span_seq_index)
    inputs['span_candidates'] = np.array(padded_batch_candidates)
    inputs['span_candidate_len'] = np.array(batch_candidate_len)
    inputs['span_candidate_type'] = np.array(padded_batch_candidates_type)
    inputs['span_candidate_conf'] = np.array(padded_batch_candidates_conf)

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

    covered_answer = 0
    gold_answer = 0
    span_tags = inputs['span_tags']
    for i, pred in enumerate(span_preds):
        for j, p in enumerate(pred):
            if span_tags[i][j] != 0 and p == span_tags[j]:
                covered_answer += 1
            if span_tags[i][j] != 0:
                gold_answer += 1

    # clean overlapped mentions, choose mentions with highest conf
    # score
    ner_pred_rtn = []
    linking_pred_rtn = []
    seq_len = inputs['seq_len']
    hit_answer = 0

    for j in range(len(seq_len)):
        span_pred = span_preds[j]

        # remove spans that exceed sequence length before padding
        span_pred = [s for s in span_pred if max(s[0]) < seq_len[j]]

        # sort span prediction by confidence
        sorted_span_pred = sorted(span_pred, key=lambda x: x[2], reverse=True)

        #
        # choose span with top confidence and remove conflicted span
        #
        # build index conflict table
        conflict_table = defaultdict(set)
        for k, s in enumerate(sorted_span_pred):
            index, _, _, _, _ = s
            for l in index:
                conflict_table[l].add(k)

        # choose spans with higher probability
        pred_by_token = dict()
        span_to_ignore = set()
        for k, s in enumerate(sorted_span_pred):
            index, _, _, _, _ = s
            if k in span_to_ignore:
                continue
            # convert span labels to bio labels
            try:
                index, ner_pred, _, linking_pred, __ = s

                if span_tags[j][k] != 0 and ner_pred == span_tags[j][k]:
                    hit_answer += 1

                for k, l in enumerate(sorted(index)):
                    if len(index) == 1:
                        n_p = tag_to_id['S-' + id_to_span_tag[ner_pred]]
                    elif k == 0:
                        n_p = tag_to_id['B-' + id_to_span_tag[ner_pred]]
                    elif k == len(index) - 1:
                        n_p = tag_to_id['E-' + id_to_span_tag[ner_pred]]
                    else:
                        n_p = tag_to_id['I-' + id_to_span_tag[ner_pred]]
                    pred_by_token[l] = (n_p, linking_pred)
                # ignore others spans that have index conflicts
                for l in index:
                    span_to_ignore |= conflict_table[l]
            # sometimes with very few training data, bioes tags are not all
            # covered, ignore the span if predicted tags are not in training
            # set.
            except KeyError:
                continue

        ner_pred_rtn.append(
            [pred_by_token[k][0] if k in pred_by_token else tag_to_id['O']
             for k in range(seq_len[j])]
        )
        linking_pred_rtn.append(
            [pred_by_token[k][1] if k in pred_by_token else entity_to_id['<O>']
             for k in range(seq_len[j])]
        )

    print('gold_answer', gold_answer)
    print('covered_answer', covered_answer)
    print('hit_answer', hit_answer)

    return ner_pred_rtn, linking_pred_rtn


def process_all_o_pred(inputs, tag_to_id, entity_to_id):
    ner_pred_rtn = []
    linking_pred_rtn = []
    seq_len = inputs['seq_len']
    for j in range(len(seq_len)):
        ner_pred_rtn.append([tag_to_id['O']] * seq_len[j])
        linking_pred_rtn.append([entity_to_id['<O>']] * seq_len[j])

    return ner_pred_rtn, linking_pred_rtn


def evaluate_linking(preds, dataset, id_to_entity):
    num_perfect_mtn = 0  # number of linkable perfect mentions
    num_correct_pred = 0
    predicted_linking_bio = []
    for i, sentence in enumerate(dataset):
        span_index = sentence['span_index']
        span_gold_candidate = sentence['span_gold_candidate']
        span_candidates = sentence['span_candidates']
        span_tag = sentence['span_tag']
        linkable_mtn = []
        for j, index in enumerate(span_index):
            if span_tag[j] != 0:
                # only evaluate non-<O> entities, <NIL> entities are included.
                if span_candidates[j][span_gold_candidate[j]] == 0:
                    continue
                entity_index = span_candidates[j][span_gold_candidate[j]]
                linkable_mtn.append((index, entity_index))
                num_perfect_mtn += 1

        sentence_linking_pred = preds[i]
        gold_kbid = sentence['gold_kbid']
        for mtn in linkable_mtn:
            kbid_pred = sentence_linking_pred[mtn[0][0]]
            gld_kbid = gold_kbid[mtn[0][0]]
            if kbid_pred == gld_kbid:
                num_correct_pred += 1

        # output linking results as bio
        sent_out = [' '.join((w, id_to_entity[gold_kbid[j]], id_to_entity[sentence_linking_pred[j]]))
                    for j, w in enumerate(sentence['str_words'])]
        predicted_linking_bio.append('\n'.join(sent_out))

    acc = num_correct_pred / num_perfect_mtn

    return num_perfect_mtn, num_correct_pred, acc, '\n\n'.join(predicted_linking_bio)


def evaluate_edl(ner_preds, linking_preds,
                 metric, parameters, dataset, phase, id_to_tag, id_to_entity):
    # evaluate tagging
    epoch_f1, epoch_acc, predicted_ner_bio = utils.evaluate_ner(
        parameters, ner_preds, dataset[phase], id_to_tag
    )
    if metric == 'f1':
        metric_score = epoch_f1
    elif metric == 'acc':
        metric_score = epoch_acc
    print(
        '{} F1: {:.4f} Acc: {:.4f}\n'.format(phase, epoch_f1, epoch_acc)
    )

    # evaluate linking
    num_perfect_mtn, num_correct_pred, acc, predicted_linking_bio = evaluate_linking(
        linking_preds, dataset[phase], id_to_entity)
    print('=> linking results:')
    print('%d perfect mentions are in the dataset.' % num_perfect_mtn)
    print('%d mentions are linked correctly.' % num_correct_pred)
    print('linking accuracy: %.4f' % (num_correct_pred / num_perfect_mtn))

    # combine ner and linking bio results
    edl_results = []
    ner_lines = predicted_ner_bio.splitlines()
    linking_lines = predicted_linking_bio.splitlines()

    assert len(ner_lines) == len(linking_lines)

    for i, line in enumerate(linking_lines):
        if not line:
            edl_results.append(line)

        linking_elements = line.split()
        ner_elements = ner_lines[i].split()
        new_line_elements = linking_elements + ner_elements[-2:]
        edl_results.append(' '.join(new_line_elements))

    predicted_bio = '\n'.join(edl_results)

    return metric_score, predicted_bio
