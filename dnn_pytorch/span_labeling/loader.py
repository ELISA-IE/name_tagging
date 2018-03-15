import json
import os
import urllib.request
import itertools
import numpy as np

import sys
sys.path.append('/data/m1/panx2/code/edl/')
# import edl.linker
# from edl.models.text import EntityMention, NominalMention, Entity

from dnn_pytorch.seq_labeling import loader
from dnn_pytorch.span_labeling.utils import generate_spans


def load_spans(sentences, tag_scheme, is_train=True):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    #
    # generate all possible spans for each sentence
    #
    max_span_len = 5
    spans = []
    for sent in sentences:
        span_meta = {}  # to store meta data about spans

        words = [w[0] for w in sent]
        # generate all possible spans
        sent_spans = generate_spans(words, max_len=max_span_len)
        span_index = generate_spans(
            np.arange(len(words)), max_len=max_span_len
        )
        span_meta['sent_spans'] = sent_spans
        span_meta['span_index'] = span_index

        #
        # if in training phase, pre-process the tagging and linking labels
        #
        if is_train:
            ner_label = [w[-1] for w in sent]
            kbid_label = [w[-2] for w in sent]

            span_ner_label = generate_spans(ner_label, max_len=max_span_len)
            span_kbid_label = generate_spans(kbid_label, max_len=max_span_len)

            updated_spans = []
            updated_span_index = []
            updated_span_ner_label = []
            updated_span_kbid_label = []
            for i, ner_label in enumerate(span_ner_label):
                s_ner_label = 'O'
                if tag_scheme == 'iobes':
                    if len(ner_label) == 1 and ner_label[0].startswith('S'):
                        s_ner_label = ner_label[0][2:]
                    elif ner_label[0].startswith('B') and ner_label[-1].startswith('E'):
                        if len(ner_label) == 2:
                            s_ner_label = ner_label[0][2:]
                        elif all([item.startswith('I') for item in ner_label[1:-1]]):
                            s_ner_label = ner_label[0][2:]
                elif tag_scheme == 'iob':
                    # todo
                    pass
                if s_ner_label != 'O':
                    assert all([span_kbid_label[i][j] == span_kbid_label[i][0] for j in
                                range(len(span_kbid_label[i]))])
                    s_kbid_label = span_kbid_label[i][0]
                else:
                    s_kbid_label = '<O>'

                # only keep perfect mention spans when evaluating linking
                # if s_ner_label == 'O':
                #     continue

                updated_spans.append(sent_spans[i])
                updated_span_index.append(span_index[i])
                updated_span_ner_label.append(s_ner_label)
                updated_span_kbid_label.append(s_kbid_label)

            sent_spans = updated_spans
            span_ner_label = updated_span_ner_label
            span_kbid_label = updated_span_kbid_label
            span_index = updated_span_index

            assert len(sent_spans) == len(span_ner_label) == len(span_kbid_label)

            span_meta.update({
                'sent_spans': sent_spans,
                'span_index': span_index,
                'span_ner_label': span_ner_label,
                'span_kbid_label': span_kbid_label
            })

        # generate span linking candidates using elisa api
        span_candidates = []
        span_candidate_conf = []
        span_candidate_type = []
        for i, s in enumerate(sent_spans):
            span_text = ' '.join(s)

            # candidates, candidate_conf, candidate_type = linking(span_text, 10)

            # following two lines generate all 0 candidates for debugging purpose.
            candidates = ['<NIL>'] * 10
            candidate_conf = [0] * 10
            candidate_type = ['<O>'] * 10

            # '<O>' means the span is not a mention, '<NIL>' means the span is a
            # mention but it does not have a kb id. ignore the entity candidate if
            # it's not in entity_to_id. 'o_conf' and 'nil_conf' will be learned
            # from training data
            candidates += ['<O>', '<NIL>']
            candidate_conf += [0, 0]
            candidate_type += ['<O>', '<NIL>']

            span_candidates.append(candidates)
            span_candidate_conf.append(candidate_conf)
            span_candidate_type.append(candidate_type)

        # to store various types of meta data about spans
        span_meta.update({
            'span_candidates': span_candidates,
            'span_candidate_type': span_candidate_type,
            'span_candidate_conf': span_candidate_conf
        })

        spans.append(span_meta)

    return spans


def prepare_sentence(sentence, span, feat_column,
                     word_to_id, char_to_id, tag_to_id, span_tag_to_id,
                     feat_to_id_list, entity_to_id, etype_to_id,
                     lower=False, is_train=True):
    """
    Prepare a sentence for evaluation.
    """

    def f(x): return x.lower() if lower else x

    rtn = loader.prepare_sentence(
        sentence, feat_column,
        word_to_id, char_to_id, tag_to_id, feat_to_id_list,
        lower, is_train
    )

    gold_kbid = [entity_to_id[w[-2]] for w in sentence]
    rtn['gold_kbid'] = gold_kbid

    span_index = span['span_index']

    #
    # only keep candidates that are in KB
    #
    span_candidates = []
    span_candidate_conf = []
    span_candidate_type = []
    for i, c in enumerate(span['span_candidates']):
        candidate = []
        candidate_conf = []
        candidate_type = []
        for j, e in enumerate(c):
            if e not in entity_to_id:
                continue
            candidate.append(entity_to_id[e])
            candidate_conf.append(span['span_candidate_conf'][i][j])
            candidate_type.append(etype_to_id[span['span_candidate_type'][i][j]])
        span_candidates.append(candidate)
        span_candidate_conf.append(candidate_conf)
        span_candidate_type.append(candidate_type)

    if is_train:
        span_tag = [span_tag_to_id[s] for s in span['span_ner_label']]
        span_kbid_label = [entity_to_id[k] for k in span['span_kbid_label']]
        span_gold_candidate = []

        # find out gold candidate index in the span candidates
        for i in range(len(span_index)):
            if span_kbid_label[i] in span_candidates[i]:
                candidate_index = span_candidates[i].index(span_kbid_label[i])
                span_gold_candidate.append(candidate_index)
            # add gold to candidate if it's not in the retrieved candidates
            else:
                span_candidates[i] = [span_kbid_label[i]] + span_candidates[i]
                span_candidate_conf[i] = [1] + span_candidate_conf[i]
                span_gold_candidate.append(0)
            # set gold candidate to <NIL> if the gold candidate is not in the
            # retrieved candidates
            # else:
            #     candidate_index = span_candidates[i].index(entity_to_id['<NIL>'])
            #     span_gold_candidate.append(candidate_index)

            assert span_gold_candidate[i] < len(span_candidates[i])

        rtn['span_tag'] = span_tag
        rtn['span_gold_candidate'] = span_gold_candidate

    rtn['span_index'] = span_index
    rtn['span_candidates'] = span_candidates
    rtn['span_candidate_conf'] = span_candidate_conf
    rtn['span_candidate_type'] = span_candidate_type

    return rtn


def prepare_dataset(sentences, spans, feat_column,
                    word_to_id, char_to_id, tag_to_id, span_tag_to_id,
                    feat_to_id_list, entity_to_id, etype_to_id,
                    lower=False, is_train=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    data = []
    for sent, span in zip(sentences, spans):
        data.append(
            prepare_sentence(sent, span, feat_column,
                             word_to_id, char_to_id, tag_to_id, span_tag_to_id,
                             feat_to_id_list, entity_to_id, etype_to_id,
                             lower, is_train=is_train)
        )

    return data


def span_tag_mapping(spans):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    ner_tags = itertools.chain.from_iterable([s['span_ner_label'] for s in spans])
    dico = create_dico(list(ner_tags))
    tag_to_id, id_to_tag = loader.create_mapping(dico)
    print("Found %i unique span named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def etype_mapping(spans):
    etypes = itertools.chain.from_iterable([s['span_candidate_type'] for s in spans])
    dico = loader.create_dico(list(etypes))
    etype_to_id, id_to_etype = loader.create_mapping(dico)
    print("Found %i unique candidate entity types" % len(dico))
    return dico, etype_to_id, id_to_etype


def entity_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    entities = [[word[-2] for word in s] for s in sentences]
    dico = loader.create_dico(entities)
    entity_to_id, id_to_entity = loader.create_mapping(dico)
    print("Found %i unique entities" % len(dico))
    return dico, entity_to_id, id_to_entity


def remove_o_tags(inputs, span_tag_to_id):
    rtn = []
    for seq in inputs:
        span_index = seq['span_index']
        span_tag = seq['span_tag']

        non_o_span_index = []
        non_o_span_tag = []
        for i in range(len(span_index)):
            if span_tag[i] != span_tag_to_id['O']:
                non_o_span_index.append(span_index[i])
                non_o_span_tag.append(span_tag[i])

        seq['span_index'] = non_o_span_index
        seq['span_tag'] = non_o_span_tag

        rtn.append(seq)

    return rtn


def linking_api(query):
    url = 'http://blender02.cs.rpi.edu:3301/linking?mention=%s&lang=%s' % (
        urllib.request.quote(query), 'en')
    linking_result = urllib.request.urlopen(url, timeout=5).read()
    linking_result = json.loads(linking_result)

    candidates = [item['kbid'] for item in linking_result['results']]

    candidate_conf = [item['confidence'] for item in linking_result['results']]

    return candidates, candidate_conf


def linking(query, n):
    em = EntityMention(query)
    edl.linker.add_candidate_entities(em, lang='en', n=n)
    edl.linker.rank_candidate_entities(em, etype=None)

    if em.candidates:
        candidates, candidate_conf = zip(*[[entity.kbid, entity.confidence] for entity in em.candidates])
        candidate_type = [edl.linker.get_etype(e) for e in candidates]
    else:
        candidates, candidate_conf, candidate_type = [], [], []

    for i, t in enumerate(candidate_type):
        if t is None:
            candidate_type[i] = 'None'

    return list(candidates), list(candidate_conf), list(candidate_type)


def augment_with_entity_pretrained(dico_entity, entity_emb):
    print('Augmenting entity mapping table with pretrained entity embeddings.')
    assert os.path.isfile(entity_emb)

    # Load pretrained embeddings from file
    pretrained = []
    if entity_emb:
        for line in open(entity_emb):
            if not line.strip():
                continue
            entity = line.split()[0].strip()
            pretrained.append(entity)

    pretrained = set(pretrained)
    for e in pretrained:
        if e not in dico_entity:
            dico_entity[e] = 0

    entity_to_id, id_to_entity = loader.create_mapping(dico_entity)

    return dico_entity, entity_to_id, id_to_entity


# def prepare_sentence(sentence, feat_column,
#                      word_to_id, char_to_id, tag_to_id, span_tag_to_id,
#                      feat_to_id_list, entity_to_id, etype_to_id,
#                      lower=False, is_train=True, tag_scheme='iob'):
#     """
#     Prepare a sentence for evaluation.
#     """
#     rtn = loader.prepare_sentence(
#         sentence, feat_column,
#         word_to_id, char_to_id, tag_to_id, feat_to_id_list,
#         lower, is_train
#     )
#
#     # generate all possible spans
#     max_span_len = 10
#     sent_len = len(rtn['words'])
#     span_index = generate_spans(np.arange(sent_len), max_len=max_span_len)
#     entity = [w[-2] for w in sentence[:sent_len]]
#     span_entity = generate_spans(entity, max_len=max_span_len)
#
#     # label tags at span level
#     span_tag = []
#     if is_train:
#         raw_tag = [w[-1] for w in sentence[:sent_len]]
#         span_tag = generate_spans(raw_tag, max_len=max_span_len)
#         updated_span_tag = []
#         updated_span_entity = []
#         updated_span_index = []
#         for i, s in enumerate(span_tag):
#             label = 'O'
#             if tag_scheme == 'iobes':
#                 if len(s) == 1 and s[0].startswith('S'):
#                     label = s[0][2:]
#                 elif s[0].startswith('B') and s[-1].startswith('E'):
#                     if len(s) == 2:
#                         label = s[0][2:]
#                     elif all([item.startswith('I') for item in s[1:-1]]):
#                         label = s[0][2:]
#             elif tag_scheme == 'iob':
#                 # todo
#                 pass
#             if label != 'O':
#                 assert all([span_entity[i][j] == span_entity[i][0] for j in range(len(span_entity[i]))])
#                 entity = span_entity[i][0]
#             else:
#                 entity = '<O>'
#
#             # only keep perfect mention spans when evaluating linking
#             if label == 'O':
#                 continue
#
#             updated_span_index.append(span_index[i])
#             updated_span_entity.append(entity_to_id[entity])
#             updated_span_tag.append(span_tag_to_id[label])
#
#         span_tag = updated_span_tag
#         span_index = updated_span_index
#         span_entity = updated_span_entity
#
#         assert len(span_tag) == len(span_index) == len(span_entity)
#
#     # generate span entity candidates using elisa api
#     span_candidates = []
#     span_candidate_conf = []
#     span_candidate_type = []
#     span_gold_candidate = []
#     for i, s in enumerate(span_index):
#         span_text = ' '.join([sentence[index][0] for index in s])
#
#         candidates, candidate_conf, candidate_type = linking(span_text, 10)
#
#         # following two lines generate all 0 candidates for the testing purpose.
#         # candidates = ['<NIL>'] * 10
#         # candidate_conf = [0] * 10
#
#         # '<O>' means the span is not a mention, '<NIL>' means the span is a
#         # mention but it does not have a kb id. ignore the entity candidate if
#         # it's not in entity_to_id. 'o_conf' and 'nil_conf' will be learned
#         # from training data
#         padded_candidates = []
#         for j, item in enumerate(candidates):
#             if item in entity_to_id:
#                 etype_id = etype_to_id[candidate_type[j]] if candidate_type[j] in etype_to_id \
#                     else etype_to_id['OTHER']
#                 padded_candidates.append(
#                     (entity_to_id[item], candidate_conf[j], etype_id)
#                 )
#         padded_candidates += [
#             (entity_to_id['<O>'], 'o_conf', etype_id['OTHER']),
#             (entity_to_id['<NIL>'], 'nil_conf', etype_id['OTHER'])
#         ]
#
#         candidates_id, candidate_conf = zip(*padded_candidates)
#         candidates_id = list(candidates_id)
#         candidate_conf = list(candidate_conf)
#         candidate_type = list(candidate_type)
#
#         if is_train:
#             if span_entity[i] in candidates_id:
#                 candidate_index = candidates_id.index(span_entity[i])
#                 span_gold_candidate.append(candidate_index)
#             # add gold to candidate if it's not in the retrieved candidates
#             # else:
#             #     candidates_id = [span_entity[i]] + candidates_id
#             #     candidate_conf = [1] + candidate_conf
#             #     span_gold_candidate.append(0)
#             # set gold candidate to <NIL> if the gold candidate is not in the
#             # retrieved candidates
#             else:
#                 candidate_index = candidates_id.index(entity_to_id['<NIL>'])
#                 span_gold_candidate.append(candidate_index)
#
#         assert span_gold_candidate[i] < len(candidates_id)
#
#         span_candidates.append(candidates_id)
#         span_candidate_conf.append(candidate_conf)
#         span_candidate_type.append(candidate_type)
#
#     # regard span candidate type as one of span features
#     span_features = [[etype] for etype in span_candidate_type]
#
#     rtn['span_index'] = span_index
#     rtn['span_tag'] = span_tag
#     rtn['span_gold_candidate'] = span_gold_candidate
#     rtn['span_candidates'] = span_candidates
#     rtn['span_candidate_conf'] = span_candidate_conf
#     rtn['span_features'] = span_features
#
#     return rtn


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for item in item_list:
        if item not in dico:
            dico[item] = 1
        else:
            dico[item] += 1

    return dico