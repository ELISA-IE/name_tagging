import json
import os
import functools
import numpy as np
import urllib.request
import requests

import sys
sys.path.append('/data/m1/panx2/code/edl/')
import edl.linker
from edl.models.text import EntityMention, NominalMention, Entity

from dnn_pytorch.seq_labeling import loader
from dnn_pytorch.span_labeling.utils import generate_spans


def prepare_sentence(sentence, feat_column,
                     word_to_id, char_to_id, tag_to_id, span_tag_to_id,
                     feat_to_id_list, entity_to_id,
                     lower=False, is_train=True, tag_scheme='iob'):
    """
    Prepare a sentence for evaluation.
    """
    rtn = loader.prepare_sentence(
        sentence, feat_column,
        word_to_id, char_to_id, tag_to_id, feat_to_id_list,
        lower, is_train
    )

    # generate all possible spans
    max_span_len = 10
    sent_len = len(rtn['words'])
    span_index = generate_spans(np.arange(sent_len), max_len=max_span_len)
    entity = [w[-2] for w in sentence[:sent_len]]
    span_entity = generate_spans(entity, max_len=max_span_len)

    # label tags at span level
    span_tag = []
    if is_train:
        raw_tag = [w[-1] for w in sentence[:sent_len]]
        span_tag = generate_spans(raw_tag, max_len=max_span_len)
        updated_span_tag = []
        updated_span_entity = []
        updated_span_index = []
        for i, s in enumerate(span_tag):
            label = 'O'
            if tag_scheme == 'iobes':
                if len(s) == 1 and s[0].startswith('S'):
                    label = s[0][2:]
                elif s[0].startswith('B') and s[-1].startswith('E'):
                    if len(s) == 2:
                        label = s[0][2:]
                    elif all([item.startswith('I') for item in s[1:-1]]):
                        label = s[0][2:]
            elif tag_scheme == 'iob':
                # todo
                pass
            if label != 'O':
                assert all([span_entity[i][j] == span_entity[i][0] for j in range(len(span_entity[i]))])
                entity = span_entity[i][0]
            else:
                entity = '<NIL>'

            # only keep perfect mention spans when evaluating linking
            # if label == 'O':
            #     continue

            updated_span_index.append(span_index[i])
            updated_span_entity.append(entity_to_id[entity])
            updated_span_tag.append(span_tag_to_id[label])

        span_tag = updated_span_tag
        span_index = updated_span_index
        span_entity = updated_span_entity

        assert len(span_tag) == len(span_index) == len(span_entity)

    # generate span entity candidates using elisa api
    span_candidates = []
    span_candidate_conf = []
    span_entity_tag = []
    for i, s in enumerate(span_index):
        span_text = ' '.join([sentence[index][0] for index in s])

        candidates, candidate_conf = linking(span_text, 10)

        # following two lines generate all 0 candidates for the testing purpose.
        # candidates = ['<NIL>'] * 10
        # candidate_conf = [0] * 10

        candidates_id = [entity_to_id[item] if item in entity_to_id else 0 for
                         item in candidates]

        if is_train:
            if span_entity[i] in candidates_id:
                entity_tag = candidates_id.index(span_entity[i])
                span_entity_tag.append(entity_tag)
            else:
                candidates_id = [span_entity[i]] + candidates_id
                candidate_conf = [1] + candidate_conf
                span_entity_tag.append(0)

        span_candidates.append(candidates_id)
        span_candidate_conf.append(candidate_conf)

    rtn['span_index'] = span_index
    rtn['span_tag'] = span_tag
    rtn['span_entity_tag'] = span_entity_tag
    rtn['span_candidates'] = span_candidates
    rtn['span_candidate_conf'] = span_candidate_conf

    return rtn


def prepare_dataset(sentences, feat_column,
                    word_to_id, char_to_id, tag_to_id, span_tag_to_id,
                    feat_to_id_list, entity_to_id,
                    lower=False, is_train=True, tag_scheme='iob'):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    data = []
    for i, s in enumerate(sentences):
        data.append(
            prepare_sentence(s, feat_column,
                             word_to_id, char_to_id, tag_to_id, span_tag_to_id,
                             feat_to_id_list, entity_to_id,
                             lower, is_train=is_train, tag_scheme=tag_scheme)
        )
        # requests = sum([len(item['span_index']) for item in data])
        # print(requests)
    return data


def span_tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    # remove BIOES tags and remain types for span labeling
    tags = [[t[2:] if t != 'O' else t for t in sent] for sent in tags]
    dico = loader.create_dico(tags)
    tag_to_id, id_to_tag = loader.create_mapping(dico)
    print("Found %i unique span named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def entity_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    entities = [[word[-2] for word in s] for s in sentences]
    dico = loader.create_dico(entities)
    dico['<NIL>'] = 1000000000
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

    candidates_conf = [item['confidence'] for item in linking_result['results']]

    return candidates, candidates_conf


@functools.lru_cache(maxsize=None)
def linking(query, n):
    em = EntityMention(query)
    edl.linker.add_candidate_entities(em, lang='en', n=n)
    edl.linker.rank_candidate_entities(em, etype=None)

    if em.candidates:
        candidates, candidates_conf = zip(*[[entity.kbid, entity.confidence] for entity in em.candidates])
    else:
        candidates, candidates_conf = (), ()

    return list(candidates), list(candidates_conf)


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
    print(len(entity_to_id))
    return dico_entity, entity_to_id, id_to_entity

