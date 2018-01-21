import numpy as np
from dnn_pytorch.seq_labeling import loader
from dnn_pytorch.span_labeling.utils import generate_spans


def prepare_sentence(sentence, feat_column,
                     word_to_id, char_to_id, tag_to_id, span_tag_to_id,
                     feat_to_id_list, kbid_to_id,
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
    raw_tag = [w[-1] for w in sentence[:sent_len]]
    span_tag = generate_spans(raw_tag, max_len=max_span_len)
    kbid = [w[1] for w in sentence[:sent_len]]
    span_kbid = generate_spans(kbid, max_len=max_span_len)

    # label tags at span level
    updated_span_tag = []
    updated_span_kbid = []
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
            assert all([span_kbid[i][j] == span_kbid[i][0] for j in range(len(span_kbid[i]))])
            kbid = span_kbid[i][0]
        else:
            kbid = 'NIL'
        updated_span_kbid.append(kbid_to_id[kbid])
        updated_span_tag.append(span_tag_to_id[label])
    span_tag = updated_span_tag

    assert len(span_tag) == len(span_index)

    rtn['span_index'] = span_index
    rtn['span_tag'] = span_tag
    rtn['span_kbid'] = updated_span_kbid

    return rtn


def prepare_dataset(sentences, feat_column,
                    word_to_id, char_to_id, tag_to_id, span_tag_to_id,
                    feat_to_id_list, kbid_to_id,
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
                             feat_to_id_list, kbid_to_id,
                             lower, is_train=is_train, tag_scheme='iobes')
        )

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


def kbid_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    kbids = [[word[1] for word in s] for s in sentences]
    dico = loader.create_dico(kbids)
    kbid_to_id, id_to_kbid = loader.create_mapping(dico)
    print("Found %i unique kb ids" % len(dico))
    return dico, kbid_to_id, id_to_kbid


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