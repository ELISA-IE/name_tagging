import argparse
import time
import torch

from ml_pytorch.seq_labeling.nn import SeqLabeling
from ml_pytorch.seq_labeling.utils import create_input, iobes_iob
from ml_pytorch.seq_labeling.loader import prepare_dataset, load_sentences

# external features
from ml_pytorch.seq_labeling.generate_features import generate_features


# Read parameters from command line
parser = argparse.ArgumentParser()

parser.add_argument(
    "-m", "--model", default="",
    help="Model location"
)
parser.add_argument(
    "-i", "--input", default="",
    help="Input bio file location"
)
parser.add_argument(
    "-o", "--output", default="",
    help="Output bio file location"
)

args = parser.parse_args()

print('loading model from:', args.model)
state = torch.load(args.model)

parameters = state['parameters']
mappings = state['mappings']

# Load reverse mappings
word_to_id, char_to_id, tag_to_id = [
    {v: k for k, v in x.items()}
    for x in [mappings['id_to_word'], mappings['id_to_char'], mappings['id_to_tag']]
    ]
feat_to_id_list = [
    {v: k for k, v in id_to_feat.items()}
    for id_to_feat in mappings['id_to_feat_list']
    ]

# eval sentences
eval_sentences = load_sentences(args.input,
                                parameters['lower'],
                                parameters['zeros'])

eval_feats, eval_stem = generate_features(eval_sentences, parameters)

eval_dataset = prepare_dataset(
    eval_sentences,
    eval_feats, eval_stem,
    word_to_id, char_to_id, tag_to_id, feat_to_id_list, parameters['lower']
)

print("%i sentences in eval set." % len(eval_dataset))

# initialize model
model = SeqLabeling(word_vocab_size=len(word_to_id), **parameters)
model.load_state_dict(state['state_dict'])
model.train(False)

since = time.time()
batch_size = 100
f_output = open(args.output, 'w')

# Iterate over data.
print('tagging...')
for i in range(0, len(eval_dataset), batch_size):
    inputs, index_mapping, batch_len = create_input(eval_dataset[i:i+batch_size], parameters)

    # forward
    outputs, loss = model.forward(inputs, batch_len)
    if parameters['crf']:
        preds = [outputs[index_mapping[j]].data
                 for j in range(len(outputs))]
    else:
        _, _preds = torch.max(outputs.data, 2)

        preds = [
            _preds[index_mapping[j]][:batch_len[index_mapping[j]]]
            for j in range(len(index_mapping))
            ]
    for j, pred in enumerate(preds):
        pred = [mappings['id_to_tag'][p] for p in pred]
        # Output tags in the IOB2 format
        if parameters['tag_scheme'] == 'iobes':
            pred = iobes_iob(pred)
        # Write tags
        assert len(pred) == len(eval_sentences[i+j])
        f_output.write('%s\n\n' % '\n'.join('%s%s%s' % (' '.join(w), ' ', z)
                                            for w, z in zip(eval_sentences[i+j],
                                                            pred)))
        if (i + j + 1) % 500 == 0:
            print(i+j+1)

end = time.time()  # epoch end time
print('time elapssed: %f seconds' % round(
    (end - since), 2))

