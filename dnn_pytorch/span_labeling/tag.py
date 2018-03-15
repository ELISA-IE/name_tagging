import argparse
import time
import torch

from dnn_pytorch.seq_labeling.utils import iobes_iob
from dnn_pytorch.seq_labeling.loader import load_sentences, update_tag_scheme

from dnn_pytorch.span_labeling.nn import SpanLabeling
from dnn_pytorch.span_labeling.loader import prepare_dataset, load_spans
from dnn_pytorch.span_labeling.utils import create_input, process_preds


# Read parameters from command line
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="",
    help="Model location"
)
parser.add_argument(
    "--input", default="",
    help="Input bio file location"
)
parser.add_argument(
    "--output", default="",
    help="Output bio file location"
)
parser.add_argument(
    "--batch_size", default="50",
    type=int, help="batch size"
)
parser.add_argument(
    "--gpu", default="0",
    type=int, help="default is 0. set 1 to use gpu."
)
args = parser.parse_args()

print('loading model from:', args.model)
if args.gpu:
    state = torch.load(args.model)
else:
    state = torch.load(args.model, map_location=lambda storage, loc: storage)

parameters = state['parameters']
mappings = state['mappings']

# Load reverse mappings
word_to_id, char_to_id, tag_to_id, span_tag_to_id, entity_to_id, etype_to_id = [
    {v: k for k, v in x.items()}
    for x in [mappings['id_to_word'], mappings['id_to_char'],
              mappings['id_to_tag'], mappings['id_to_span_tag'],
              mappings['id_to_entity'], mappings['id_to_etype']]
    ]
feat_to_id_list = [
    {v: k for k, v in id_to_feat.items()}
    for id_to_feat in mappings['id_to_feat_list']
    ]

# eval sentences
eval_sentences = load_sentences(
    args.input,
    parameters['lower'],
    parameters['zeros']
)

update_tag_scheme(eval_sentences, parameters['tag_scheme'])

eval_spans = load_spans(eval_sentences, parameters['tag_scheme'])

eval_dataset = prepare_dataset(
    eval_sentences, eval_spans, parameters['feat_column'],
    word_to_id, char_to_id, tag_to_id, span_tag_to_id, feat_to_id_list,
    entity_to_id, etype_to_id,
    parameters['lower'], is_train=False
)

print("%i sentences in eval set." % len(eval_dataset))

# initialize model
model = SpanLabeling(parameters)
model.load_state_dict(state['state_dict'])
model.train(False)

since = time.time()
batch_size = args.batch_size
f_output = open(args.output, 'w')

# Iterate over data.
print('tagging...')
num_instances = 0
for i in range(0, len(eval_dataset), batch_size):
    inputs = create_input(eval_dataset[i:i+batch_size], parameters, add_label=False)

    # forward
    ner_prob, linking_prob, loss, tagging_loss, linking_loss = model.forward(inputs)

    seq_index_mapping = inputs['seq_index_mapping']

    raw_ner_preds, raw_linking_preds = process_preds(
        ner_prob, linking_prob, inputs, mappings['id_to_span_tag'],
        tag_to_id, entity_to_id
    )
    ner_preds = [raw_ner_preds[seq_index_mapping[j]]
                 for j in range(len(seq_index_mapping))]

    linking_preds = [raw_linking_preds[seq_index_mapping[j]]
                     for j in range(len(seq_index_mapping))]

    for j, (d, p, q) in enumerate(
            zip(eval_dataset[i:i+batch_size], ner_preds, linking_preds)
    ):
        assert len(d['words']) == len(p) == len(q)
        p_tags = [mappings['id_to_tag'][item] for item in p]
        q_tags = [mappings['id_to_entity'][item] for item in q]
        if parameters['tag_scheme'] == 'iobes':
            p_tags = iobes_iob(p_tags)

        # Write tags
        f_output.write('%s\n\n' % '\n'.join('%s %s %s' % (' '.join(w), y, z)
                                            for w, y, z in
                                            zip(eval_sentences[i + j],
                                                q_tags, p_tags)))

        if (i + j + 1) % 500 == 0:
            print(i+j+1)

end = time.time()  # epoch end time
print('time elapssed: %f seconds' % round(
    (end - since), 2))

