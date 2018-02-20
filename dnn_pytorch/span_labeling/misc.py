from dnn_pytorch.span_labeling.loader import linking


def compute_linking_candidate_coverage(bio_linking_file):
    print('=> retrieving mentions...')
    mentions = extract_mentions(bio_linking_file)

    print('=> retrieving linking candidates...')
    num_hit = 0
    num_top1_hit = 0
    num_nil = 0
    top_n_candidate = 50
    for m in mentions:
        query = ' '.join([item[0] for item in m])

        # skip NIL mention
        kb_id = m[0][1]
        if kb_id == "NIL" or kb_id.startswith('NIL'):
            num_nil += 1
            continue

        candidates, conf = linking(query, top_n_candidate)

        if kb_id in candidates:
            num_hit += 1
            if kb_id == candidates[0]:
                num_top1_hit += 1

    acc = num_hit / (len(mentions) - num_nil)
    top1_acc = num_top1_hit / (len(mentions) - num_nil)

    print('retrieve top %d candidates.' % top_n_candidate)
    print('%d mentions found in the file, %d are NIL.' % (len(mentions), num_nil))
    print('%d linking candidates are retrieved for each mention.' % top_n_candidate)
    print('%d mentions have ground truth in its candidates.' % num_hit)
    print('%d mentions have ground truth in its top 1 candidate.' % num_top1_hit)
    print('the accuracy is %.4f' % acc)
    print('the top 1 accuracy is %.4f' % top1_acc)


def extract_mentions(bio_linking_file):
    mentions = []
    for sent in open(bio_linking_file).read().split("\n\n"):
        # generate mentions for each sequence
        words = sent.splitlines()
        sent_mentions = []
        current_mention = []
        mention_count = 0
        for i, line in enumerate(words):
            elements = line.split()
            token = elements[0]
            kb_id = elements[-2]
            label = elements[-1]

            # parse mention in bioes scheme
            if label == 'O' or label.startswith('B'):
                if current_mention:
                    sent_mentions.append(current_mention)
                if label.startswith('B'):
                    current_mention = [(token, kb_id, label)]
                    mention_count += 1
                else:
                    current_mention = []
            elif label.startswith('I'):
                current_mention.append((token, kb_id, label))

            if i == len(words) - 1:
                if current_mention:
                    sent_mentions.append(current_mention)

        try:
            assert mention_count == len(sent_mentions)
        except:
            print(words[0])

        mentions += sent_mentions

    return mentions


if __name__ == "__main__":
    # train_file = '/nas/data/m1/zhangb8/ml/pytorch/example/data/eng.train.linking.bio'
    # dev_file = '/nas/data/m1/zhangb8/ml/pytorch/example/data/eng.testa.linking.bio'
    # test_file = '/nas/data/m1/zhangb8/ml/pytorch/example/data/eng.testb.linking.bio'

    train_file = '/nas/data/m1/zhangb8/kbp_edl/data/edl17/trilingual/eng/train/edl15.train+eval.nam.bio'
    dev_file = '/nas/data/m1/zhangb8/kbp_edl/data/edl17/trilingual/eng/dev/edl16.eval.nam.bio'
    test_file = '/nas/data/m1/zhangb8/kbp_edl/data/edl17/trilingual/eng/test/edl17.eval.nam.bio'

    compute_linking_candidate_coverage(train_file)
    compute_linking_candidate_coverage(dev_file)
    compute_linking_candidate_coverage(test_file)



