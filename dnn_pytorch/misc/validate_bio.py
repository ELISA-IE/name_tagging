import argparse


def validate_bio(bio_path):
    o_i_error = 0
    for sent in open(bio_path).read().split('\n\n'):
        sent = sent.strip()
        if not sent:
            continue

        tokens = [t.split() for t in sent.splitlines()]
        for i, t in enumerate(tokens):
            sys_label = t[-1]
            if sys_label.startswith('I'):
                if i == 0 or tokens[i-1][-1] == 'O':
                    o_i_error += 1

    print('O-I errors,', o_i_error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('bio_path')
    args = parser.parse_args()

    validate_bio(args.bio_path)