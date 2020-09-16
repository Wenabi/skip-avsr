from os import path


def compute_wer(predictions_dict, ground_truth_dict, split_words=False):
    wer = 0
    err_dict = {}
    for fname, prediction in predictions_dict.items():
        prediction = _strip_extra_chars(prediction)
        ground_truth = _strip_extra_chars(ground_truth_dict[fname])

        if split_words is True:
            prediction = ''.join(prediction).split()
            ground_truth = ''.join(ground_truth).split()

        er = levenshtein(ground_truth, prediction) / float(len(ground_truth))
        wer += er
        err_dict[fname] = er

    return wer / (float(len(predictions_dict)) or 1), err_dict


def levenshtein(ground_truth, prediction):
    r"""
    Calculates the Levenshtein distance between ground_truth and prediction.
    """
    n, m = len(ground_truth), len(prediction)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        ground_truth, prediction = prediction, ground_truth
        n, m = m, n

    current = list(range(n+1))
    for i in range(1, m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1, n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if ground_truth[j - 1] != prediction[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


def _strip_extra_chars(prediction):
    return [value for value in prediction if value not in ('EOS', 'END', 'MASK')]


def write_sequences_to_labelfile(sequence_dict, fname, original_dict, error_dict):
    items = []
    for (k, v) in sequence_dict.items():
        label_str = ''.join(_strip_extra_chars(v))
        truth = ''.join(_strip_extra_chars(original_dict[k]))
        items.append(' '.join([k, label_str, '[{}] [{:.3f}]'.format(truth, error_dict[k])]) + '\n')

    with open(fname, 'w') as f:
        f.writelines(items)

    del items


def get_files(file_list, dataset_dir, remove_sa=True, shuffle_sentences=False):
    with open(file_list, 'r') as f:
        contents = f.read().splitlines()

    contents = [path.join(dataset_dir, line.split()[0]) for line in contents]

    if remove_sa is True:
        contents = [line for line in contents if '/sa' not in line]

    if shuffle_sentences is True:
        from random import shuffle
        shuffle(contents)

    return contents


def compute_measures(predictions_dict, ground_truth_dict, split_words=False):
    p, n = 0, 0
    if len(predictions_dict.items()) > 0:
        for fname, prediction in predictions_dict.items():
            prediction = _strip_extra_chars(prediction)
            ground_truth = _strip_extra_chars(ground_truth_dict[fname])
            
            if split_words is True:
                prediction = ''.join(prediction).split()
                ground_truth = ''.join(ground_truth).split()
            # print("prediction", prediction)
            # print("ground_truth", ground_truth)
            # print("F1-Score", f1_score(ground_truth, prediction, average='micro'))
            intersection = 0
            p_b = 0
            n_b = 0
            for u in set(ground_truth):
                # Indices
                ri = [i for i, e in enumerate(ground_truth) if e == u]
                ai = [i for i, e in enumerate(prediction) if e == u]
                
                intersection += len([e for e in ai if e in ri])
                p_b += len(ri)
                n_b += len(ai)
            p += intersection / p_b if p_b > 0 else 0
            n += intersection / n_b if n_b > 0 else 0
            # print("MyF1-score", (2*(intersection/p_b)*(intersection/n_b))/((intersection/p_b)+(intersection/n_b)))
        p = p / len(predictions_dict.items())
        n = n / len(predictions_dict.items())
        f = (2 * p * n) / (p + n) if (p + n) > 0 else 0
    else:
        p, n, f = 0, 0, 0
    return p, n, f