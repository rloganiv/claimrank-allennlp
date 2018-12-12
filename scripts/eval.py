"""
Self-contained evaluation script
"""
import argparse
from collections import Counter
import copy
from functools import reduce
import logging
from math import exp, log
import re
import subprocess
from statistics import mean


STOPWORDS = {
    'the', 'and', 'for','that', 'you', 'your', 'but', 'with', 'my', 'this',
    'that', 'those', 'these', 'they', 'she', 'have', 'had',
    'am','are','aren','be','been','being','can','can','could','couldn','did','didn','do','does','doesn','doing',
    'done','don',
    'get','gets','getting','got','had','hadn','has','hasn','have','haven','having',
    'he','is','isn','it','may','might','must',"n't", "-lrb-","-rrb-"
    'mustn','ought','oughtn','shall', 'she','should','shouldn', 'that',
    'was','wasn','we','were', 'weren','will','won','would','wouldn',"'s",
    'a','an','from', '&', 'at', 'in', 'our', 'as', 'to', 'or','of', 'by', 'my',
    'mine','your','yours', 'her', 'hers', 'his', 'their', 'our', '(',
    ')','[',']','{','}',',','.'
}


def ngram_count(words, n):
    if n <= len(words):
        return Counter(zip(*[words[i:] for i in range(n)]))
    return Counter()


def max_count(c1, c2):
    keys = set(c1.keys() + c2.keys())
    return Counter({k: max(c1[k], c2[k]) for k in keys})


def min_count(c1, c2):
    return Counter({k: min(c1[k], c2[k]) for k in c1})


def closest_min_length(candidate, references):
    l0 = len(candidate)
    return min((abs(len(r) - l0), len(r)) for r in references)[1]


def safe_log(n):
    if n <= 0:
        return -9999999999
    return log(n)


def tokenize(txt):
    return txt.strip().split()


def precision_n(candidate, references, n, use_cand_tot=False):
    ref_max = reduce(max_count, [ngram_count(ref, n) for ref in references])
    candidate_ngram_count = ngram_count(candidate, n)

    if use_cand_tot:
        total = sum(ref_max.values())
    else:
        total = sum(candidate_ngram_count.values())

    correct = sum(reduce(min_count, (ref_max, candidate_ngram_count)).values())
    score = (correct / total) if total else 0
    return score, correct, total


def multi_bleu(candidates, all_references, tokenize_fn=tokenize, maxn=4,
               filter_func=None, use_bp=True, compute_for_input=False):
    correct = [0] * maxn
    total = [0] * maxn
    cand_tot_length = 0
    ref_closest_length = 0

    # If compute_for_input, use_bp!=compute_for_input should be true
    # Else doesn't really matter
    assert not compute_for_input or use_bp!=compute_for_input , \
            "Using brevity penalty while computing scores for input"

    # for candidate, references in zip(candidates, zip(*all_references)):
    for candidate, references in zip(candidates, all_references):

        if not isinstance(candidate, list):
            candidate = tokenize_fn(candidate)
        if not isinstance(references[0], list):
            # references = map(tokenize_fn, references)
            references = [tokenize_fn(r) for r in references]

        cand_tot_length += len(candidate)
        ref_closest_length += closest_min_length(candidate, references)

        for n in range(maxn):
            if n == 0 and filter_func:
                candidate_filtered = filter_func(candidate)
                references_filtered = map(filter_func, references)
                sc, cor, tot = precision_n(candidate_filtered,
                                           references_filtered, n + 1,
                                           use_cand_tot=True if
                                           compute_for_input else False)
            else:
                sc, cor, tot = precision_n(candidate,
                                           references, n + 1,
                                           use_cand_tot=True if
                                           compute_for_input else False)
            correct[n] += cor
            total[n] += tot

    precisions = [(correct[n] / total[n]) if correct[n] else 0 for n in range(maxn)]

    if cand_tot_length < ref_closest_length:
        brevity_penalty = exp(1 - ref_closest_length / cand_tot_length)
    else:
        brevity_penalty = 1
    if use_bp:
        score = 100 * brevity_penalty * exp(
                        sum(safe_log(precisions[n]) for n in range(maxn)) / maxn)
    else:
        score = 100 * exp(
            sum(safe_log(precisions[n]) for n in range(maxn)) / maxn)
    prec_pc = [100 * p for p in precisions]
    if len(prec_pc) < 4:
       prec_pc.extend([0.0] * (4-len(prec_pc)))
    return score, prec_pc, brevity_penalty, cand_tot_length, ref_closest_length


def compute_meteor(prediction_path, target_path, meteor_path):
    cmd_template = 'java -Xmx2G -jar {meteor_path} "{prediction_path}" "{target_path}"'
    cmd = cmd_template.format(prediction_path=prediction_path,
                              target_path=target_path,
                              meteor_path=meteor_path)
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    output = output.decode('utf-8')

    score_re = re.compile(r'(?P<ignore>Final score:\s+)(?P<score>\d\.\d+)')
    score_string = score_re.search(output).group('score')
    if score_string:
        return 100 * float(score_string)
    else:
        raise RuntimeError('No METEOR score detected')


def compute_prf(predictions, targets, average_method='micro'):

    assert average_method in ['micro', 'macro'], 'Bad average_method'

    tp = Counter()
    fp = Counter()
    fn = Counter()

    if average_method == 'macro':
        all_words = set()

    for prediction, target in zip(predictions, targets):

        # Tokenize (and filter stopwords)
        prediction_tokens = [x for x in tokenize(prediction) if x not in STOPWORDS]
        target_tokens = [x for x in tokenize(target) if x not in STOPWORDS]


        # Get counts
        prediction_count = Counter(prediction_tokens)
        target_count = Counter(target_tokens)

        # Get sets
        intersection = set(prediction_tokens).intersection(set(target_tokens))
        prediction_only = set(prediction_tokens).difference(set(target_tokens))
        target_only = set(target_tokens).difference(set(prediction_tokens))

        # Update corpus level stats
        for token in intersection:
            tp[token] += min(prediction_count[token], target_count[token])
            fn[token] += max(0, target_count[token] - prediction_count[token])
        for token in prediction_only:
            fp[token] += prediction_count[token]
        for token in target_only:
            fn[token] += target_count[token]

        # Update global token set
        if average_method == 'macro':
            all_words.update(prediction_tokens)
            all_words.update(target_tokens)

    if average_method == 'micro':
        precision = sum(tp.values())/(sum(tp.values()) + sum(fp.values()))
        recall = sum(tp.values())/(sum(tp.values()) + sum(fn.values()))
        f1 = 2 * precision * recall / (precision + recall)

    if average_method == 'macro':
        precisions = []
        recalls = []
        f1s = []

        for word in all_words:

            p_word = tp[word] / (tp[word] + fp[word] + 1e-13)
            r_word = tp[word] / (tp[word] + fn[word] + 1e-13)
            f1_word = 2 * p_word * r_word / (p_word + r_word + 1e-13)

            precisions.append(p_word)
            recalls.append(r_word)
            f1s.append(f1_word)

        precision = mean(precisions)
        recall = mean(recalls)
        f1 = mean(f1s)

    return 100 * precision, 100 * recall, 100 * f1


def main(_):

    if args.verbose:
        print('Evaluating: %s' % args.predictions)

    # Load data into Python
    with open(args.predictions, 'r') as prediction_file, \
         open(args.targets, 'r') as target_file:

        predictions = [line.strip() for line in prediction_file]
        targets = [line.strip() for line in target_file]

    # Sanity check
    assert len(predictions) == len(targets), 'Length mismatch'

    # Compute precision, recall and F1 (without stopwords)
    precision, recall, f1 = compute_prf(predictions, targets,
                                        average_method=args.average_method)

    # Compute BLEU score
    # Stupid hack: multi_bleu expects list of references
    _targets = [[x] for x in targets]
    bleu_score, *_ = multi_bleu(predictions, _targets)

    # Compute METEOR score
    # Note: this processes prediction files directly
    meteor_score = compute_meteor(args.predictions,
                                  args.targets,
                                  args.meteor_path)

    if args.verbose:
        fstring = '\nP: %0.2f, R: %0.2f, F1: %0.2f, BLEU: %0.2f, METEOR: %0.2f\n'
    else:
        fstring = '%0.2f & %0.2f & %0.2f & %0.2f & %0.2f \\\\'

    print(fstring % (precision, recall, f1, bleu_score, meteor_score))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PoMo evaluation script.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('predictions', type=str,
                        help='Text file containing a prediction on each line')
    parser.add_argument('targets', type=str,
                        help='Text file containing the corresponding targets')
    parser.add_argument('--meteor_path', type=str, required=True,
                        help='Path to meteor-1.5.jar')
    parser.add_argument('--average_method', type=str, default='micro',
                        help='Average used for precision, recall, and f1')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='If enabled, prints verbose output instead of '
                             'LaTeX table.')
    args, _ = parser.parse_known_args()

    main(_)

