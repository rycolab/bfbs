import sacrebleu, numpy as np
import pandas as pd
from mosestokenizer import MosesDetokenizer
import sys
import glob
import collections
file_prefix = sys.argv[1]
reference = sys.argv[2]
language = sys.argv[3]

def file_to_lines(f):
    return open(f).read().splitlines()

def detokenize(f):
    with MosesDetokenizer(language) as detokenize:
        lines = file_to_lines(f)
        return [detokenize(s.split()) for s in lines]

def load(f):
    return list(open(f))

def scores(refs, f):
    fs = detokenize(f)
    assert len(fs) <= len(refs)
    return np.array([sacrebleu.sentence_bleu(out, [ref]).score
                     for out, ref in zip(fs, refs[:len(fs)])])

def build_dataframe():
    refs = file_to_lines(reference)
    data = collections.defaultdict(dict)
    files = glob.glob(file_prefix+'[0-9].txt')
    files.extend(glob.glob(file_prefix+'[0-9][0-9].txt'))
    if not files:
        print("No files matching pattern found")
        exit(0)
    for i, filename in enumerate(files, 1):
        for sen_idx, score in enumerate(scores(refs, filename)):
            data[sen_idx][i] = score
    pd.DataFrame(list(data.values())).to_csv(file_prefix+'bleu.csv', index=False)

build_dataframe()