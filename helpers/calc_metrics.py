import os
import pandas as pd
from rouge import Rouge
import nltk
import random
import sys

def calc_legacy_rouge(refs, hyps, directory="eval"):
    from pyrouge import Rouge155
    r = Rouge155()
    system_dir = os.path.join(directory, 'hyp')
    model_dir = os.path.join(directory, 'ref')
    if not os.path.isdir(system_dir):
        os.makedirs(system_dir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    r.system_dir = system_dir
    r.model_dir = model_dir
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_filename_pattern = '#ID#_reference.txt'
    for i, (ref, hyp) in enumerate(zip(refs, hyps)):
        hyp_file_path = os.path.join(r.system_dir, "%06d_decoded.txt" % i)
        with open(hyp_file_path, "w") as w:
            hyp_sentences = hyp.split(" s_s ")
            w.write("\n".join(hyp_sentences))
        ref_file_path = os.path.join(r.model_dir, "%06d_reference.txt" % i)
        with open(ref_file_path, "w") as w:
            ref_sentences = ref.split(" s_s ")
            w.write("\n".join(ref_sentences))
    output = r.convert_and_evaluate()
    result = r.output_to_dict(output)
    log_str = ""
    for x in ["1","2","l"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x,y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = result[key]
            val_cb = result[key_cb]
            val_ce = result[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    return log_str


def calc_metrics(refs, hyps):
    print("Count:", len(hyps))
    print('Text:', data['text'].iloc[-1])
    print("Ref:", refs[-1])
    print("Hyp:", hyps[-1])
    

    from nltk.translate.bleu_score import corpus_bleu
    print("BLEU: ", corpus_bleu([[r] if r is not list else r for r in refs], hyps))
    rouge = Rouge()
    scores = rouge.get_scores(hyps, refs, avg=True)
    print("ROUGE: ", scores)



path = sys.argv[1]
data = pd.read_csv(path, sep='\t', names=['text', 'pred', 'true'])
print(data.shape)
data = data.dropna()
print('Result:\n')
refs = list(data['true'])
hyps = list(data['pred'])

print(calc_metrics(refs, hyps))

