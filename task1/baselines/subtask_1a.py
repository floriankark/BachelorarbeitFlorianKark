import jsonlines
import json
import random
import logging
import argparse
from os.path import join, dirname, basename
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os

import sys
sys.path.append('.')

from scorer.subtask_1 import evaluate
from format_checker.subtask_1 import check_format

random.seed(0)
ROOT_DIR = dirname(dirname(__file__))

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

def run_majority_baseline(data_dir, train_fpath, test_fpath, results_fpath):
    train_objs = [obj for obj in jsonlines.open(join(data_dir, train_fpath))]
    test_objs = [obj for obj in jsonlines.open(join(data_dir, test_fpath))]

    pipeline = DummyClassifier(strategy="most_frequent")
    pipeline.fit([obj["tweet_text"] for obj in train_objs], [obj["class_label"] for obj in train_objs])

    with open(results_fpath, "w") as results_file:
        predicted_distance = pipeline.predict([obj["tweet_text"] for obj in test_objs])
        results_file.write("tweet_id\tclass_label\trun_id\n")

        for i, line in enumerate(test_objs):
            label = predicted_distance[i]

            results_file.write("{}\t{}\t{}\n".format(line["tweet_id"], label, "majority"))


def run_random_baseline(data_dir, data_fpath, results_fpath):
    gold_objs = [obj for obj in jsonlines.open(join(data_dir, data_fpath))]
    #label_list= [obj["class_label"] for obj in gold_objs]
    label_list= ["Yes","No"]


    with open(results_fpath, "w") as results_file:
        results_file.write("tweet_id\tclass_label\trun_id\n")
        for i, line in enumerate(gold_objs):
            results_file.write('{}\t{}\t{}\n'.format(line["tweet_id"], random.choice(label_list), "random"))


def run_ngram_baseline(data_dir, train_fpath, test_fpath, results_fpath):
    train_text_lab = [[obj["tweet_id"], obj["tweet_text"], obj["class_label"]] for obj in jsonlines.open(join(data_dir, train_fpath))]
    test_text_lab = [[obj["tweet_id"], obj["tweet_text"], obj["class_label"]] for obj in jsonlines.open(join(data_dir, test_fpath))]

    pipeline = Pipeline([
        ('ngrams', TfidfVectorizer(ngram_range=(1, 1),lowercase=True,use_idf=True,max_df=0.95, min_df=3,max_features=5000)),
        ('clf', SVC(C=1, kernel='linear', random_state=0))
    ])
    pipeline.fit([obj[1] for obj in train_text_lab], [obj[2] for obj in train_text_lab])

    with open(results_fpath, "w") as results_file:
        predicted_distance = pipeline.predict([obj[1] for obj in test_text_lab])
        results_file.write("tweet_id\tclass_label\trun_id\n")

        for i, line in enumerate(test_text_lab):
            label = predicted_distance[i]
            results_file.write("{}\t{}\t{}\n".format(line[0], label, "ngram"))


def run_imgbert_baseline(data_dir, split, train_fpath, test_fpath, results_fpath):
    tr_feats = json.load(open(join(data_dir, "features", "train_feats.json")))
    te_feats = json.load(open(join(data_dir, "features", "%s_feats.json"%(split))))
    train_id_lab = [[obj["tweet_id"], obj["class_label"]] for obj in jsonlines.open(join(data_dir, train_fpath))]
    test_id_lab = [[obj["tweet_id"], obj["class_label"]] for obj in jsonlines.open(join(data_dir, test_fpath))]

    tr_cat_feats = [tr_feats["imgfeats"][obj[0]]+tr_feats["textfeats"][obj[0]] for obj in train_id_lab]
    tr_cat_feats = np.array(tr_cat_feats)
    te_cat_feats = [te_feats["imgfeats"][obj[0]]+te_feats["textfeats"][obj[0]] for obj in test_id_lab]
    te_cat_feats = np.array(te_cat_feats)

    clf = SVC(C=1, kernel='linear', random_state=0)
    clf.fit(tr_cat_feats, [obj[1] for obj in train_id_lab])

    with open(results_fpath, "w") as results_file:
        predicted_distance = clf.predict(te_cat_feats)
        results_file.write("tweet_id\tclass_label\trun_id\n")

        for i, line in enumerate(test_id_lab):
            label = predicted_distance[i]

            results_file.write("{}\t{}\t{}\n".format(line[0], label, "imgbert"))



def run_baselines(data_dir, test_split, train_fpath, test_fpath, lang):
    ## Write test file in format
    test_objs = [obj for obj in jsonlines.open(join(data_dir, test_fpath))]
    gold_fpath = join(data_dir, f'{basename(test_fpath)}')

    majority_baseline_fpath = join(data_dir,
                                 f'majority_baseline_{basename(test_fpath.replace("jsonl", "txt"))}')
    run_majority_baseline(data_dir, train_fpath, test_fpath, majority_baseline_fpath)
    if check_format(majority_baseline_fpath):
        acc, precision, recall, f1 = evaluate(gold_fpath, majority_baseline_fpath, subtask="A")
        logging.info(f"Majority Baseline for {lang} F1 (positive class): {f1}")

    random_baseline_fpath = join(data_dir, f'random_baseline_{basename(test_fpath.replace("jsonl", "txt"))}')
    run_random_baseline(data_dir, test_fpath, random_baseline_fpath)
    if check_format(random_baseline_fpath):
        acc, precision, recall, f1 = evaluate(gold_fpath, random_baseline_fpath, subtask="A")
        logging.info(f"Random Baseline for {lang} F1 (positive class): {f1}")

    ngram_baseline_fpath = join(data_dir, f'ngram_baseline_{basename(test_fpath.replace("jsonl", "txt"))}')
    run_ngram_baseline(data_dir, train_fpath, test_fpath, ngram_baseline_fpath)
    if check_format(ngram_baseline_fpath):
        acc, precision, recall, f1 = evaluate(gold_fpath, ngram_baseline_fpath, subtask="A")
        logging.info(f"Ngram Baseline for {lang} F1 (positive class): {f1}")

    imgbert_baseline_fpath = join(data_dir, f'imgbert_baseline_{basename(test_fpath.replace("jsonl", "txt"))}')
    run_imgbert_baseline(data_dir, test_split, train_fpath, test_fpath, imgbert_baseline_fpath)
    if check_format(imgbert_baseline_fpath):
        acc, precision, recall, f1 = evaluate(gold_fpath, imgbert_baseline_fpath, subtask="A")
        logging.info(f"ImgBert Baseline for {lang} F1 (positive class): {f1}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", "-d", required=True, type=str,
                        default="~/clef2023-checkthat-lab/task1/data/subtask-1A-english/",
                        help="The absolute path to the training data")
    parser.add_argument("--test-split", "-s", required=True, type=str,
                        default="dev", help="Test split name")
    parser.add_argument("--train-file-name", "-tr", required=True, type=str,
                        default="CT23_1A_checkworthy_multimodal_english_train.jsonl",
                        help="Training file name")
    parser.add_argument("--test-file-name", "-te", required=True, type=str,
                        default="CT23_1A_checkworthy_multimodal_english_dev.jsonl",
                        help="Test file name")
    parser.add_argument("--lang", "-l", required=True, type=str, default="english",
                        help="Options: arabic | english")
    args = parser.parse_args()
    run_baselines(args.data_dir, args.test_split, args.train_file_name, args.test_file_name, args.lang)