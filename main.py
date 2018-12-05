# -*- coding: utf-8 -*-

import os

import pandas as pd
from gensim.models.keyedvectors import KeyedVectors

import constants
from clustering import classifiers_en, sentence

os.chdir("/Users/esthergonzalez/TesisDecidim/code/data")

def load_proposal_data():
    return pd.read_csv(classifiers_en.CURRENT_CATEGORY_KEY[constants.PROPOSALS_RESULT_PATH], sep=",", header=0)


def load_pre_trained_english_corpus():
    word2vec_model = KeyedVectors.load_word2vec_format("data/word2vec/GoogleNews-vectors-negative300.bin", binary=True)
    word2vec_model.wv.vocab
    return word2vec_model


def save_proposals_corpus(proposals):
    text_lem = []
    text_orig = []

    for index, row in proposals.iterrows():
        lem_result = sentence.words_transformation(row["title_en"], row["body_en"])
        orig_result = row["title_en"] + " " + row["body_en"]
        text_lem.append(" ".join(lem_result))
        text_orig.append(orig_result)

    df = pd.DataFrame({'text': text_lem})
    df.to_csv(open(constants.LEM_PROPOSALS_PATH, 'w'), encoding='utf-8')

    df = pd.DataFrame({'text': text_orig})
    df.to_csv(open(constants.ORIGINAL_PROPOSALS_PATH, 'w'), encoding='utf-8')


def __get_by_subcategories(proposals):
    subcategories = constants.subcategories.keys()
    proposals = proposals.loc[proposals['Subcategory'].isin(subcategories)]
    for sub in subcategories:
        proposals.loc[proposals['Subcategory'] == sub, 'Subcategory'] = constants.subcategories[sub]
    return proposals


def __get_by_ids(proposals):
    ids = [5142, 4318, 2282, 5293, 3750, 2551, 1556, 2290, 6671, 8565, 3673, 7485, 5809, 8667, 2386, 40, 10442, 10441,
           10429, 0424, 107, 115, 9052, 117, 137, 138, 4983, 162, 2087, 7010, 219, 9070, 222, 6229, 6215, 245, 6215,
           249, 6140, 6741, 4219, 2552, 4930, 274, 2293, 2358, 375, 9152, 390, 391, 4333, 2450, 6701, 10310, 4348, 6704,
           6697, 479, 480, 2435, 502, 2510, 537, 5359, 5356, 5349, 5348, 593, 5113, 624, 6237, 7934, 6031, 708, 837,
           843, 4685, 1684, 8136, 8113, 8681, 10895, 8099, 4122, 8131, 4695, 3921, 10868, 10915, 6863, 792, 915, 7999,
           223, 7991, 10791, 10276, 7985, 998, 999, 162, 2357, 7423, 6752, 6021, 6022, 1052, 9836, 1089, 7199, 4116,
           2778, 10897, 10828, 10834, 6484, 538, 9199, 6767, 6768, 8133, 10566, 10875, 6765, 3927, 4702, 4387, 10872,
           7592, 4419, 5050, 10920, 8435, 3632, 3372, 3638, 2891, 8453, 1889, 2896, 6997, 7255, 4561, 8300, 3019, 7408,
           8304, 5893, 8786, 9906, 9791, 7566, 1866, 2042, 2145, 7564, 10792, 8854, 10795, 1862, 1906, 2592, 10788,
           2506, 8583, 1824, 2293, 2358, 2962, 3193, 5113, 6050, 6142, 6255, 6744, 8678, 10235, 9041, 2976, 2072, 2146,
           4976, 9553, 1990, 3738, 9221, 3053, 5099, 1918, 4749, 7047, 7044, 3210, 7374, 7064, 5383, 7073, 5810, 6710,
           7323, 2006, 9745, 5617, 10489, 7996, 6966, 6555, 3765, 3569, 4137, 9047, 8340, 6234, 4257, 1298, 5487, 10643]
    return proposals.loc[proposals['Proposal ID'].isin(ids)]


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    proposals = load_proposal_data()

    if classifiers_en.CURRENT_CATEGORY_NAME == 'ALL':
        proposals = __get_by_subcategories(proposals)
        proposals = __get_by_ids(proposals)
    else:
        proposals['size'] = None
        proposals['size'] = proposals.groupby('result_id').transform(len)
        proposals.drop(proposals.loc[proposals['size'] == 1].index, axis=0, inplace=True)
        proposals.drop('size', axis=1, inplace=True)


    proposals = proposals.drop_duplicates(subset="Proposal ID")

    # classifiers_en.compute_lsa_third_attempt(proposals)
    # classifiers_en.compute_tf_idf_similarity(proposals)

    word2vec_model = load_pre_trained_english_corpus()
    classifiers_en.compute_doc2vec(proposals, word2vec_model, use_tf=True)






    # classifiers_en.compute_lsa_second_attempt(proposals)
    #
    # classifiers_en.compute_lsa(proposals, unique_words)
    # proposals, unique_words = load_proposal_data()
    # save_proposals_corpus(proposals, unique_words)

    # sent2vec_model = load_pre_trained_sent2vec()
    # proposals, unique_words = load_proposal_data()
    # categories = words_stats.get_categories(proposals)
    # sentence2vec.compute_sent2vec(proposals, sent2vec_model, categories, unique_words)

    # duplicates_proposals = proposals[proposals.duplicated(subset="Proposal ID", keep=False)].copy()
    # # duplicates_proposals.groupby('Proposal ID').ngroup().add(1)
    # duplicates_proposals["class"] = duplicates_proposals.groupby('Proposal ID').ngroup().add(1)
    # for idx, row in duplicates_proposals.iterrows():
    #     proposals.loc[idx]["class"] = row["class"]