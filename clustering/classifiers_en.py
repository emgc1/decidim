# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 12:54:50 2018
@author: Esther Gonzalez
"""

import cPickle as pickle
import os

import numpy as np
import pandas as pd
from scipy.io import mmwrite
from scipy.spatial.distance import pdist
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

import constants
import metrics_util as m_util
from clustering import sentence

# Set dir
os.chdir("/Users/esthergonzalez/TesisDecidim")
path_to_files = "data_en"
model_paths = "final/models/"


METRIC = 'cosine'
TRUTH_COLUMN = "result_id"
# CURRENT_CATEGORY_NAME = constants.JUSTICIA_GLOBAL
CURRENT_CATEGORY_NAME = 'ALL'
# CURRENT_CATEGORY_KEY = constants.CATEGORIES[CURRENT_CATEGORY_NAME]
CURRENT_CATEGORY_KEY = constants.CATEGORIES[CURRENT_CATEGORY_NAME]
CURRENT_CATEGORY = CURRENT_CATEGORY_KEY[constants.ENGLISH]


def word2vec_mean(words, words_model):
    if words:
        word2vec_model = words_model
        words_in_corpus = [word for word in words if word in word2vec_model.wv.vocab]
        if words_in_corpus:
            return np.mean(word2vec_model[words_in_corpus], axis=0)


def drop_highly_correlated_features(doc2vec):
    df = pd.DataFrame(doc2vec)
    # Create correlation matrix
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.93)]
    df = df.drop(df.columns[to_drop], axis=1)
    return df.as_matrix()


def build_tf_idf(proposals, max_df, min_df, file_name=None, vect_file_name=None):
    texts = []
    plot_labels = []
    for index, row in proposals.iterrows():
        text = ' '.join(sentence.words_transformation(row[constants.TITLE_COLUMN], row[constants.BODY_COLUMN]))
        texts.append(text)
        plot_labels.append("{} - {}".format(row['Proposal ID'], row['result_id']))
    text_df = pd.DataFrame({'text': texts})
    tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words='english', use_idf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_df.text)
    if file_name and vect_file_name:
        mmwrite(file_name, tfidf_matrix)
        with open(vect_file_name, 'w') as fout:
            pickle.dump(tfidf_vectorizer, fout)
    return tfidf_matrix, tfidf_vectorizer


def compute_doc2vec(proposals, words_model, use_tf=False):
    doc2vec = []
    labels = []
    ground_truth = []

    _, tfidf_vectorizer = build_tf_idf(proposals, 0.5, 2)
    for index, row in proposals.iterrows():
        text = sentence.words_transformation(row[constants.TITLE_COLUMN], row[constants.BODY_COLUMN])
        if use_tf:
            text = [word for word in text if word in tfidf_vectorizer.get_feature_names()]
        ca_mean = word2vec_mean(text, words_model)
        if type(ca_mean) is np.ndarray:
            doc2vec.append(ca_mean)
            if CURRENT_CATEGORY_NAME == 'ALL':
                labels.append("{} - {}".format(row['Subcategory'], row['Proposal ID']))
            else:
                labels.append("{} - {}".format(row['Proposal ID'], row['result_id']))
            if CURRENT_CATEGORY_NAME == 'ALL':
                ground_truth.append(row['Subcategory'])
            else:
                ground_truth.append(row[TRUTH_COLUMN])
    distances = pdist(doc2vec, METRIC)
    m_util.compute_scores(distances, labels, ground_truth, "Word2Vec", CURRENT_CATEGORY, METRIC)


def compute_tf_idf_similarity(proposals):
    if METRIC == 'euclidean':
        return
    tfidf_matrix, _ = build_tf_idf(proposals, 0.9, 2)

    if CURRENT_CATEGORY_NAME == 'ALL':
        proposals['labels'] = proposals.apply(lambda row: "{} - {}".format(row['Proposal ID'], row['Subcategory']), axis=1)
        plot_labels = proposals['labels'].values
        labels_true = proposals["Subcategory"].values
    else:
        proposals['labels'] = proposals.apply(lambda row: "{} - {}".format(row['Proposal ID'], row['result_id']), axis=1)
        labels_true = proposals["result_id"].values
        plot_labels = proposals['labels'].values
    dist_matrix = pdist(np.asarray(tfidf_matrix.toarray()), metric=METRIC)
    m_util.compute_scores(dist_matrix, plot_labels, labels_true, "Tf-Idf", CURRENT_CATEGORY, METRIC)


def compute_lsa_third_attempt(proposals):
    if METRIC == 'euclidean':
        return
    max_df = 0.9
    min_df = 1
    k=5
    tfidf_matrix, _ = build_tf_idf(proposals, max_df, min_df)
    lsa = make_pipeline(TruncatedSVD(k), Normalizer(copy=False))
    lsa_matrix = lsa.fit_transform(tfidf_matrix)

    if CURRENT_CATEGORY_NAME == 'ALL':
        proposals['labels'] = proposals.apply(lambda row: "{} - {}".format(row['Proposal ID'], row['Subcategory']), axis=1)
        plot_labels = proposals['labels'].values
        labels_true = proposals["Subcategory"].values
    else:
        proposals['labels'] = proposals.apply(lambda row: "{} - {}".format(row['Proposal ID'], row['result_id']), axis=1)
        labels_true = proposals["result_id"].values
        plot_labels = proposals['labels'].values
    dist_matrix = pdist(lsa_matrix, metric=METRIC)
    m_util.compute_scores(dist_matrix, plot_labels, labels_true, "LSA", CURRENT_CATEGORY, METRIC)