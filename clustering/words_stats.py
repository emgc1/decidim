from collections import Counter
import numpy as np
import os
import pandas as pd
import nltk
import stop_words as st
from string import punctuation
from gensim.models.keyedvectors import KeyedVectors
import matplotlib.pyplot as plt
from clustering.classifiers_en import build_tf_idf
from clustering.sentence import words_transformation
import constants
COLUMNS = ['title en', 'body en']
model_paths = "final/models/"


def get_words_histogram(proposals, word2vec_model):
    max_df = 0.5
    min_df = 2
    df = pd.DataFrame(columns=['tfidf', 'word2vec', 'tfidf_word2vec', 'not_word2vec'])
    file_name = model_paths+"/words_count.csv"
    _, tfidf_vectorizer = build_tf_idf(proposals, max_df, min_df)
    for index, row in proposals.iterrows():
        try:
            text_tfidf = []
            text_word2vec = []
            text_tfidf_word2vec = []
            not_text_word2vec = []
            text = words_transformation(row[constants.TITLE_COLUMN], row[constants.BODY_COLUMN])
            for word in text:
                in_word2vec = False
                in_tfidf = False
                if word in word2vec_model.wv.vocab:
                    in_word2vec = True
                    text_word2vec.append(word)
                if word in tfidf_vectorizer.get_feature_names():
                    in_tfidf = True
                    text_tfidf.append(word)
                if in_tfidf and in_word2vec:
                    text_tfidf_word2vec.append(word)
                if not in_word2vec:
                    not_text_word2vec.append(word)
            df = df.append({'tfidf': ' '.join(text_tfidf),
                            'word2vec': ' '.join(text_word2vec),
                            'tfidf_word2vec': ' '.join(text_tfidf_word2vec),
                            'not_word2vec': ' '.join(not_text_word2vec)}, ignore_index=True)
        except:
            full_result_out = open(file_name, 'w')
            df.to_csv(full_result_out, encoding='utf-8')

        df['tfidf_count'] = df.tfidf.apply(lambda x: len(x.split(' ')) if len(x.split(' ')) > 1 or len(x.split(' ')[0]) > 0 else 0)
        df['tfidf_word2vec_count'] = df.tfidf_word2vec.apply(lambda x: len(x.split(' ')) if len(x.split(' ')) > 1 or len(x.split(' ')[0]) > 0 else 0)
        df['word2vec_count'] = df.word2vec.apply(lambda x: len(x.split(' ')) if len(x.split(' ')) > 1 or len(x.split(' ')[0]) > 0 else 0)
        df['not_word2vec_count'] = df.not_word2vec.apply(lambda x: len(x.split(' ')) if len(x.split(' ')) > 1 or len(x.split(' ')[0]) > 0 else 0)

        full_result_out = open(file_name, 'w')
        df.to_csv(full_result_out, encoding='utf-8')

    fig=plt.figure()
    df.hist(column="word2vec_count", bins=70)
    plt.title('Length of proposals in Word2Vec Model')
    plt.xlabel("Quantity of Words")
    plt.ylabel("Quantity of Proposals")
    plt.savefig(model_paths+"/word2vec_words.png")
    plt.close()

    df.hist(column="tfidf_count", bins=70)
    plt.title('Length of Proposals in TF-IDF')
    plt.xlabel("Quantity of Words")
    plt.ylabel("Quantity of Proposals")
    plt.savefig(model_paths+"/tfidf_words.png")
    plt.close()

    df.hist(column="tfidf_word2vec_count", bins=70)
    plt.title('Length of Proposals in Word2Vec Model')
    plt.xlabel("Quantity of Words")
    plt.ylabel("Quantity of Proposals")
    plt.savefig(model_paths+"/tfidf_word2vec_words.png")
    plt.close()

    df.hist(column="not_word2vec_count", bins=25)
    plt.title('Words not in Word2Vec Model')
    plt.xlabel("Quantity of Words")
    plt.ylabel("Quantity of Proposals")
    plt.savefig(model_paths+"/not_word2vec_words.png")
    plt.close()


def load_data(path):
    return pd.read_csv(path, sep=",", header=0)


def build_word_list(df, columns):
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = st.get_stop_words('english')
    # stop_words_es = st.get_stop_words('spanish')
    en_stop_words = nltk.corpus.stopwords.words('english')

    word_list = []
    for index, row in df.iterrows():
        text = ""
        for col in columns:
            text += row[col].decode('utf-8').strip()+" "

        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # take only words
        for token in tokens:
            # if re.search('[a-zA-Z]', token) and not re.search('^\'', token):
            if token not in punctuation and token not in en_stop_words and token not in stop_words:
                filtered_tokens.append(token)
        # delete stop words
        # filtered_tokens = [word for word in filtered_tokens if word not in stop_words and word not in stop_words_es]
        word_list += filtered_tokens

    return word_list


def order_and_select(word_list, qty):
    counts = Counter(word_list)
    labels, values = zip(*counts.items())
    # sort your values in descending order
    indSort = np.argsort(values)[::-1]

    # rearrange your data
    labels = np.array(labels)[indSort]
    values = np.array(values)[indSort]

    labels_firsts = labels[:qty]
    values_firsts = values[:qty]

    labels_lasts = labels[labels.size-qty:]
    values_lasts = values[values.size-qty:]

    return labels_firsts, values_firsts, labels_lasts, values_lasts



def get_word2_vec_representation(labels):
    os.chdir("/Users/esthergonzalez/TesisDecidim")
    # word_vectors = KeyedVectors.load("data/word2vec/GoogleNews-vectors-negative300.bin.gz")
    word_vectors = KeyedVectors.load_word2vec_format("data/word2vec/GoogleNews-vectors-negative300.bin.gz", binary=True)
    found_labels = [word for word in labels if word in word_vectors.wv.vocab]
    words2vec_matrix = word_vectors[found_labels]
    return words2vec_matrix, found_labels


def cos_sim(a, b):
    """ 2 vectors a, b and returns the cosine similarity according
    to the definition of the dot product
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def get_unique_words(df):
    word_list = build_word_list(df, ["title_en", "body_en"])
    counts = Counter(word_list)
    unique = []
    for key, count in counts.iteritems():
        if count == 1:
            unique.append(key)
    return unique


def get_categories(df):
    categories = {}
    idx = 0
    for category in set(list(df.loc[:, 'Category'])):
        categories[category] = idx
        idx += 1
    return categories