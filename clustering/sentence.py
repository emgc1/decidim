import string
import nltk
import nltk.corpus
import numpy as np
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
STEM = False
PORTER_PREFERRED = True
COLUMNS = ['title en', 'body en']
###########################################################

lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('stopwords')
en_stop_words = nltk.corpus.stopwords.words('english')
# stemmer = SnowballStemmer("english")
snowball_stemmer = nltk.SnowballStemmer("english")
porter_stemmer = nltk.stem.PorterStemmer()
regex = re.compile('[%s]' % re.escape(string.punctuation))


def words_transformation(title, description=None, unique_words=[], category=None):
    if category and not pd.isna(category):
        text = (category+" "+title+" "+description)
    elif description:
        text = (title+" "+description)
    else:
        text = title
    text = text.decode('utf-8')
    # Split sentence into words
    # tokens = [regex.sub('', word).lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    tokens = [regex.sub('', word).lower().replace(u'\u200b', '').strip() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    tokens = [token for token in tokens if len(token) > 2 if token not in en_stop_words]
    # Remove punctuation and stopwords
    filtered_words = []
    for token in tokens:
        if token not in en_stop_words and token not in unique_words:
            if not STEM:
                filtered_words.append(word_lemmatization(token))
            else:
                filtered_words.append(word_stemming(token))
    return filtered_words
    # return tokens


def word_lemmatization(word, tag='n'):
    return lemmatizer.lemmatize(word, tag)


def word_stemming(word):
    if PORTER_PREFERRED:
        return porter_stemmer.stem(word)
    else:
        return snowball_stemmer.stem(word)


def add_category_features(category, word2vec, categories):
    feature = np.zeros(len(categories))
    feature[categories[category]] = 1
    return np.concatenate((word2vec, feature))
