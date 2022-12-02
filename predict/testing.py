import re
import numpy as np
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import TweetTokenizer

# cleaning + lowercase


def cleanlower(ulasan):
    ulasan = ulasan.str.replace(r"[^a-zA-Z]", ' ', regex=True)
    ulasan = ulasan.str.lower()
    return ulasan


# stemming
stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()


def stemming(ulasan):
    ulasan = ulasan.apply(lambda x: stemmer.stem(x))
    return ulasan

# tokenizing


def tokenizing(ulasan):
    tokenizer = TweetTokenizer(
        preserve_case=False, strip_handles=True, reduce_len=True)
    ulasan = ulasan.apply(lambda x: tokenizer.tokenize(x))
    return ulasan


# stopword
# string_khusus = ['id','ulasan']
url = 'https://raw.githubusercontent.com/appermana1/data/master/stopword.csv'
stopword_import = pd.read_csv(url)
stopword_custom = stopword_import['0']
stopword_custom = stopword_custom.to_numpy()
# print(stopword_custom)


def stopwords(ulasan):
    output = " ".join([i for i in ulasan if i not in stopword_custom])
    return output


def preprocessing(ulasan):
    b = cleanlower(ulasan)
    c = stemming(b)
    d = tokenizing(c)
    e = d.apply(lambda x: stopwords(x))
    return e

# Split


def split(ulasan):
    ulasan = ulasan.str.split()
    return ulasan


def unique_term(word):
    world = ''
    for a in word:
        world = np.union1d(a, world)
    return world


def bag_of_word(ulasan):

    from sklearn.feature_extraction.text import CountVectorizer
    coun_vect = CountVectorizer()
    count_matrix = coun_vect.fit_transform(ulasan)
    count_array = count_matrix.toarray()
    df_bow = pd.DataFrame(
        data=count_array, columns=coun_vect.get_feature_names())
    max_value = np.max(count_array)

    bow = dict()
    bow['max'] = max_value
    bow['bow'] = df_bow

    return bow


def fix_preprocessing(df_ulasan):
    prp = preprocessing(df_ulasan)
    df_bow = bag_of_word(prp)
    cek_term = unique_term(tokenizing(prp))
    return df_bow


def validate_test_bow(test_bow, df_likelihood_term):
    df_validate = pd.DataFrame()
    for a in test_bow:
        for term in df_likelihood_term:
            if a == term:
                df_validate[a] = test_bow[a]
    return df_validate


def prob_posterior_doc(df_prior, prior_label, max_values_test_bow, validate_test_bow, df_likelihood, df_likelihood_term):
    doc_prob = pd.DataFrame()
    for label in prior_label:
        dict_replace_value = dict()
        dict_replace_value = validate_test_bow.copy()
        for term in validate_test_bow:
            get_term = df_likelihood[df_likelihood_term == term]
            get_value = get_term[label]
            for c in get_value:
                dict_replace_value[term] = dict_replace_value[term]*c
        likelihood = dict_replace_value.sum(axis=1)
        get_prior_value = df_prior[prior_label == label]
        for prior in get_prior_value['prior']:
            # rumus 3.7
            doc_prob[label] = prior*likelihood
    return doc_prob


def predict_doc(prob_doc, df_test_ulasan, df_test, validate_bow):
    df_pred = pd.DataFrame()
    # print(prob_doc)
    max_values = prob_doc.max(axis=1)
    df_pred = pd.concat([df_test, prob_doc], axis=1)
    for i in range(0, len(df_test_ulasan)):
        df_pred['pred'] = 0.0

    for i in range(0, len(df_test_ulasan)):
        if max_values[i] != 0:
            for x in prob_doc:
                if prob_doc[x][i] == max_values[i]:
                    # kolom pred harus inisiasi jml baris
                    df_pred['pred'][i] = x
                    # print(x)
    return df_pred


def predict_dataset(df_test, df_prior, df_likelihood):
    test_bow = fix_preprocessing(df_test['ulasan'])

    test_validate = validate_test_bow(test_bow['bow'], df_likelihood['term'])

    prob_doc = prob_posterior_doc(
        df_prior, df_prior['label'], test_bow['max'], test_validate, df_likelihood, df_likelihood['term'])

    pred = predict_doc(prob_doc, df_test['ulasan'], df_test, test_validate)

    test = dict()
    test['df_pred'] = pred
    test['validate'] = test_validate
    test['prob_doc'] = prob_doc

    return test
