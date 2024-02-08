import time
import datetime
import re
import string
import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# SHOW TIMESTAMP
def show_time():
    timestamp = time.time()
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    human_readable_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')
    return human_readable_time

def encode2_ohe(df_text):
    vectorizer = CountVectorizer(binary=True, stop_words='english')

    # Fit and transform the text data
    X = vectorizer.fit_transform(df_text)

    # Convert the transformed data to a DataFrame
    one_hot_encoded_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    # Concatenate the output column with the one-hot encoded features
    #one_hot_encoded_df = pd.concat([data['output'], one_hot_encoded_df], axis=1)

    return (one_hot_encoded_df)


def encode2_bow(df_text):
    BOW=CountVectorizer()
    document_matrix=BOW.fit_transform(df_text)
    vocabulary_list=BOW.vocabulary_
    return (vocabulary_list, document_matrix.toarray())

def encode2_bigram(df_text):
    bigram=CountVectorizer(ngram_range=(2,2))
    document_matrix=bigram.fit_transform(df_text)
    vocabulary_list=bigram.vocabulary_
    return (vocabulary_list, document_matrix.toarray())

def encode2_tdidf(df_text):
    tfidf=TfidfVectorizer()
    document_matrix=tfidf.fit_transform(df_text)
    vocabulary_list=tfidf.get_feature_names_out()
    return (vocabulary_list, document_matrix.toarray())