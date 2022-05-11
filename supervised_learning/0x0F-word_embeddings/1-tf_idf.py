#!/usr/bin/env python3
"""1. TF-IDF"""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """A function that creates a TF-IDF embedding"""
    if vocab is None:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)
        vocab = vectorizer.get_feature_names()
    else:
        vectorizer = TfidfVectorizer(vocabulary=vocab)
        X = vectorizer.fit_transform(sentences)

    return X.toarray(), vocab
