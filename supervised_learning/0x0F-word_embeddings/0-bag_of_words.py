#!/usr/bin/env python3
"""0. Bag Of Words"""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """A function that creates a bag of words embedding matrix"""
    if vocab is None:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(sentences)
        vocab = vectorizer.get_feature_names()
    else:
        vectorizer = CountVectorizer(vocabulary=vocab)
        X = vectorizer.fit_transform(sentences)

    return X.toarray(), vocab
