#!/usr/bin/env python
# -*- coding: utf-8 -*-
#   
# Copyright (C) 2024
# 
# @author: Ezra Fu <erzhengf@andrew.cmu.edu>
# based on work by 
# Ishita <igoyal@andrew.cmu.edu> 
# Suyash <schavan@andrew.cmu.edu>
# Abhishek <asrivas4@andrew.cmu.edu>

"""
11-411/611 NLP Assignment 2
N-gram Language Model Implementation

Complete the LanguageModel class and other TO-DO methods.
"""

#######################################
# Import Statements
#######################################
from utils import *
from collections import Counter
from itertools import product
import argparse
import random
import math

#######################################
# TODO: get_ngrams()
#######################################
def get_ngrams(list_of_words, n):
    """
    Returns a list of n-grams for a list of words.
    Args
    ----
    list_of_words: List[str]
        List of already preprocessed and flattened (1D) list of tokens e.g. ["<s>", "hello", "</s>", "<s>", "bye", "</s>"]
    n: int
        n-gram order e.g. 1, 2, 3
    
    Returns:
        n_grams: List[Tuple]
            Returns a list containing n-gram tuples
    """
    n_grams = []
    for i in range(len(list_of_words) - n + 1):
        n_grams.append(tuple(list_of_words[i:i+n]))
        
    return n_grams

#######################################
# TODO: NGramLanguageModel()
#######################################
class NGramLanguageModel():
    def __init__(self, n, train_data, alpha=1):
        """
        Language model class.

        Args
        ____
        n: int
            n-gram order
        train_data: List[List]
            already preprocessed unflattened list of sentences. e.g. [["<s>", "hello", "my", "</s>"], ["<s>", "hi", "there", "</s>"]]
        alpha: float
            Smoothing parameter

        Other attributes:
            self.tokens: list of individual tokens present in the training corpus
            self.vocab: vocabulary dict with counts
            self.model: n-gram language model, i.e., n-gram dict with probabilties
            self.n_grams_counts: dictionary for storing the frequency of ngrams in the training data, keys being the tuple of words(n-grams) and value being their frequency
            self.prefix_counts: dictionary for storing the frequency of the (n-1) grams in the data, similar to the self.n_grams_counts
            As an example:
            For a trigram model, the n-gram would be (w1,w2,w3), the corresponding [n-1] gram would be (w1,w2)
        """
        self.n = n
        self.train_data = train_data
        self.smoothing = alpha
        self.tokens = []
        self.vocab = {}
        self.model = {}
        self.n_grams_counts = {}
        self.prefix_counts = {}

        self.build()


    def build(self):
        """
        Returns a n-gram dict with their smoothed probabilities. Remember to consider the edge case of n=1 as well

        You are expected to update the self.n_grams_counts and self.prefix_counts, and use those calculate the probabilities.
        """
        flattened_train_data = flatten(self.train_data)
        self.tokens = flattened_train_data
        self.vocab = Counter(flattened_train_data)
        n_grams = get_ngrams(flattened_train_data, self.n)
        self.n_grams_counts = Counter(n_grams)
        # Calculate prefix counts for higher-order n-grams
        if self.n > 1:
            for ngram in n_grams:
                prefix = ngram[:-1]
                self.prefix_counts[prefix] = self.prefix_counts.get(prefix, 0) + 1

        probs = self.get_smooth_probabilities(self.n_grams_counts)
        self.model = probs

        return probs


    def get_smooth_probabilities(self, ngrams):
        """
        Returns the smoothed probability of the n-gram, using Laplace Smoothing.
        Remember to consider the edge case of  n = 1
        HINT: Use self.n_gram_counts, self.tokens and self.prefix_counts
        """
        probs = {}
        V = len(self.vocab)  # Vocabulary size

        if self.n == 1:
            # For unigrams, use the frequency counts directly
            total_tokens = len(self.tokens)
            for ngram, count in ngrams.items():
                probs[ngram] = (count + self.smoothing) / (total_tokens + self.smoothing * V)
        else:
            # For higher-order n-grams, use the prefix counts
            for ngram, count in ngrams.items():
                prefix = ngram[:-1]
                prefix_count = self.prefix_counts.get(prefix, 0)
                probs[ngram] = (count + self.smoothing) / (prefix_count + self.smoothing * V)

        return probs

    def get_prob(self, ngram):
        """
        Returns the probability of the n-gram, using Laplace Smoothing.

        Args
        ____
        ngram: tuple
            n-gram tuple

        Returns
        _______
        float
            probability of the n-gram
        """
        if ngram in self.model:
            return self.model[ngram]
        
        V =  len(self.vocab)  # Vocabulary size
        if self.n == 1:
            total_tokens = len(self.tokens)
            prob = self.smoothing / (total_tokens + self.smoothing * V)
        else:
            prefix = ngram[:-1]
            prefix_count = self.prefix_counts.get(prefix, 0)
            prob = self.smoothing / (prefix_count + self.smoothing * V)
        
        self.model[ngram] = prob
        
        return prob

    def perplexity(self, test_data):
        """
        Returns perplexity calculated on the test data.
        Args
        ----------
        test_data: List[List]
            Already preprocessed nested list of sentences

        Returns
        -------
        float
            Calculated perplexity value
        """
        total_log_prob = 0

        flattened_test_data = flatten(test_data)
        N_tokens = len(flattened_test_data)
        if N_tokens == 0:
            return 0.0
        
        ngrams = get_ngrams(flattened_test_data, self.n)

        for ngram in ngrams:
            prob = self.get_prob(ngram)
            if prob == 0:
                return float('inf')
            total_log_prob += math.log(prob)
        perplexity = math.exp(-total_log_prob / N_tokens)

        return perplexity
