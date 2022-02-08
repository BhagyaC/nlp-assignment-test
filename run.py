#!/usr/bin/env python

import random
import sys
import time

import numpy as np

from sgd import sgd
from data_process import A1dataset
from word2vec import word2vec_sgd_wrapper, skipgram, neg_sampling_loss_and_gradient

# Check Python Version
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 5

# Reset the random seed to make sure that everyone gets the same results
random.seed(42)
np.random.seed(6289)

dataset = A1dataset()
tokens = dataset.tokens()  # word2Idx
n_words = len(tokens)

# We are going to train 10-dimensional vectors for this assignment
dim_vectors = 10

# Context size
C = 5

startTime = time.time()
word_vectors = np.concatenate(
    ((np.random.rand(n_words, dim_vectors) - 0.5) /
     dim_vectors, np.zeros((n_words, dim_vectors))),
    axis=0)

word_vectors = sgd(
    lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
                                     neg_sampling_loss_and_gradient),
    word_vectors, 0.2, 35000, tokens, None, True, PRINT_EVERY=10)

print("sanity check: cost at convergence should be around or below 9")
print("training took %d seconds" % (time.time() - startTime))
