#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random

import numpy as np


class A1dataset:
    def __init__(self, path=None, dataset='penntreebank', table_size=15000):
        self._reject_prob = None
        self._sample_table = None
        self._all_sentences = None
        self._num_sentences = None
        self._cum_sent_len = None
        self._sent_lengths = None
        self._rev_tokens = None
        self._wordcount = None
        self._token_freq = None
        self._tokens = None
        self._sentences = None
        self._dataset = dataset

        app_root = os.path.dirname(os.path.abspath(__file__))
        if not path:
            path = os.path.join(app_root, "data")

        self.path = path
        self.table_size = table_size

    def tokens(self):
        if hasattr(self, "_tokens") and self._tokens:
            return self._tokens

        tokens = dict()
        token_freq = dict()
        wordcount = 0
        rev_tokens = []
        idx = 0

        for sentence in self.sentences():
            for w in sentence:
                wordcount += 1
                if not w in tokens:
                    tokens[w] = idx
                    rev_tokens += [w]
                    token_freq[w] = 1
                    idx += 1
                else:
                    token_freq[w] += 1

        tokens["UNK"] = idx
        rev_tokens += ["UNK"]
        token_freq["UNK"] = 1
        wordcount += 1

        self._tokens = tokens
        self._token_freq = token_freq
        self._wordcount = wordcount
        self._rev_tokens = rev_tokens
        return self._tokens

    def sentences(self):
        if hasattr(self, "_sentences") and self._sentences:
            return self._sentences

        sentences = []
        with open(self.path + f"/{self._dataset}/sentences.txt", "r") as f:
        # with open(self.path + f"/{self._dataset}/sentences.txt", "r") as f:
            for line in f:
                splitted = line.strip().split()
                # Deal with some peculiar encoding issues with this file
                sentences += [[w.lower() for w in splitted]]

        self._sentences = sentences
        self._sent_lengths = np.array([len(s) for s in sentences])
        self._cum_sent_len = np.cumsum(self._sent_lengths)

        return self._sentences

    def num_sentences(self):
        if hasattr(self, "_numSentences") and self._num_sentences:
            return self._num_sentences
        else:
            self._num_sentences = len(self.sentences())
            return self._num_sentences

    def all_sentences(self):
        if hasattr(self, "_all_sentences") and self._all_sentences:
            return self._all_sentences

        sentences = self.sentences()
        reject_prob = self.reject_prob()
        tokens = self.tokens()
        all_sentences = [[w for w in s
                          if 0 >= reject_prob[tokens[w]] or random.random() >= reject_prob[tokens[w]]]
                         for s in sentences * 30]

        all_sentences = [s for s in all_sentences if len(s) > 1]

        self._all_sentences = all_sentences

        return self._all_sentences

    def get_random_context(self, C=5):
        allsent = self.all_sentences()
        sentID = random.randint(0, len(allsent) - 1)
        sent = allsent[sentID]
        wordID = random.randint(0, len(sent) - 1)

        context = sent[max(0, wordID - C):wordID]  # 前5个
        if wordID + 1 < len(sent):
            context += sent[wordID + 1:min(len(sent), wordID + C + 1)]  # 后5个

        centerword = sent[wordID]
        context = [w for w in context if w != centerword]

        if len(context) > 0:
            return centerword, context
        else:
            return self.get_random_context(C)

    def sample_table(self):
        if hasattr(self, '_sample_table') and self._sample_table is not None:
            return self._sample_table

        n_tokens = len(self.tokens())
        sampling_freq = np.zeros((n_tokens,))
        self.all_sentences()
        i = 0
        for w in range(n_tokens):
            w = self._rev_tokens[i]
            if w in self._token_freq:
                freq = 1.0 * self._token_freq[w]
                # Reweigh
                freq = freq ** 0.75
            else:
                freq = 0.0
            sampling_freq[i] = freq
            i += 1

        sampling_freq /= np.sum(sampling_freq)
        sampling_freq = np.cumsum(sampling_freq) * self.table_size

        self._sample_table = [0] * self.table_size

        j = 0
        for i in range(self.table_size):
            while i > sampling_freq[j]:
                j += 1
            self._sample_table[i] = j

        return self._sample_table

    def reject_prob(self):
        if hasattr(self, '_reject_prob') and self._reject_prob is not None:
            return self._reject_prob

        threshold = 1e-5 * self._wordcount

        n_tokens = len(self.tokens())
        reject_prob = np.zeros((n_tokens,))
        for i in range(n_tokens):
            w = self._rev_tokens[i]
            freq = 1.0 * self._token_freq[w]
            # Reweigh
            reject_prob[i] = max(0, 1 - np.sqrt(threshold / freq))

        self._reject_prob = reject_prob
        return self._reject_prob

    def sample_token_idx(self):
        return self.sample_table()[random.randint(0, self.table_size - 1)]
