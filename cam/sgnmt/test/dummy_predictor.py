import logging
import os

from cam.sgnmt import utils
from cam.sgnmt.predictors.core import Predictor

import numpy as np
import copy
import hashlib


class DummyPredictor(Predictor):
    """Predictor for using fairseq models."""

    def __init__(self, vocab_size=10, n_cpu_threads=-1, seed=0):
        """Initializes a fairseq predictor.

        Args:
            model_path (string): Path to the fairseq model (*.pt). Like
                                 --path in fairseq-interactive.
            lang_pair (string): Language pair string (e.g. 'en-fr').
            user_dir (string): Path to fairseq user directory.
            n_cpu_threads (int): Number of CPU threads. If negative,
                                 use GPU.
        """
        super(DummyPredictor, self).__init__()
        self.vocab_size = vocab_size
        self.rg = np.random.default_rng(seed=seed)
        self.eos_id = utils.EOS_ID
        self.num_dists = 1000
        # Create fake distributions with random number generator
        self.prob_dists = [self.rg.standard_normal(self.vocab_size) for i in range(self.num_dists)]

    def get_unk_probability(self, posterior):
        """Fetch posterior[utils.UNK_ID]"""
        return utils.common_get(posterior, utils.UNK_ID, utils.NEG_INF)
                
    def predict_next(self):
        s = str(self.src) + str(self.consumed)
        hash_key = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % self.num_dists
        unnorm_posterior = copy.copy(self.prob_dists[hash_key])
        unnorm_posterior[self.eos_id] -= unnorm_posterior.max()/len(self.consumed)
        return utils.log_softmax(unnorm_posterior, temperature=0.2)
    
    def initialize(self, src_sentence):
        """Initialize source tensors, reset consumed."""
        self.src = src_sentence
        self.consumed =  [utils.GO_ID]
   
    def consume(self, word):
        """Append ``word`` to the current history."""
        self.consumed.append(word)
    
    def get_empty_str_prob(self):
        pass

    def get_state(self):
        """The predictor state is the complete history."""
        return self.consumed, [[]]
    
    def set_state(self, state):
        """The predictor state is the complete history."""
        consumed, inc_states = state
        self.consumed = copy.copy(consumed)

    def is_equal(self, state1, state2):
        """Returns true if the history is the same """
        return state1[0] == state2[0]

