# -*- coding: utf-8 -*-
# coding=utf-8
# Copyright 2019 The SGNMT Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of the A* search strategy """


import copy
import logging
import numpy as np
import time
import math

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder, PartialHypothesis


class SamplingDecoder(Decoder):
    
    def __init__(self, decoder_args):
        """Creates a new A* decoder instance. The following values are
        fetched from `decoder_args`:
        
            beam (int): beam width.
            early_stopping (bool): If this is true, partial hypotheses
                                   with score worse than the current
                                   best complete scores are not
                                   expanded. This applies when nbest is
                                   larger than one and inadmissible
                                   heuristics are used
            nbest (int): If this is set to a positive value, we do not
                         stop decoding at the first complete path, but
                         continue search until we collected this many
                         complete hypothesis. With an admissible
                         heuristic, this will yield an exact n-best
                         list.
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(SamplingDecoder, self).__init__(decoder_args)
        self.reward_coef = decoder_args.reward_coefficient
        self.reward_type = decoder_args.reward_type
        assert not (decoder_args.subtract_uni or decoder_args.subtract_marg) or decoder_args.ppmi

        self.epsilon = decoder_args.epsilon
        self.lmbda = decoder_args.lmbda
        self.use_heuristics = decoder_args.heuristics
        self.size_threshold = self.beam*decoder_args.memory_threshold_coef\
            if decoder_args.memory_threshold_coef > 0 else utils.INF

        self.guidos = utils.split_comma(decoder_args.guido)
        self.guido_lambdas = utils.split_comma(decoder_args.guido_lambdas, func=float)
        if any(g in ['variance', 'local_variance'] for g in self.guidos):
            self.not_monotonic = True

        self.nbest = decoder_args.nbest
        self.temperature = decoder_args.temperature

        
    def decode(self, src_sentence):
        self.initialize_predictors(src_sentence)
        self.reward_bound(src_sentence)

        hypos = [PartialHypothesis(self.get_predictor_states())]*self.nbest

        t = 0
        while hypos and t < self.max_len:
            next_hypos = []
            for seed, hypo in enumerate(hypos):
                if hypo.get_last_word() == utils.EOS_ID:
                    hypo.score = self.get_adjusted_score(hypo)
                    self.add_full_hypo(hypo.generate_full_hypothesis())
                else:
                    next_hypos.append(self._expand_hypo(hypo, seed=seed, temperature=self.temperature))
            hypos = next_hypos
            t+=1

        for hypo in hypos:
            hypo.score = self.get_adjusted_score(hypo)
            self.add_full_hypo(hypo.generate_full_hypothesis())
                
        return self.get_full_hypos_sorted()


    def reward_bound(self, src_sentence):
        if self.reward_type == "bounded":
            # french is 0.72
            self.l = len(src_sentence)
        elif self.reward_type == "max":
            self.l = self.max_len
    
    def _expand_hypo(self, hypo, seed=0, temperature=1.):
        """Get the best beam size expansions of ``hypo``.
        
        Args:
            hypo (PartialHypothesis): Hypothesis to expand
        
        Returns:
            list. List of child hypotheses
        """

        self.set_predictor_states(copy.copy(hypo.predictor_states))
        if not hypo.word_to_consume is None: # Consume if cheap expand
            self.consume(hypo.word_to_consume)
            hypo.word_to_consume = None

        ids, posterior, score_breakdown = self.apply_predictors()
 
        max_score = utils.max(posterior)
        vf = np.vectorize(lambda x: self.get_pos_score(hypo, x, max_score))
        scores = vf(posterior)
        hypo.predictor_states = self.get_predictor_states()

        shifted = utils.softmax(scores/temperature)
        ind = self._sample(shifted, seed)
        return hypo.cheap_expand(
                        ids[ind],
                        posterior[ind],
                        score_breakdown[ind], 
                        max_score=max_score) 

    def _sample(self, posterior, seed):
        np.random.seed(seed=seed)
        dist = np.random.multinomial(1, posterior, size=1)
        return int(np.where(dist[0]==1)[0])







    
