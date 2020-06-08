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
        assert decoder_args.fairseq_temperature >= 1.0
        self.nbest = decoder_args.nbest

        
    def decode(self, src_sentence):
        self.initialize_predictors(src_sentence)
        hypos = [PartialHypothesis(self.get_predictor_states())]*self.nbest

        t = 0
        while hypos and t < self.max_len:
            next_hypos = []
            for seed, hypo in enumerate(hypos):
                if hypo.get_last_word() == utils.EOS_ID:
                    hypo.score = self.get_adjusted_score(hypo)
                    self.add_full_hypo(hypo.generate_full_hypothesis())
                else:
                    self._expand_hypo(hypo, seed=seed)
                    next_hypos.append(hypo)
            hypos = next_hypos
            t+=1

        for hypo in hypos:
            hypo.score = self.get_adjusted_score(hypo)
            self.add_full_hypo(hypo.generate_full_hypothesis())
                
        return self.get_full_hypos_sorted()

    
    def _expand_hypo(self, hypo, seed=0):

        self.set_predictor_states(hypo.predictor_states)
        ids, posterior = self.apply_predictors()
        probabilites = utils.softmax(posterior)
        next_word = self._sample(probabilites, seed)

        hypo.predictor_states = self.get_predictor_states()
        hypo.score += posterior[next_word]
        hypo.score_breakdown.append(posterior[next_word])
        hypo.trgt_sentence += [next_word]
        self.consume(next_word)


    def _sample(self, posterior, seed):
        np.random.seed(seed=seed)
        dist = np.random.multinomial(1, posterior, size=1)
        return int(np.where(dist[0]==1)[0])







    
