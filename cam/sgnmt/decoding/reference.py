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


class ReferenceDecoder(Decoder):
    
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
        super(ReferenceDecoder, self).__init__(decoder_args)
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

        
    def decode(self, src_sentence, trgt_sentence):
        self.trgt_sentence = trgt_sentence + [utils.EOS_ID]
        self.initialize_predictors(src_sentence)
        self.reward_bound(src_sentence)

        hypo = PartialHypothesis(self.get_predictor_states())
        while hypo.get_last_word() != utils.EOS_ID:
            self._expand_hypo(hypo)
                
        hypo.score = self.get_adjusted_score(hypo)
        self.add_full_hypo(hypo.generate_full_hypothesis())
        return self.get_full_hypos_sorted()


    def reward_bound(self, src_sentence):
        if self.reward_type == "bounded":
            # french is 0.72
            self.l = len(src_sentence)
        elif self.reward_type == "max":
            self.l = self.max_len

    def _expand_hypo(self,hypo):

        self.set_predictor_states(hypo.predictor_states)
        next_word = self.trgt_sentence[len(hypo.trgt_sentence)]
        ids, posterior, score_breakdown = self.apply_predictors()

        max_score = utils.max(posterior)
        hypo.predictor_states = self.get_predictor_states()

        hypo.score += posterior[next_word] 
        hypo.score_breakdown.append(score_breakdown[next_word])
        hypo.statistics.push(posterior[next_word], cur_max=max_score)
        hypo.trgt_sentence += [next_word]
        self.consume(next_word)
                
