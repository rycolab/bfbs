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

"""Heuristics are used during A* decoding and are called to compose the
estimated look ahead costs. The ``Heuristic`` super class is defined
in the ``core`` module. 
"""

import copy
import logging

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Heuristic, Decoder
from cam.sgnmt.decoding.greedy import GreedyDecoder
from cam.sgnmt.utils import MESSAGE_TYPE_DEFAULT


class PredictorHeuristic(Heuristic):
    """The predictor heuristic relies on the 
    ``estimate_future_costs()`` implementation of the predictors. Use
    this heuristic to access predictor specific future cost functions,
    e.g. shortest path for the fst predictor.
    """
    
    def estimate_future_cost(self, hypo):
        """Returns the weighted sum of predictor estimates. """
        return Decoder.combi_arithmetic_unnormalized([
                                    (pred.estimate_future_cost(hypo), w)
                                            for (pred, w) in self.predictors])
    
    def initialize(self, src_sentence):
        """Calls ``initialize_heuristic()`` on all predictors. """
        for (pred, _) in self.predictors:
            pred.initialize_heuristic(src_sentence)
    
    def notify(self, message, message_type = MESSAGE_TYPE_DEFAULT):
        """This heuristic passes through notifications to the 
        predictors.
        """
        for (pred, _) in self.predictors:
            pred.notify(message, message_type)


class ScorePerWordHeuristic(Heuristic):
    """Using this heuristic results in length normalized scores instead
    of the pure sum of predictor scores for a partial hypothesis.
    Therefore, it is not a heuristic like in the classical A* sense.
    Instead, using the A* decoder with this heuristic simulates beam
    search which always keeps the hypotheses with the best per word
    scores.
    """
    
    def estimate_future_cost(self, hypo):
        """A* will put ``cost-score`` on the heap. In order to simulate
        length normalized beam search, we want to use ``-score/length``
        as partial hypothesis score. Therefore, this method returns
        ``-score/length + score``
        """
        if len(hypo.trgt_sentence) > 0:
            return hypo.score - hypo.score/len(hypo.trgt_sentence)
        return 0.0
    
    def initialize(self, src_sentence):
        """Empty method."""
        pass


class LastTokenHeuristic(Heuristic):
    """This heuristic reflects the score of the last token in the
    translation prefix only, ie. not the accumulated score. Using this
    with pure_heuristic_estimates leads to expanding the partial 
    hypothesis with the end token with the best individual score. This
    can be useful in search spaces in which bad translation prefixes
    imply low individual scores later.
    """
    
    def estimate_future_cost(self, hypo):
        """Returns the negative score of the last token in hypo."""
        return -Decoder.combi_arithmetic_unnormalized(hypo.score_breakdown[-1])
    
    def initialize(self, src_sentence):
        """Empty method."""
        pass



