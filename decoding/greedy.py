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

"""Implementation of the greedy search strategy """

import utils
from decoding.core import Decoder, PartialHypothesis
import logging


class GreedyDecoder(Decoder):
    """The greedy decoder does not revise decisions and therefore does
    not have to maintain predictor states. Therefore, this 
    implementation is particularly simple and can be used as template
    for more complex decoders. The greedy decoder can be imitated with
    the ``BeamDecoder`` with beam size 1.
    """
    name = "greedy"
    def __init__(self, decoder_args):
        """Initialize the greedy decoder. """
        super(GreedyDecoder, self).__init__(decoder_args)
    
    def decode(self, src_sentence):
        """Decode a single source sentence in a greedy way: Always take
        the highest scoring word as next word and proceed to the next
        position. This makes it possible to decode without using the 
        predictors ``get_state()`` and ``set_state()`` methods as we
        do not have to keep track of predictor states.
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S> which make up the source sentence
        
        Returns:
            list. A list of a single best ``Hypothesis`` instance."""
        self.initialize_predictor(src_sentence)
        hypothesis = PartialHypothesis(self.get_predictor_states())
        while hypothesis.get_last_word() != utils.EOS_ID and len(hypothesis) < self.max_len:
            ids, posterior, original_posterior = self.apply_predictor(
                                                    hypothesis if self.gumbel else None, 1)
            trgt_word = ids[0]
            if self.gumbel:
                hypothesis.base_score += original_posterior[0]
                hypothesis.score_breakdown.append(original_posterior[0])
            else: 
                hypothesis.score += posterior[0]
                hypothesis.score_breakdown.append(posterior[0])
            hypothesis.trgt_sentence.append(trgt_word)
            
            self.consume(trgt_word)
        self.add_full_hypo(hypothesis.generate_full_hypothesis())
        return self.full_hypos
