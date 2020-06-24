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

"""Implementation of the beam search strategy """

import copy
import logging
import time

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder, PartialHypothesis
import numpy as np


class BeamDecoder(Decoder):
    """This decoder implements standard beam search and several
    variants of it such as diversity promoting beam search and beam
    search with heuristic future cost estimates. This implementation
    supports risk-free pruning and hypotheses recombination.
    """
    
    def __init__(self, decoder_args):
        """Creates a new beam decoder instance. The following values
        are fetched from `decoder_args`:
        
            hypo_recombination (bool): Activates hypo recombination 
            beam (int): Absolute beam size. A beam of 12 means
                        that we keep track of 12 active hypotheses
            sub_beam (int): Number of children per hypothesis. Set to
                            beam size if zero.
            pure_heuristic_scores (bool): Hypotheses to keep in the beam
                                          are normally selected 
                                          according the sum of partial
                                          hypo score and future cost
                                          estimates. If set to true, 
                                          partial hypo scores are 
                                          ignored.
            diversity_factor (float): If this is set to a positive 
                                      value we add diversity promoting
                                      penalization terms to the partial
                                      hypothesis scores following Li
                                      and Jurafsky, 2016
            early_stopping (bool): If true, we stop when the best
                                   scoring hypothesis ends with </S>.
                                   If false, we stop when all hypotheses
                                   end with </S>. Enable if you are
                                   only interested in the single best
                                   decoding result. If you want to 
                                   create full 12-best lists, disable

        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(BeamDecoder, self).__init__(decoder_args)
        self.diversity_factor = decoder_args.decoder_diversity_factor
        self.diverse_decoding = (self.diversity_factor > 0.0)
        if self.diversity_factor > 0.0:
            logging.fatal("Diversity promoting beam search is not implemented "
                          "yet")
        self.nbest = max(1, decoder_args.nbest)
        self.beam_size = decoder_args.beam
        self.sub_beam_size = decoder_args.sub_beam
        if self.sub_beam_size <= 0:
            self.sub_beam_size = decoder_args.beam
        if decoder_args.early_stopping:
            self.stop_criterion = self._best_eos 
        else:
            self.stop_criterion = self._all_eos
        self.reward = None #1.3

    
    
    def _best_eos(self, hypos):
        """Returns true if the best hypothesis ends with </S>"""
        if self.reward:
            ln_scores = [self.get_adjusted_score(hypo) for hypo in hypos]
            return hypos[np.argsort(ln_scores)[-1]].get_last_word() != utils.EOS_ID
            
        return hypos[0].get_last_word() != utils.EOS_ID

    def _all_eos(self, hypos):
        """Returns true if the all hypotheses end with </S>"""
        for hypo in hypos:
            if hypo.get_last_word() != utils.EOS_ID:
                return True
        return False
    

    
    def _filter_equal_hypos(self, hypos, scores):
        """Apply hypo recombination to the hypotheses in ``hypos``.
        
        Args:
            hypos (list): List of hypotheses
            scores (list): hypo scores with heuristic estimates
        
        Return:
            list. List with hypotheses in ``hypos`` after applying
            hypotheses recombination.
        """
        new_hypos = []
        for idx in reversed(np.argsort(scores)):
            candidate = hypos[idx]
            self.set_predictor_states(copy.deepcopy(candidate.predictor_states))
            if not candidate.word_to_consume is None:
                self.consume(candidate.word_to_consume)
                candidate.word_to_consume = None
                self.count += 1
                candidate.predictor_states = self.get_predictor_states()
            valid = True
            for hypo in new_hypos:
                if self.are_equal_predictor_states(
                                                hypo.predictor_states,
                                                candidate.predictor_states):
                    logging.debug("Hypo recombination: %s > %s" % (
                                                 hypo.trgt_sentence,
                                                 candidate.trgt_sentence))
                    valid = False
                    break
            if valid:
                new_hypos.append(candidate)
                if len(new_hypos) >= self.beam_size:
                    break
        return new_hypos

    def _get_next_hypos(self, all_hypos, all_scores):
        """Get hypos for the next iteration. """
        hypos = [all_hypos[idx]
                        for idx in np.argsort(all_scores)[-self.beam_size:]]
        hypos.reverse()
        return hypos
    
    
    def _get_initial_hypos(self):
        """Get the list of initial ``PartialHypothesis``. """
        return [PartialHypothesis(self.get_predictor_states())]
    
    def decode(self, src_sentence):
        """Decodes a single source sentence using beam search. """
        self.count = 0
        self.time = 0
        self.initialize_predictors(src_sentence)
        hypos = self._get_initial_hypos()
        it = 0
        if self.reward:
            self.l = len(src_sentence)
        while self.stop_criterion(hypos):
            if it > self.max_len: # prevent infinite loops
                break
            it = it + 1
            next_hypos = []
            next_scores = []
            self.min_score = utils.NEG_INF
            self.best_scores = []
            for hypo in hypos:
                if hypo.get_last_word() == utils.EOS_ID:
                    next_hypos.append(hypo)
                    next_scores.append(self.get_adjusted_score(hypo))
                    continue 
                for next_hypo in self._expand_hypo(hypo, self.sub_beam_size):
                    next_score = self.get_adjusted_score(next_hypo)
                    if next_score > self.min_score:
                        next_hypos.append(next_hypo)
                        next_scores.append(next_score)
            hypos = self._get_next_hypos(next_hypos, next_scores)
        for hypo in hypos:
            if hypo.get_last_word() == utils.EOS_ID:
                hypo.score = self.get_adjusted_score(hypo)
                self.add_full_hypo(hypo.generate_full_hypothesis()) 
        if not self.full_hypos:
            logging.warn("No complete hypotheses found for %s" % src_sentence)

        if len(self.full_hypos) < self.nbest:
            logging.warn("Adding incomplete hypotheses as candidates for %s" % src_sentence)
            for hypo in hypos:
                if hypo.get_last_word() != utils.EOS_ID:
                    hypo.score = self.get_adjusted_score(hypo)
                    self.add_full_hypo(hypo.generate_full_hypothesis()) 
            
        print("Count", self.count)
        return self.get_full_hypos_sorted()
