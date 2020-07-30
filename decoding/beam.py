"""Implementation of the beam search strategy """

import copy
import logging
import time

import utils
from decoding.core import Decoder, PartialHypothesis
import numpy as np


class BeamDecoder(Decoder):
    """This decoder implements standard beam search and several
    variants of it such as diversity promoting beam search and beam
    search with heuristic future cost estimates. This implementation
    supports risk-free pruning.
    """
    name = 'beam'
    def __init__(self, decoder_args):
        """Creates a new beam decoder instance. The following values
        are fetched from `decoder_args`:
        
            hypo_recombination (bool): Activates hypo recombination 
            beam (int): Absolute beam size. A beam of 12 means
                        that we keep track of 12 active hypotheses
 
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
        self.nbest = max(1, decoder_args.nbest)
        self.beam_size = decoder_args.beam if not self.gumbel else self.nbest
        if decoder_args.early_stopping:
            self.stop_criterion = self._best_eos 
        else:
            self.stop_criterion = self._all_eos
        self.reward = None  
    
    def _best_eos(self, hypos):
        """Returns true if the best hypothesis ends with </S>"""
        ln_scores = [self.get_adjusted_score(hypo) for hypo in hypos]
        best_inds = utils.argmax_n(ln_scores, self.nbest)
        return all([hypos[ind].get_last_word() == utils.EOS_ID for ind in best_inds])
            
    def _all_eos(self, hypos):
        """Returns true if the all hypotheses end with </S>"""
        return all([hypo.get_last_word() == utils.EOS_ID for hypo in hypos])


    def _get_next_hypos(self, all_hypos, all_scores):
        """Get hypos for the next iteration. """

        inds = utils.argmax_n(all_scores, self.beam_size)
        return [all_hypos[ind] for ind in inds]
    
    def _get_initial_hypos(self):
        """Get the list of initial ``PartialHypothesis``. """
        return [PartialHypothesis(self.get_predictor_states())]
    
    def decode(self, src_sentence):
        """Decodes a single source sentence using beam search. """
        self.count = 0
        self.time = 0
        self.initialize_predictor(src_sentence)
        hypos = self._get_initial_hypos()
        it = 0
        if self.reward:
            self.l = len(src_sentence)
        while not self.stop_criterion(hypos) and it < self.max_len:
            it = it + 1
            next_hypos = []
            next_scores = []
            for hypo in hypos:
                if hypo.get_last_word() == utils.EOS_ID:
                    next_hypos.append(hypo)
                    next_scores.append(self.get_adjusted_score(hypo))
                    continue 
                for next_hypo in self._expand_hypo(hypo, self.beam_size):
                    next_hypos.append(next_hypo)
                    next_scores.append(self.get_adjusted_score(next_hypo))
            hypos = self._get_next_hypos(next_hypos, next_scores)
            
        return self.get_full_hypos_sorted(hypos)


class DiverseBeamDecoder(BeamDecoder):
    """This decoder implements diversity promoting beam search Vijayakumar et. al. (2016).
    """
    name = 'diverse_beam'
    def __init__(self, decoder_args):
        
        super(DiverseBeamDecoder, self).__init__(decoder_args)
        assert not self.gumbel

        self.beam_size = decoder_args.beam
        self.num_groups = decoder_args.diversity_groups
        self.lmbda = decoder_args.diversity_reward
        self.group_sizes = [self.beam_size//self.num_groups]*self.num_groups
        for i in range(self.beam_size - self.group_sizes[0]*self.num_groups):
            self.group_sizes[i] += 1
        assert sum(self.group_sizes) == self.beam_size
        
    def _get_initial_hypos(self):
        """Get the list of initial ``PartialHypothesis``. """
        bos_hypo = PartialHypothesis(self.get_predictor_states())
        hypos = self._expand_hypo(bos_hypo, self.beam_size)
        inds = list(np.cumsum(self.group_sizes))
        return [hypos[a:b] for a,b in zip([0] + inds[:-1], inds)]

    def _get_next_hypos(self, all_hypos, size, other_groups=None):
        """Get hypos for the next iteration. """
        all_scores = np.array([self.get_adjusted_score(hypo) for hypo in all_hypos])
        if other_groups:
            all_scores = all_scores + self.lmbda*self.hamming_distance_penalty(all_hypos, 
                                                            utils.flattened(other_groups))
        inds = utils.argmax_n(all_scores, size)
        return [all_hypos[ind] for ind in inds]

    def decode(self, src_sentence):
        """Decodes a single source sentence using beam search. """
        self.count = 0
        self.time = 0
        self.initialize_predictor(src_sentence)
        hypos = self._get_initial_hypos()
        it = 1
        if self.reward:
            self.l = len(src_sentence)
        while not self.stop_criterion(utils.flattened(hypos)) and it < self.max_len:
            it = it + 1
            next_hypos = []
            for i, group in enumerate(hypos):
                next_group = []
                for hypo in group:
                    if hypo.get_last_word() == utils.EOS_ID:
                        next_group.append(hypo)
                        continue 
                    for next_hypo in self._expand_hypo(hypo):
                        next_group.append(next_hypo)
                next_hypos.append(self._get_next_hypos(next_group, self.group_sizes[i], next_hypos))
            hypos = next_hypos

        return self.get_full_hypos_sorted(utils.flattened(hypos))
                        
    @staticmethod
    def hamming_distance_penalty(set1, set2):
        longest_hypo = len(max(set1 + set2, key=len))
        hypos = utils.as_ndarray(set1, min_length=longest_hypo)
        other_hypos = utils.as_ndarray(set2, min_length=longest_hypo)
        return np.apply_along_axis(lambda x: utils.hamming_distance(x, other_hypos), 1, hypos)

