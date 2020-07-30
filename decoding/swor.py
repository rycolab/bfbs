import numpy as np
import time
import copy
import logging
from collections import defaultdict
from datastructures.sum_heap import SumHeap 

import utils
import sampling_utils
from decoding.core import Decoder, PartialHypothesis


class BasicSworDecoder(Decoder):
    name = "basic_swor"
    def __init__(self, decoder_args):
        """Creates a new SWOR decoder instance. The following values are
        fetched from `decoder_args`:
        
            nbest (int): number of desired samples. 
            temperature (float): temperature for shifting probability distributions
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(BasicSworDecoder, self).__init__(decoder_args)
        self.nbest = decoder_args.nbest
        self.early_stopping = decoder_args.early_stopping
        assert not self.gumbel
        
    def decode(self, src_sentence, seed=0):
        self.initialize_predictor(src_sentence)
        self.covered_lprob = utils.NEG_INF

        while len(self.full_hypos) < self.nbest and self.samples_left():
            if np.exp(self.covered_lprob) >= 1.0 - utils.MACHINE_EPS:
                logging.warn("Samples cover 100% of probability. Behavior beyond this point is undefined")
            self.reset_predictor(src_sentence)
            hypo = PartialHypothesis(self.get_predictor_states())
            hypo, score = self._expand_hypo(hypo, seed=seed+len(self.full_hypos))
            self.add_full_hypo(hypo.generate_full_hypothesis())
            self.covered_lprob = utils.log_add(self.covered_lprob, score)
            
        logging.info("%d sentences covering %f probability" %(len(self.full_hypos), np.exp(self.covered_lprob)))
        return self.full_hypos

    def initialize_predictor(self, src_sentence):
        self.dists = MapDist()
        super().initialize_predictor(src_sentence)

    def _expand_hypo(self, hypo, seed=0):
        if hypo.get_last_word() == utils.EOS_ID or len(hypo) == self.max_len:
            return hypo, 0.0

        prefix = tuple(hypo.trgt_sentence)
        if not prefix in self.dists:
            if self.start:
                # prefix has no longer previously been seen. One deep copy to get started
                hypo.predictor_states = copy.deepcopy(hypo.predictor_states)
                self.set_predictor_states(hypo.predictor_states)
                self.start = False
            if hypo.word_to_consume is not None:
                self.consume(hypo.word_to_consume)
                hypo.word_to_consume = None
    
            ids, posterior, _ = self.apply_predictor()
            # assert not np.any(np.isnan(lprobabilities))
            self.dists.add_dist(prefix, ids, utils.log_softmax(posterior, self.temperature), self.get_predictor_states())
        
        ids, lprobabilities, adjusted_lprobabilities, states = self.dists.get(prefix)
        hypo.predictor_states = states

        ind = adjusted_lprobabilities.sample()
        next_word = ids[ind]

        hypo.score += adjusted_lprobabilities[ind]
        hypo.score_breakdown.append(lprobabilities[ind])
        hypo.trgt_sentence += [next_word]
        hypo.word_to_consume = next_word
        hypo, score = self._expand_hypo(hypo, seed=seed)
        score += lprobabilities[ind] 
        self.dists.adjust(prefix, next_word, score)
        return hypo, score
         
    def reset_predictor(self, src_sentence):
        self.start = True
        self.predictor.initialize(src_sentence)

    def samples_left(self):
        if len(self.full_hypos) == 0:
            return True
        if self.early_stopping and np.exp(self.covered_lprob) >= 1.0 - utils.MACHINE_EPS:
            return False
        start_hash = tuple()
        _, _, adjusted_lprobabilities, _ = self.dists.get(start_hash)
        n, d = adjusted_lprobabilities.n, adjusted_lprobabilities.d
        return np.any(~np.isnan(adjusted_lprobabilities.S[d:d+n]) > utils.NEG_INF )


class SworDecoder(BasicSworDecoder):
    name = "swor"
    def __init__(self, decoder_args):
        """Creates a new SWOR decoder instance. The following values are
        fetched from `decoder_args`:
        
            nbest (int): number of desired samples. 
            temperature (float): temperature for shifting probability distributions
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(SworDecoder, self).__init__(decoder_args)

    def _expand_hypo(self, hypo, seed=0):
        if hypo.get_last_word() == utils.EOS_ID or len(hypo) == self.max_len:
            return hypo, self.dists.marg(tuple(hypo.trgt_sentence))
        prefix = tuple(hypo.trgt_sentence)
        if not prefix in self.dists:
            if self.start:
                # prefix has no longer previously been seen. One deep copy to get started
                hypo.predictor_states = copy.deepcopy(hypo.predictor_states)
                self.set_predictor_states(hypo.predictor_states)
                self.start = False
            if hypo.word_to_consume is not None:
                self.consume(hypo.word_to_consume)
                hypo.word_to_consume = None
            
            ids, posterior, _ = self.apply_predictor()
            marg = self.dists.marg(prefix)
            lprobabilities = utils.log_softmax(posterior, self.temperature) + marg
            self.dists.add_dist(prefix, ids, lprobabilities, self.get_predictor_states())

        ids, lprobabilities, adjusted_lprobabilities, states = self.dists.get(prefix)
        hypo.predictor_states = states

        ind = adjusted_lprobabilities.sample()
        next_word = ids[ind]

        hypo.score += adjusted_lprobabilities[ind]
        hypo.score_breakdown.append(lprobabilities[ind])
        hypo.trgt_sentence += [next_word]
        hypo.word_to_consume = next_word
        hypo, final = self._expand_hypo(hypo, seed=seed)
        self.dists.adjust(prefix, next_word, final)
        return hypo, final


class MemEfficientSworDecoder(BasicSworDecoder):
    name = "mem_eff_swor"
    def __init__(self, decoder_args):
        """Creates a new SWOR decoder instance. The following values are
        fetched from `decoder_args`:
        
            nbest (int): number of desired samples. 
            temperature (float): temperature for shifting probability distributions
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(MemEfficientSworDecoder, self).__init__(decoder_args)

    
    def _expand_hypo(self, hypo, seed=0):
        if hypo.get_last_word() == utils.EOS_ID or len(hypo) == self.max_len:
            return hypo, 0.0
        if hypo.word_to_consume is not None:
            self.consume(hypo.word_to_consume)
            hypo.word_to_consume = None
        prefix = tuple(hypo.trgt_sentence)
        ids, posterior, _ = self.apply_predictor()
        lprobabilities = utils.log_softmax(posterior, self.temperature)
        adjusted_lprobabilities = self.adjust_probabilities(lprobabilities, prefix, ids)

        ind = sampling_utils.log_multinomial_sample(adjusted_lprobabilities, seed)
        next_word = ids[ind]

        hypo.score += adjusted_lprobabilities[ind]
        hypo.score_breakdown.append(lprobabilities[ind])
        hypo.trgt_sentence += [next_word]
        hypo.word_to_consume = next_word
        hypo, score = self._expand_hypo(hypo, seed=seed)
        score += lprobabilities[ind] 
        self.ids[prefix][next_word] = utils.log_add(score, self.ids[prefix][next_word])
        return hypo, score
        

    def adjust_probabilities(self, lprobabilities, hash_rep, ids):
        lprobabilities = np.copy(lprobabilities)
        for k, val in self.ids[hash_rep].items():
            ind = utils.binary_search(ids, k)
            lprobabilities[ind] = utils.log_minus(lprobabilities[ind], val)
        return lprobabilities

    def initialize_predictor(self, src_sentence):
        self.ids = defaultdict(lambda: defaultdict(lambda: utils.NEG_INF))
        self.src_sentence = src_sentence
        super().initialize_predictor(self.src_sentence)

    def samples_left(self):
        if len(self.full_hypos) == 0:
            return True
        if self.early_stopping and np.exp(self.covered_lprob) >= 1.0 - utils.MACHINE_EPS:
            return False
        self.reset_predictor(self.src_sentence)
        ids, posterior, _ = self.apply_predictor()
        start_hash = tuple()
        lprobabilities = utils.log_softmax(posterior, self.temperature)
        adjusted_lprobabilities = self.adjust_probabilities(lprobabilities, start_hash, ids)
        return np.any(~np.isnan(adjusted_lprobabilities) > utils.NEG_INF )


class CPSworDecoder(Decoder):
    name = "cp_swor"
    def __init__(self, decoder_args):
        """Creates a new SWOR decoder instance. The following values are
        fetched from `decoder_args`:
        
            nbest (int): number of desired samples. 
            temperature (float): temperature for shifting probability distributions
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(CPSworDecoder, self).__init__(decoder_args)
        self.nbest = decoder_args.nbest
        self.early_stopping = decoder_args.early_stopping
        self.sample_beam = self.nbest
        assert not self.gumbel
    
    def decode(self, src_sentence, seed=0):
        self.initialize_predictor(src_sentence)
        self.covered_lprob = utils.NEG_INF
        
        it = 0
        self.beam_prob = 0.
        self.sample_beam = self.nbest
        hypos = [PartialHypothesis(self.get_predictor_states())]
        #old_hypos = []

        while not self._all_eos(hypos) and it < self.max_len:
            it += 1
            next_hypos = []
            next_scores = []
            for hypo in hypos:
                if hypo.get_last_word() == utils.EOS_ID:
                    next_hypos.append(hypo)
                    next_scores.append(self.get_adjusted_score(hypo))
                    continue 
                for next_hypo in self._expand_hypo(hypo, self.sample_beam):
                    next_scores.append(self.get_adjusted_score(next_hypo))
                    next_hypos.append(next_hypo)

            hypos = self._get_next_hypos(next_hypos, next_scores, seed=seed)
            #old_hypos.extend([h.score for h in next_hypos if h not in hypos])

        assert self.beam_prob <= 1
        return self.get_full_hypos_sorted(hypos)

    def _get_next_hypos(self, hypos, scores, seed=0):
        # faster to append to python list then convert to np array
        scores = np.array(scores)
        inds, cur_beam_prob, inc_probs = sampling_utils.log_sample_k_dpp(scores, 
                                                        self.sample_beam, seed=seed)
        self.beam_prob += cur_beam_prob
        for i in inds:
            hypos[i].base_score += min(0.,inc_probs[i])
        return [hypos[ind] for ind in inds]

    def _all_eos(self, hypos):
        """Returns true if the all hypotheses end with </S>"""
        return all([hypo.get_last_word() == utils.EOS_ID for hypo in hypos])


class PSworDecoder(CPSworDecoder):
    name = "p_swor"
    def __init__(self, decoder_args):
        """Creates a new SWOR decoder instance. The following values are
        fetched from `decoder_args`:
        
            nbest (int): number of desired samples. 
            temperature (float): temperature for shifting probability distributions
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(PSworDecoder, self).__init__(decoder_args)

    def decode(self, src_sentence, seed=0):
        self.initialize_predictor(src_sentence)
        desired_k = self.nbest#np.power(self.nbest, 1./self.max_len)
        
        it = 0
        hypos = [PartialHypothesis(self.get_predictor_states())]
        hypos[0].base_score = 1.
        while not self._all_eos(hypos) and it < self.max_len:
            it += 1
            next_hypos = []
            next_scores = []
            for hypo in hypos:
                if hypo.get_last_word() == utils.EOS_ID:
                    self.add_full_hypo(hypo.generate_full_hypothesis()) 
                    continue 
                expansions, dist = self._expand_hypo(hypo, return_dist=True)
                c = sampling_utils.get_const(dist + hypo.score, desired_k)
                c /= hypo.base_score
                c = np.power(c, 1./(self.max_len - len(hypo)))
                hypo.base_score *= c
                for next_hypo in expansions:
                    next_scores.append(next_hypo.score_breakdown[-1] + np.log(c))
                    next_hypos.append(next_hypo)

            hypos = self._get_next_hypos(next_hypos, next_scores, seed=seed)
            
        return self.get_full_hypos_sorted(hypos)
    

    def _get_next_hypos(self, hypos, scores, seed=0):
        # faster to append to python list then convert to np array
        scores = np.array(scores)
        inds, inc_probs = sampling_utils.log_sample_poisson(scores, 
                                                normalize=False, seed=seed)
        for i in inds:
            hypos[i].base_score += min(0.,inc_probs[i])
        return [hypos[ind] for ind in inds]

    def _all_eos(self, hypos):
        """Returns true if the all hypotheses end with </S>"""
        return all([hypo.get_last_word() == utils.EOS_ID for hypo in hypos])


class MapDist(object):

    def __init__(self):
        self.dist_map = {}

    def __contains__(self, key):
        return tuple(key) in self.dist_map

    def add_dist(self, prefix, ids, dist, states):
        self.dist_map[prefix] = Dist(ids, dist, states)

    def adjust(self, prefix, next_word, val):
        self.dist_map[prefix].adjust(next_word, val)

    def get(self, prefix):
        return self.dist_map[prefix].values()

    def marg(self, prefix):
        if not prefix[:-1] in self.dist_map:
            return 0
        return self.dist_map[prefix[:-1]].get_current(prefix[-1])


class Dist(object):

    def __init__(self, ids, lprobabilities, predictor_states):
        self.ids = ids
        self.lprobabilities = SumHeap(lprobabilities, log_space=True)
        self.adjustments = SumHeap(np.full_like(lprobabilities, utils.NEG_INF, dtype=np.float64), log_space=True)
        self.predictor_states = copy.deepcopy(predictor_states)
        self.adjusted_lprobabilities = SumHeap(lprobabilities, log_space=True)
    
    def get_current(self, k):
        ind = utils.binary_search(self.ids, k)
        return self.adjusted_lprobabilities[ind]

    def adjust(self, k, val):
        ind = utils.binary_search(self.ids, k)
        self.adjustments[ind] = utils.log_add(self.adjustments[ind], val)
        self.adjusted_lprobabilities[ind] = utils.log_minus(self.lprobabilities[ind], self.adjustments[ind])

    def values(self):
        return self.ids, self.lprobabilities, self.adjusted_lprobabilities, self.predictor_states


