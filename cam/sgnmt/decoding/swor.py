import numpy as np
import time
import copy
from collections import defaultdict

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder, PartialHypothesis


class BasicSworDecoder(Decoder):
    
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
        self.temperature = decoder_args.temperature
        assert not self.gumbel
        
    def decode(self, src_sentence):
        self.src_sentence = src_sentence
        self.initialize_predictors(self.src_sentence)

        while len(self.full_hypos) < self.nbest and self.samples_left():
            self.reset_predictors(src_sentence)
            hypo = PartialHypothesis(self.get_predictor_states())
            self.start = True
            while hypo.get_last_word() != utils.EOS_ID and len(hypo) < self.max_len:
                self._expand_hypo(hypo, seed=len(self.full_hypos))
            self._add_full_hypo(hypo.generate_full_hypothesis())

        prob = sum([np.exp(sum(h.score_breakdown)) for h in self.full_hypos])
        print(len(self.full_hypos), "sentences covering", prob, "probability")
        return self.full_hypos

    
    def _expand_hypo(self, hypo, seed=0):
        hash_rep = tuple(hypo.trgt_sentence)
        if hash_rep in self.dists:
            ids, lprobabilities, adjusted_lprobabilities = self.dists[hash_rep].values()
            hypo.predictor_states = self.dists[hash_rep].predictor_states
        else:
            if self.start:
                # prefix has no longer previously been seen. One deep copy to get started
                hypo.predictor_states = copy.deepcopy(hypo.predictor_states)
                self.set_predictor_states(hypo.predictor_states)
                self.start = False
            if hypo.word_to_consume is not None:
                self.consume(hypo.word_to_consume)
                hypo.word_to_consume = None
            ids, posterior, _ = self.apply_predictors()
            lprobabilities = adjusted_lprobabilities = utils.log_softmax(posterior, self.temperature)
            # assert not np.any(np.isnan(lprobabilities))
            self.dists[hash_rep] = Dist(ids, lprobabilities, self.get_predictor_states())

        ind = utils.gumbel_max_sample(adjusted_lprobabilities, seed)
        next_word = ids[ind]

        hypo.score += adjusted_lprobabilities[ind]
        hypo.score_breakdown.append(lprobabilities[ind])
        hypo.trgt_sentence += [next_word]
        hypo.word_to_consume = next_word
       

    def _add_full_hypo(self, hypo):
        assert len(hypo.trgt_sentence) == len(hypo.score_breakdown)
        super().add_full_hypo(hypo)
        for i in range(len(hypo)):
            hash_rep = tuple(hypo.trgt_sentence[:i])
            self.dists[hash_rep].adjust(hypo.trgt_sentence[i], sum(hypo.score_breakdown[i:]))
            
        
    def reset_predictors(self, src_sentence):
        for idx, (p, _) in enumerate(self.predictors):
            p.set_current_sen_id(self.current_sen_id)
            p.initialize(src_sentence)
        for h in self.heuristics:
            h.initialize(src_sentence)

    def initialize_predictors(self, hypo):
        self.dists = {}
        super().initialize_predictors(hypo)

    def samples_left(self):
        if len(self.full_hypos) == 0:
            return True
        start_hash = tuple()
        _, _, adjusted_lprobabilities = self.dists[start_hash].values()
        return np.any(adjusted_lprobabilities > utils.NEG_INF)


class MemEfficientSworDecoder(BasicSworDecoder):
    
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

        ids, posterior, _ = self.apply_predictors()
        lprobabilities = utils.log_softmax(posterior, self.temperature)
        adjusted_lprobabilities = self.adjust_probabilities(lprobabilities, hypo, ids)

        ind = utils.gumbel_max_sample(adjusted_lprobabilities, seed)
        next_word = ids[ind]

        hypo.score += adjusted_lprobabilities[ind]
        hypo.score_breakdown.append(lprobabilities[ind])
        hypo.trgt_sentence += [next_word]
        self.consume(next_word)


    def _add_full_hypo(self, hypo):
        assert len(hypo.trgt_sentence) == len(hypo.score_breakdown)
        super().add_full_hypo(hypo)
        for i in range(len(hypo)):
            prefix = tuple(hypo.trgt_sentence[:i])
            val = sum(hypo.score_breakdown[i:])
            self.ids[prefix][hypo.trgt_sentence[i]] = utils.log_add(val, self.ids[prefix][hypo.trgt_sentence[i]])
        

    def adjust_probabilities(self, lprobabilities, hypo, ids):
        lprobabilities = np.copy(lprobabilities)
        hash_rep = tuple(hypo.trgt_sentence)
        for k, val in self.ids[hash_rep].items():
            ind = utils.binary_search(ids, k)
            lprobabilities[ind] = utils.log_minus(lprobabilities[ind], val)
        return lprobabilities

    def initialize_predictors(self, hypo):
        self.ids = defaultdict(lambda: defaultdict(lambda: utils.NEG_INF))
        super().initialize_predictors(hypo)

    def samples_left(self):
        if len(self.full_hypos) == 0:
            return True
        self.reset_predictors(self.src_sentence)
        ids, posterior, _ = self.apply_predictors()
        empty_hypo = PartialHypothesis()
        lprobabilities = utils.log_softmax(posterior, self.temperature)
        adjusted_lprobabilities = self.adjust_probabilities(lprobabilities, empty_hypo, ids)
        return np.any(adjusted_lprobabilities > utils.NEG_INF)


class Dist(object):

    def __init__(self, ids, lprobabilities, predictor_states):
        self.ids = ids
        self.lprobabilities = lprobabilities
        self.adjustments = np.full_like(lprobabilities, utils.NEG_INF, dtype=np.float64)
        #self.adjusted_lprobabilities = np.copy(lprobabilities)
        self.predictor_states = copy.deepcopy(predictor_states)

    def adjust(self, k, val):
        ind = utils.binary_search(self.ids, k)
        self.adjustments[ind] = utils.log_add(self.adjustments[ind], val)
        #self.adjusted_lprobabilities[ind] = utils.log_minus(self.adjusted_lprobabilities[ind], val)        
        
    def values(self):
        adjusted_lprobabilities = utils.vectorized_log_minus(self.lprobabilities, self.adjustments)
        return self.ids, self.lprobabilities, adjusted_lprobabilities

    
