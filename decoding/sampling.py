import copy
import time
import utils
import sampling_utils

import numpy as np
from bisect import bisect
from decoding.core import Decoder, PartialHypothesis


class SamplingDecoder(Decoder):

    name = "sampling"
    def __init__(self, decoder_args):
        super(SamplingDecoder, self).__init__(decoder_args)
        self.nbest = decoder_args.nbest
        assert not self.gumbel
        
    def decode(self, src_sentence, seed=0):
        self.initialize_predictor(src_sentence)
        hypos = [PartialHypothesis(copy.deepcopy(self.get_predictor_states())) for i in range(self.nbest)]

        t = 0
        while hypos and t < self.max_len:
            next_hypos = []
            for sen_seed, hypo in enumerate(hypos):
                if hypo.get_last_word() == utils.EOS_ID:
                    hypo.score = self.get_adjusted_score(hypo)
                    self.add_full_hypo(hypo.generate_full_hypothesis())
                else:
                    self._expand_hypo(hypo, seed=seed+sen_seed)
                    next_hypos.append(hypo)
            hypos = next_hypos
            t+=1

        for hypo in hypos:
            hypo.score = self.get_adjusted_score(hypo)
            self.add_full_hypo(hypo.generate_full_hypothesis())
                
        return self.get_full_hypos_sorted()

    
    def _expand_hypo(self, hypo, seed=0):

        self.set_predictor_states(hypo.predictor_states)
        ids, posterior, _ = self.apply_predictor()
        ind = self._sample(posterior, seed)
        next_word = ids[ind]

        hypo.predictor_states = self.get_predictor_states()
        hypo.score += posterior[ind]
        hypo.score_breakdown.append(posterior[ind])
        hypo.trgt_sentence += [next_word]
        self.consume(next_word)

    def _sample(self, posterior, seed):
        return sampling_utils.log_multinomial_sample(posterior, seed=seed)


class NucleusSamplingDecoder(SamplingDecoder):

    name = "nucleus_sampling"
    def __init__(self, decoder_args):
        
        super(NucleusSamplingDecoder, self).__init__(decoder_args)
        self.nucleus_threshold = decoder_args.nucleus_threshold

    def _sample(self, posterior, seed):
        self._truncate_log_dist(posterior, np.log(self.nucleus_threshold))
        return sampling_utils.log_multinomial_sample(posterior, seed=seed)

    @staticmethod
    def _truncate_log_dist(dist, threshold):
        """in-place method to truncate distribution to core elements 
        in `threshold` bound"""
        sorted_inds = np.argsort(-dist)
        sorted_dist = dist[sorted_inds]
        c = np.logaddexp.accumulate(sorted_dist) 
        last = bisect(c, threshold)
        dist[sorted_inds[last+1:]] = utils.NEG_INF
    
