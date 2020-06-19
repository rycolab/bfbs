import copy
import numpy as np
import time

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder, PartialHypothesis


class SamplingDecoder(Decoder):
    
    def __init__(self, decoder_args):
        """Creates a new A* decoder instance. The following values are
        fetched from `decoder_args`:
        
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
        self.nbest = decoder_args.nbest
        assert not self.gumbel
        
    def decode(self, src_sentence):
        self.initialize_predictors(src_sentence)
        hypos = [PartialHypothesis(copy.deepcopy(self.get_predictor_states())) for i in range(self.nbest)]

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
        ids, posterior, _ = self.apply_predictors()
        probabilites = utils.softmax(posterior)
        ind = self._sample(probabilites, seed)
        next_word = ids[ind]

        hypo.predictor_states = self.get_predictor_states()
        hypo.score += posterior[ind]
        hypo.score_breakdown.append(posterior[ind])
        hypo.trgt_sentence += [next_word]
        self.consume(next_word)


    def _sample(self, posterior, seed):
        np.random.seed(seed=seed)
        choices = range(len(posterior)) 
        return np.random.choice(choices, p=posterior)
    
