import numpy as np
import time
from collections import defaultdict

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder, PartialHypothesis


class BasicSworDecoder(Decoder):
    
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
        super(BasicSworDecoder, self).__init__(decoder_args)
        self.nbest = decoder_args.nbest
        self.temperature = decoder_args.temperature
        assert not self.gumbel
        
    def decode(self, src_sentence):

        self.ids = defaultdict(lambda: defaultdict(float))
        self.initialize_predictors(src_sentence)

        while len(self.full_hypos) < self.nbest:
            self.reset_predictors(src_sentence)
            hypo = PartialHypothesis(self.get_predictor_states())
            #self.set_predictor_states(hypo.predictor_states)
            while hypo.get_last_word() != utils.EOS_ID and len(hypo) < self.max_len:
                self._expand_hypo(hypo, seed=len(self.full_hypos))
            hypo.score = self.get_adjusted_score(hypo)
            self.add_full_hypo(hypo.generate_full_hypothesis())
    
        return self.full_hypos#self.get_full_hypos_sorted() 

    
    def _expand_hypo(self, hypo, seed=0):

        ids, posterior, _ = self.apply_predictors()
        probabilities = utils.softmax(posterior, temperature=self.temperature)
        adjusted_probabilities = self.adjust_probabilities(probabilities, hypo, ids)

        ind = self._sample(adjusted_probabilities, seed)
        next_word = ids[ind]

        hypo.score += np.log(adjusted_probabilities[ind])
        hypo.score_breakdown.append(probabilities[ind])
        hypo.trgt_sentence += [next_word]
        self.consume(next_word)


    def _sample(self, posterior, seed):
        np.random.seed(seed=seed)
        choices = range(len(posterior)) 
        return np.random.choice(choices, p=posterior)

    def add_full_hypo(self, hypo):
        for i in range(len(hypo)):
            prefix = tuple(hypo.trgt_sentence[:i])
            self.ids[prefix][hypo.trgt_sentence[i]] += utils.prod(hypo.score_breakdown[i:])
        super().add_full_hypo(hypo)

    def adjust_probabilities(self, probabilities, hypo, ids):
        
        if tuple(hypo.trgt_sentence) in self.ids:
            for k,v in self.ids[tuple(hypo.trgt_sentence)].items():
                ind = utils.binary_search(ids, k)
                probabilities[ind] -= v
        return probabilities/probabilities.sum()

    def reset_predictors(self, src_sentence):
        for idx, (p, _) in enumerate(self.predictors):
            p.set_current_sen_id(self.current_sen_id)
            p.initialize(src_sentence)
        for h in self.heuristics:
            h.initialize(src_sentence)


    
