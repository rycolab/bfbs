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
        ids, posterior, _ = self.apply_predictors()

        max_score = utils.max_(posterior)
        hypo.predictor_states = self.get_predictor_states()

        hypo.score += posterior[next_word] 
        hypo.score_breakdown.append(posterior[next_word])
        hypo.trgt_sentence += [next_word]
        self.consume(next_word)
                
