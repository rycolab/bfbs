import copy
import logging
import utils
from datastructures.min_max_queue import MinMaxHeap
from decoding.core import Decoder, PartialHypothesis
from heapq import heappush, heappop


class DijkstraDecoder(Decoder):
    
    def __init__(self, decoder_args):
        """Creates a new A* decoder instance. The following values are
        fetched from `decoder_args`:
        
            beam (int): Maximum number of active hypotheses.
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
        super(DijkstraDecoder, self).__init__(decoder_args)
        self.nbest = max(1, decoder_args.nbest)
        self.use_lower_bound = not self.gumbel
        self.capacity = decoder_args.beam if not self.gumbel else self.nbest

    def decode(self, src_sentence):
        """Decodes a single source sentence using A* search. """
        self.lower_bound = self.get_empty_hypo(src_sentence) if self.use_lower_bound else None 
        self.initialize_predictors(src_sentence)
        self.cur_capacity = self.capacity
        open_set = MinMaxHeap(reserve=self.capacity) if self.capacity > 0 else []
        self.push(open_set, 0.0, PartialHypothesis(self.get_predictor_states()))

        while open_set:
            c,hypo = self.pop(open_set)#.popmin()
            logging.debug("Expand (est=%f score=%f exp=%d best=%f): sentence: %s"
                          % (-c, 
                             hypo.score, 
                             self.apply_predictors_count, 
                             self.lower_bound.score if self.lower_bound else utils.NEG_INF, 
                             hypo.trgt_sentence))
            if hypo.get_last_word() == utils.EOS_ID: # Found best hypothesis
                hypo.score = self.get_adjusted_score(hypo)
                self.add_full_hypo(hypo.generate_full_hypothesis())
                if len(self.full_hypos) == self.nbest: # if we have enough hypos
                    return self.get_full_hypos_sorted()
                self.cur_capacity -= 1
                continue

            if len(hypo) == self.max_len: #discard and continue
                continue

            for next_hypo in self._expand_hypo(hypo, self.capacity):
                score = self.get_adjusted_score(next_hypo)
                self.push(open_set, score, next_hypo)

        if not self.full_hypos:
            self.add_full_hypo(self.lower_bound.generate_full_hypothesis())
        return self.get_full_hypos_sorted()

    
    def push(self, set_, score, hypo):
        if self.lower_bound and score < self.lower_bound.score:
            return
        if isinstance(set_, MinMaxHeap):
            if set_.size < self.cur_capacity:
                set_.insert((-score, hypo))
            else:
                # only push if hypothesis can beat lower bound.
                min_score = -set_.peekmax()[0]
                if score > min_score:
                    set_.replacemax((-score, hypo))
        else:
            heappush(set_, (-score, hypo))

    
    def pop(self, set_):
        if isinstance(set_, MinMaxHeap):
           return set_.popmin()
        else:
            return heappop(set_)

