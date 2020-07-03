import logging
from sortedcontainers import SortedDict
from collections import defaultdict
import time

import utils
from datastructures.min_max_queue import MinMaxHeap
from decoding.core import Decoder, PartialHypothesis


class DijkstraTSDecoder(Decoder):
    
    def __init__(self, decoder_args):
        """Creates a new A* decoder instance. The following values are
        fetched from `decoder_args`:
        
            beam (int): beam width.
            early_stopping (bool): If this is true, partial hypotheses
                                   with score worse than the current
                                   best complete scores are not
                                   expanded. This applies when nbest is
                                   larger than one and inadmissible
                                   heuristics are used
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
        super(DijkstraTSDecoder, self).__init__(decoder_args)
        self.nbest = max(1, decoder_args.nbest)
        self.beam = decoder_args.beam if not self.gumbel else self.nbest
        self.early_stopping = decoder_args.early_stopping

        self.size_threshold = self.beam*decoder_args.memory_threshold_coef\
            if decoder_args.memory_threshold_coef > 0 else utils.INF

        
    def decode(self, src_sentence):
        self.initialize_predictors(src_sentence)
        self.initialize_order_ds() 
        self.total_queue_size = 0
        
        while self.queue_order:
            c,t = self.get_next()
            cur_queue = self.queues[t]
            score, hypo = cur_queue.popmin() 
            self.total_queue_size -= 1
            self.time_sync[t] -= 1

            if hypo.get_last_word() == utils.EOS_ID:
                hypo.score = self.get_adjusted_score(hypo)
                self.add_full_hypo(hypo.generate_full_hypothesis())
                if self.stop(): # if stopping criterion are met
                    break
                self.update(cur_queue, t, forward_prune=True)
                continue

            if t == self.max_len:
                self.update(cur_queue, t)
                continue

            next_queue = self.queues[t+1]            
            for next_hypo in self._expand_hypo(hypo, self.beam):
                self.add_hypo(next_hypo, next_queue, t+1)
                
            self.update(cur_queue, t)
            self.update(next_queue, t+1)
        
        return self.get_full_hypos_sorted()

    def initialize_order_ds(self):
        self.queues = [MinMaxHeap() for k in range(self.max_len+1)]
        self.queues[0].insert((0.0, PartialHypothesis(self.get_predictor_states())))
        self.queue_order = SortedDict({0.0: 0})
        self.score_by_t = [0.0]
        self.score_by_t.extend([None]*self.max_len)
        self.time_sync = defaultdict(lambda: self.beam if self.beam > 0 else utils.INF)
        self.time_sync[0] = 1

    def get_next(self):
        return self.queue_order.popitem()
        
    def update(self, queue, t, forward_prune=False):
        # remove current best value associated with queue
        self.queue_order.pop(self.score_by_t[t], default=None)

        #if beam used up at current time step, can prune hypotheses from older time steps
        if self.time_sync[t] <= 0:
            self.prune(t)

        #replace with next best value if anything left in queue
        if len(queue) > 0:
            self.queue_order[-queue.peekmin()[0]] = t
            self.score_by_t[t] = -queue.peekmin()[0]

        # if previous hypothesis was complete, reduce beam in next time steps
        if forward_prune:
            i = self.max_len
            while i > t:
                self.time_sync[i] -= 1
                if self.time_sync[i] <= 0:
                    self.prune(i)
                    return
                while len(self.queues[i]) > self.time_sync[i]:
                    # remove largest element since beam is getting "smaller"
                    self.queues[i].popmax()
                i-=1
    
    def prune(self, t):
        for i in range(t+1):
            self.queue_order.pop(self.score_by_t[i], default=None)
            self.queues[i] = []
    
    def add_hypo(self, hypo, queue, t):
        score = self.get_adjusted_score(hypo)
        if score == utils.NEG_INF:
            return
        if len(queue) < self.time_sync[t]:
            queue.insert((-score, hypo))
            if self.total_queue_size >= self.size_threshold:
                self.remove_one()
            else:
                self.total_queue_size += 1
        else:
            min_score = -queue.peekmax()[0]
            if score > min_score:
                queue.replacemax((-score, hypo))
        

    def remove_one(self):
        """ helper function for memory threshold"""
        for t, q in enumerate(self.queues):
            if len(q) > 0:
                q.popmax()
                if len(q) == 0:
                    self.queue_order.pop(self.score_by_t[t], default=None)
                return

    def stop(self):
        if not self.not_monotonic:
            return len(self.full_hypos) == (min(self.beam, self.nbest) if self.early_stopping else self.beam)
        
        if not self.early_stopping and len(self.full_hypos) < self.beam:
            return False
        threshold = max(self.full_hypos)  
        if not self.early_stopping:
            threshold = sorted(self.full_hypos, reverse=True)[self.beam]
        if all([threshold.total_score > self.max_pos_score(h[1]) for q in self.queues if q for h in q.a]):
            return True
        return False

    
        
