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

"""Contains all the basic interfaces and abstract classes for decoders.
The ``Decoder`` class provides common functionality for all decoders.
The ``Hypothesis`` class represents complete hypotheses, which are 
returned by decoders. ``PartialHypothesis`` is a helper class which can
be used by predictors to represent translation prefixes.
"""

from abc import abstractmethod
import copy
import math

from cam.sgnmt import utils
from cam.sgnmt.utils import Observable, Observer, MESSAGE_TYPE_DEFAULT, \
    MESSAGE_TYPE_POSTERIOR, MESSAGE_TYPE_FULL_HYPO, NEG_INF, EPS_P
import numpy as np
from operator import mul
import logging
from functools import reduce


class Hypothesis:
    """Complete translation hypotheses are represented by an instance
    of this class. We store the produced sentence, the combined score,
    and a score breakdown to the separate predictor scores.
    """
    
    def __init__(self, trgt_sentence, total_score, score_breakdown = [], base_score=0):
        """Creates a new full hypothesis.
        
        Args:
            trgt_sentence (list): List of target word ids without <S> 
                                  or </S> which make up the target 
                                  sentence
            total_score (float): combined total score of this hypo
            score_breakdown (list): Predictor score breakdown for each
                                    target token in ``trgt_sentence``
        """
        self.trgt_sentence = trgt_sentence
        self.total_score = total_score
        self.score_breakdown = score_breakdown
        if base_score:
            self.base_score = base_score

    def __repr__(self):
        """Returns a string representation of this hypothesis."""
        return "%s (%f)" % (' '.join(str(w) for w in self.trgt_sentence),
                            self.total_score)
    def __len__(self):
        return len(self.trgt_sentence)

    def __lt__(self, other):
        return self.total_score < other.total_score
    


class PartialHypothesis(object):
    """Represents a partial hypothesis in various decoders. """
    
    def __init__(self, initial_states = None):
        """Creates a new partial hypothesis with zero score and empty
        translation prefix.
        
        Args:
            initial_states: Initial predictor states
        """
        self.predictor_states = initial_states
        self.trgt_sentence = []
        self.score, self.base_score = 0.0, 0.0
        self.score_breakdown = []
        self.word_to_consume = None


    def __lt__(self, other):
        return len(self.trgt_sentence) < len(other.trgt_sentence)

    def __len__(self):
        return len(self.trgt_sentence)
    
    def get_last_word(self):
        """Get the last word in the translation prefix. """
        if not self.trgt_sentence:
            return None
        return self.trgt_sentence[-1]
        
    def cur_length(self):
        return len(self.trgt_sentence)
    
    def generate_full_hypothesis(self):
        """Create a ``Hypothesis`` instance from this hypothesis. """
        return Hypothesis(self.trgt_sentence, self.score, self.score_breakdown, self.base_score)
    
    def _new_partial_hypo(self, states, word, score, base_score=None, use_base=False):
        """Create a new partial hypothesis, setting its state, score
        translation prefix and score breakdown.
        Args:
            states (object): Predictor states for new hypo. May be state 
                             after consuming word or current state, depending
                             whether full or cheap expansion is used
            word (int): New word to add to prefix
            score (float): Word log probability to be added to score
            score_breakdown (list): Predictor score breakdown for
                                    the new word
        """
        new_hypo = PartialHypothesis(states)
        new_hypo.score = score if use_base else self.score + score
        new_hypo.base_score = self.base_score + base_score if use_base else None
        new_hypo.score_breakdown = copy.copy(self.score_breakdown)
        new_hypo.score_breakdown.append(base_score if use_base else score)
        new_hypo.trgt_sentence = self.trgt_sentence + [word]
        
        return new_hypo

    def expand(self, word, new_states, score, score_breakdown):
        """Creates a new partial hypothesis adding a new word to the
        translation prefix with given probability and updates the
        stored predictor states.
        
        Args:
            word (int): New word to add to the translation prefix
            new_states (object): Predictor states after consuming
                                 ``word``
            score (float): Word log probability which is to be added
                           to the total hypothesis score
            score_breakdown (list): Predictor score breakdown for
                                    the new word
        """
        return self._new_partial_hypo(new_states, word, score, score_breakdown)
    
    def cheap_expand(self, word, score, base_score=None):
        """Creates a new partial hypothesis adding a new word to the
        translation prefix with given probability. Does NOT update the
        predictor states but adds a flag which signals that the last 
        word in this hypothesis has not been consumed yet by the 
        predictors. This can save memory because we can reuse the 
        current state for many hypothesis. It also saves computation
        as we do not consume words which are then discarded anyway by
        the search procedure.
        
        Args:
            word (int): New word to add to the translation prefix
            score (float): Word log probability which is to be added
                           to the total hypothesis score
            score_breakdown (list): Predictor score breakdown for
                                    the new word
        """
        hypo = self._new_partial_hypo(self.predictor_states,
                                     int(word), float(score),
                                     base_score=base_score,
                                     use_base=base_score is not None)
        hypo.word_to_consume = int(word)
        return hypo


class Heuristic(Observer):
    """A ``Heuristic`` instance can be used to estimate the future 
    costs for a given word in a given state. See the ``heuristics``
    module for implementations."""
    
    def __init__(self):
        """Creates a heuristic without predictors. """
        super(Heuristic, self).__init__()
        self.predictors = []

    def set_predictors(self, predictors):
        """Set the predictors used by this heuristic. 
        
        Args:
            predictors (list):  Predictors and their weights to be
                                used with this heuristic. Should be in
                                the same form as ``Decoder.predictors``,
                                i.e. a list of (predictor, weight)
                                tuples
        """
        self.predictors = predictors
    
    def initialize(self, src_sentence):
        """Initialize the heuristic with the given source sentence.
        This is not passed through to the heuristic predictors
        automatically but handles initialization outside the
        predictors.
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S> which make up the source sentence
        """
        pass

    @abstractmethod
    def estimate_future_cost(self, hypo):
        """Estimate the future cost (i.e. negative score) given the 
        states of the predictors set by ``set_predictors`` for a
        partial hypothesis ``hypo``. Note that this function is not 
        supposed to change predictor states. If (e.g. for the greedy 
        heuristic) this is not possible, the predictor states must be
        changed back after execution by the implementing method.
        
        Args:
            hypo (PartialHypo): Hypothesis for which to estimate the
                                future cost
        
        Returns:
            float. The future cost estimate for this heuristic
        """
        raise NotImplementedError
    
    def notify(self, message, message_type = MESSAGE_TYPE_DEFAULT):
        """This is the notification method from the ``Observer``
        super class. We implement it with an empty method here, but
        implementing sub classes can override this method to get
        notifications from the decoder instance about generated
        posterior distributions.
        
        Args:
            message (object): The posterior sent by the decoder
        """
        pass
    
class Decoder(Observable):    
    """A ``Decoder`` instance represents a particular search strategy
    such as A*, beam search, greedy search etc. Decisions are made 
    based on the outputs of one or many predictors, which are 
    maintained by the ``Decoder`` instance.
    
    Decoders are observable. They fire notifications after 
    apply_predictors has been called. All heuristics
    are observing the decoder by default.
    """
    
    def __init__(self, decoder_args):
        """Initializes the decoder instance with no predictors or 
        heuristics.
        """
        super(Decoder, self).__init__()
        self.max_len_factor = decoder_args.max_len_factor
        self.predictors = [] # Tuples (predictor, weight)
        self.heuristics = []
        self.predictor_names = []
        self.gumbel = decoder_args.gumbel
        self.allow_unk_in_output = decoder_args.allow_unk_in_output
        self.nbest = 1 # length of n-best list
        self.combine_posteriors = self._combine_posteriors_simple
        self.current_sen_id = -1
        self.apply_predictors_count = 0
        self.temperature = decoder_args.temperature
    
    def add_predictor(self, name, predictor, weight=1.0):
        """Adds a predictor to the decoder. This means that this 
        predictor is going to be used to predict the next target word
        (see ``predict_next``)
        
        Args:
            name (string): Predictor name like 'nmt' or 'fst'
            predictor (Predictor): Predictor instance
            weight (float): Predictor weight
        """
        self.predictors.append((predictor, weight))
        self.predictor_names.append(name)
    
    def remove_predictors(self):
        """Removes all predictors of this decoder. """
        self.predictors = []
        self.predictor_names = []

    def set_heuristic_predictors(self, heuristic_predictors):
        """Define the list of predictors used by heuristics. This needs
        to be called before adding heuristics with ``add_heuristic()``

        Args:
            heuristic_predictors (list):  Predictors and their weights 
                                          to be used with heuristics. 
                                          Should be in the same form 
                                          as ``Decoder.predictors``,
                                          i.e. a list of 
                                          (predictor, weight) tuples
        """
        self.heuristic_predictors = heuristic_predictors
    
    def add_heuristic(self, heuristic):
        """Add a heuristic to the decoder. For future cost estimates,
        the sum of the estimates from all heuristics added so far will
        be used. The predictors used in this heuristic have to be set
        before via ``set_heuristic_predictors()``
        
        Args:
            heuristic (Heuristic): A heuristic to use for future cost
                                   estimates
        """
        heuristic.set_predictors(self.heuristic_predictors)
        self.add_observer(heuristic)
        self.heuristics.append(heuristic)
    
    def estimate_future_cost(self, hypo):
        """Uses all heuristics which have been added with 
        ``add_heuristic`` to estimate the future cost for a given
        partial hypothesis. The estimates are used in heuristic based
        searches like A*. This function returns the future log *cost* 
        (i.e. the lower the better), assuming that the last word in the
        partial hypothesis ``hypo`` is consumed next.
        
        Args:
            hypo (PartialHypothesis): Hypothesis for which to estimate
                                      the future cost given the current
                                      predictor state
        
        Returns
            float. Future cost
        """
        return sum([h.estimate_future_cost(hypo) for h in  self.heuristics])
    
    def has_predictors(self):
        """Returns true if predictors have been added to the decoder. """
        return len(self.predictors) > 0
    
    def consume(self, word, i=None):
        """Calls ``consume()`` on all predictors. """
        for (p, _) in self.predictors:
            p.consume(word) if i is None else p.consume(word, i) # May change predictor state
    
    def _get_non_zero_words(self, predictor, posterior):
        """Get the set of words from the predictor posteriors which 
        have non-zero probability. This set of words is then passed
        through to the open vocabulary predictors.

        This method assumes that both arguments are not empty.

        Args:
            bounded_predictor: predictor
            bounded_posterior: Corresponding posterior.

        Returns:
            Iterable with all words with non-zero probability.
        """

        fin_probs = np.isfinite(posterior)
        return [i for i, b in enumerate(fin_probs) if b]
    
    def apply_predictors(self, hypo=None, top_n=0):
        """Get the distribution over the next word by combining the
        predictor scores.

        Args:
            top_n (int): If positive, return only the best n words.
        
        Returns:
            combined,score_breakdown: Two dicts. ``combined`` maps 
            target word ids to the combined score, ``score_breakdown``
            contains the scores for each predictor separately 
            represented as tuples (unweighted_score, predictor_weight)
        """
        assert hypo is not None or not self.gumbel
        # only supports 1 predictor at the moment. Can change if need for more comes up
        assert len(self.predictors) == 1
        self.apply_predictors_count += 1
        predictor = self.predictors[0][0]
        # Get posteriors
        posterior = predictor.predict_next()
        posterior = utils.log_softmax(posterior, temperature=self.temperature)
        # numerical stability check
        assert len(posterior) - np.count_nonzero(posterior) <= 1
        
        non_zero_words = self._get_non_zero_words(predictor,
                                                  posterior)
        if len(non_zero_words) == 0: # Special case: no word is possible
            non_zero_words = set([utils.EOS_ID])

        if self.gumbel:
            gumbel_full_posterior = self.gumbelify(hypo, posterior)
            ids, posterior, original_posterior = self.combine_posteriors(
                non_zero_words, gumbel_full_posterior, predictor.get_unk_probability(posterior),
                top_n=top_n, original_posterior=posterior) 
        else:
            ids, posterior, original_posterior = self.combine_posteriors(
                non_zero_words, posterior, predictor.get_unk_probability(posterior), top_n=top_n) 
                
        assert self.allow_unk_in_output or not utils.UNK_ID in ids
        
        self.notify_observers(posterior, message_type = MESSAGE_TYPE_POSTERIOR)
        return ids, posterior, original_posterior


    def gumbelify(self, hypo, posterior):
        vf = np.vectorize(lambda x: self.get_pos_score(hypo, x) - self.get_adjusted_score(hypo))
        shifted_posterior = vf(posterior)
        shifted_posterior = utils.log_softmax(shifted_posterior)

        np.random.seed(seed=0)
        gumbels = np.random.gumbel(loc=0, scale=1, size=shifted_posterior.shape)
        gumbel_posterior = shifted_posterior + gumbels + hypo.base_score
        Z = np.max(gumbel_posterior)

        v = hypo.score - gumbel_posterior + utils.logmexp(gumbel_posterior - Z)
        gumbel_full_posterior = hypo.score - np.maximum(0, v) - utils.logpexp(-np.abs(v))

        # make sure invalid tokens still have neg inf log probability
        gumbel_full_posterior[(posterior == utils.NEG_INF).nonzero()] == utils.NEG_INF
        return gumbel_full_posterior

    
    def _expand_hypo(self, hypo, limit=0):
        """Get the best beam size expansions of ``hypo``.
        
        Args:
            hypo (PartialHypothesis): Hypothesis to expand
        
        Returns:
            list. List of child hypotheses
        """

        self.set_predictor_states(copy.deepcopy(hypo.predictor_states))
        if not hypo.word_to_consume is None: # Consume if cheap expand
            self.consume(hypo.word_to_consume)
            hypo.word_to_consume = None

        ids, posterior, original_posterior = self.apply_predictors(hypo, limit)
 
        hypo.predictor_states = self.get_predictor_states()
        new_hypos = [hypo.cheap_expand(
                        trgt_word,
                        posterior[idx],
                        base_score=original_posterior[idx] if self.gumbel else None
                        ) for idx, trgt_word in enumerate(ids)]
    
        return new_hypos


    def get_pos_score(self, hypo, val, max_=None):
        """Combines hypo score with future cost estimates.""" 
        return  hypo.score + val

    def get_adjusted_score(self, hypo):
        """Combines hypo score with penalties/rewards.""" 
        current_score =  hypo.score
        if self.gumbel:
            return current_score
        if getattr(self, 'reward_type', False): 
            factor =  min(self.l, len(hypo))
            current_score += self.reward_coef*factor
            if self.heuristics:
                if hypo.get_last_word() != utils.EOS_ID:
                        potential = max(self.l - len(hypo.trgt_sentence),0) 
                        current_score += self.reward_coef*potential
        elif getattr(self, 'heuristics', False):
            if hypo.get_last_word() != utils.EOS_ID:
                remaining = self.max_len - len(hypo.trgt_sentence) 
                current_score += self.lmbda*self.epsilon*remaining

        elif getattr(self, 'length_norm', False): 
            current_score /= len(hypo)

        return current_score

    
    def _combine_posteriors_simple(self,
                                      non_zero_words,
                                      posterior,
                                      unk_prob,
                                      top_n,
                                      original_posterior=None):
        """        
        Args:
            non_zero_words (set): All words with positive probability
            posteriors: Predictor posterior distributions calculated
                        with ``predict_next()``
            unk_probs: UNK probabilities of the predictors, calculated
                       with ``get_unk_probability``
        
        Returns:
            combined,score_breakdown: like in ``apply_predictors()``
        """
        if top_n > 0:
            non_zero_words = utils.argmax_n(posterior, top_n)

        scores_func = np.vectorize(lambda x: utils.common_get(posterior, x, unk_prob))
        scores = scores_func(non_zero_words)

        orig_scores = None
        if original_posterior is not None:
            scores_func = np.vectorize(lambda x: utils.common_get(original_posterior, x, unk_prob))
            orig_scores = scores_func(non_zero_words)

        return non_zero_words, scores, orig_scores

    
    def set_current_sen_id(self, sen_id):
        self.current_sen_id = sen_id - 1  # -1 because incremented in init()
            
    def initialize_predictors(self, src_sentence):
        """First, increases the sentence id counter and calls
        ``initialize()`` on all predictors. Then, ``initialize()`` is
        called for all heuristics.
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S> which make up the source sentence
        """
        self.max_len = int(np.ceil(self.max_len_factor * len(src_sentence)))
        self.full_hypos = []
        self.current_sen_id += 1
        for idx, (p, _) in enumerate(self.predictors):
            p.set_current_sen_id(self.current_sen_id)
            p.initialize(src_sentence)
        for h in self.heuristics:
            h.initialize(src_sentence)
    
    def add_full_hypo(self, hypo):
        """Adds a new full hypothesis to ``full_hypos``. This can be
        used by implementing subclasses to add a new hypothesis to the
        result set. This method also notifies observers.
        
        Args:
            hypo (Hypothesis): New complete hypothesis
        """
        if len(self.full_hypos) == 0 or hypo.total_score > self.cur_best.total_score:
            self.cur_best = hypo
        self.full_hypos.append(hypo)
        self.notify_observers(hypo, message_type = MESSAGE_TYPE_FULL_HYPO)
    
    def get_full_hypos_sorted(self):
        """Returns ``full_hypos`` sorted by the total score. Can be 
        used by implementing subclasses as return value of
        ``decode``
        
        Returns:
            list. ``full_hypos`` sorted by ``total_score``.
        """
        return sorted(self.full_hypos,
                      key=lambda hypo: hypo.total_score,
                      reverse=True)
    
    def get_lower_score_bound(self):
        """Intended to be called by implementing subclasses. Returns a
        lower bound on the best score of the current sentence. This is
        either read from the lower bounds file (if provided) or set to
        negative infinity.
        
        Returns:
            float. Lower bound on the best score for current sentence
        """ 
        if self.current_sen_id < len(self.lower_bounds):
            return self.lower_bounds[self.current_sen_id] - EPS_P
        return NEG_INF


    def get_empty_hypo(self, src_sentence):
        self.initialize_predictors(src_sentence)
        hypo = PartialHypothesis(self.get_predictor_states())
        ind = utils.EOS_ID
        ids, posterior, _ = self.apply_predictors()

        hypo.score += posterior[ind] 
        hypo.score_breakdown.append(posterior[ind])
        hypo.trgt_sentence += [ind]

        hypo.score = self.get_adjusted_score(hypo)
        return hypo

    
    def get_max_expansions(self, max_expansions_param, src_sentence):
        """This is a helper for decoders which support the 
        ``max_node_expansions`` parameter. It returns the maximum
        number of node expansions for the given sentence.
        
        Args:
            max_expansions_param (int): max_node_expansions parameter
                                        passed through from the config
            src_sentence (list): Current source sentence
        
        Returns:
            int. Maximum number of node expansions for this decoding
            task.
        """
        if max_expansions_param > 0:
            return max_expansions_param
        if max_expansions_param < 0:
            return -len(src_sentence) * max_expansions_param
        return 100000000  
    
    def set_predictor_states(self, states):
        """Calls ``set_state()`` on all predictors. """
        i = 0
        for (p, _) in self.predictors:
            p.set_state(states[i])
            i = i + 1
    
    def get_predictor_states(self, batch=False):
        """Calls ``get_state()`` on all predictors. """
        return [p.get_state() if not batch else p.get_states() for (p, _) in self.predictors]

    @staticmethod
    def _scale_combine_non_zero_scores(non_zero_word_count,
                                       posteriors,
                                       unk_probs,
                                       pred_weights,
                                       top_n=0):
      scaled_posteriors = []
      for posterior, unk_prob, weight in zip(
              posteriors, unk_probs, pred_weights):
          if isinstance(posterior, dict):
              arr = np.full(non_zero_word_count, unk_prob)
              for word, score in posterior.items():
                  if word < non_zero_word_count:
                      arr[word] = score
              scaled_posteriors.append(arr * weight)
          else:
              n_unks = non_zero_word_count - len(posterior)
              if n_unks > 0:
                  posterior = np.concatenate((
                      posterior, np.full(n_unks, unk_prob)))
              elif n_unks < 0:
                  posterior = posterior[:n_unks]
              scaled_posteriors.append(posterior * weight)
      combined_scores = np.sum(scaled_posteriors, axis=0)
      return utils.argmax_n(combined_scores, top_n)

    
    @abstractmethod
    def decode(self, src_sentence):
        """Decodes a single source sentence. This method has to be 
        implemented by subclasses. It contains the core of the 
        implemented search strategy ``src_sentence`` is a list of
        source word ids representing the source sentence without
        <S> or </S> symbols. This method returns a list of hypotheses,
        order descending by score such that the first entry is the best
        decoding result. Implementations should delegate the scoring of
        hypotheses to the predictors via ``apply_predictors()``, and
        organize predictor states with the methods ``consume()``,
        ``get_predictor_states()`` and ``set_predictor_states()``. In
        this way, the decoder is decoupled from the scoring modules.
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S> which make up the source sentence
        
        Returns:
            list. A list of ``Hypothesis`` instances ordered by their
            score.
        
        Raises:
            ``NotImplementedError``: if the method is not implemented
        """
        raise NotImplementedError

    def are_equal_predictor_states(self, states1, states2):
        """This method applies ``is_equal`` on all predictors. It 
        returns true if all predictor states are equal.
        
        Args:
            states1 (list): First predictor states as returned by
                            ``get_predictor_states``
            states2 (list): Second predictor states as returned by
                            ``get_predictor_states``
        
        Returns:
            boolean. True if all predictor states are equal, False
            otherwise 
        """
        i = 0
        for (p, _) in self.predictors:
            if not p.is_equal(states1[i], states2[i]):
                return False
            i = i + 1
        return True
    
