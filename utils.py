from abc import abstractmethod
import operator
import logging
import os
import sys
from bisect import bisect_left 
from functools import reduce  

import numpy as np
from numpy import log, log1p, exp, expm1, inf, nan
from scipy.special import logsumexp

# Reserved IDs
GO_ID = 1
"""Reserved word ID for the start-of-sentence symbol. """


EOS_ID = 2
"""Reserved word ID for the end-of-sentence symbol. """


UNK_ID = 0
"""Reserved word ID for the unknown word (UNK). """


NEG_INF = -inf


MACHINE_EPS = np.finfo(float).eps


LOG_MACHINE_EPS = np.log(MACHINE_EPS)


INF = inf


EPS_P = 0.00001


def switch_to_fairseq_indexing():
    """Calling this method overrides the global definitions of the 
    reserved  word ids ``GO_ID``, ``EOS_ID``, and ``UNK_ID``
    with the fairseq indexing scheme. 
    """
    global GO_ID
    global EOS_ID
    global UNK_ID
    GO_ID = 0
    EOS_ID = 2
    UNK_ID = 3


def switch_to_t2t_indexing():
    """Calling this method overrides the global definitions of the 
    reserved  word ids ``GO_ID``, ``EOS_ID``, and ``UNK_ID``
    with the tensor2tensor indexing scheme. This scheme is used in all
    t2t models. 
    """
    global GO_ID
    global EOS_ID
    global UNK_ID
    GO_ID = 2 # Usually not used
    EOS_ID = 1
    UNK_ID = 3 # Don't rely on this: UNK not standardized in T2T


# Log summation


def log_sum_tropical_semiring(vals):
    """Approximates summation in log space with the max.
    
    Args:
        vals  (set): List or set of numerical values
    """
    return max(vals)


def log_sum_log_semiring(vals):
    """Uses the ``logsumexp`` function in scipy to calculate the log of
    the sum of a set of log values.
    
    Args:
        vals  (set): List or set of numerical values
    """
    return logsumexp(np.asarray([val for val in vals]))


log_sum = log_sum_log_semiring
"""Defines which log summation function to use. """


def oov_to_unk(seq, vocab_size, unk_idx=None):
    if unk_idx is None:
        unk_idx = UNK_ID
    return [x if x < vocab_size else unk_idx for x in seq]

# Maximum functions

def argmax_n(arr, n):
    """Get indices of the ``n`` maximum entries in ``arr``. The 
    parameter ``arr`` can be a dictionary. The returned index set is 
    not guaranteed to be sorted.
    
    Args:
        arr (list,array,dict):  Set of numerical values
        n  (int):  Number of values to retrieve
    
    Returns:
        List of indices or keys of the ``n`` maximum entries in ``arr``
    """
    if isinstance(arr, dict):
        return sorted(arr, key=arr.get, reverse=True)[:n]
    elif len(arr) <= n:
        return range(len(arr))
    elif hasattr(arr, 'is_cuda') and arr.is_cuda:
        return np.argpartition(arr.cpu(), -n)[-n:]
    return np.argpartition(arr, -n)[-n:]


def max_(arr):
    """Get indices of the ``n`` maximum entries in ``arr``. The 
    parameter ``arr`` can be a dictionary. The returned index set is 
    not guaranteed to be sorted.
    
    Args:
        arr (list,array,dict):  Set of numerical values
        n  (int):  Number of values to retrieve
    
    Returns:
        List of indices or keys of the ``n`` maximum entries in ``arr``
    """
    if isinstance(arr, dict):
        return max(arr.values())
    if isinstance(arr, list):
        return max(arr)
    return np.max(arr)


def argmax(arr):
    """Get the index of the maximum entry in ``arr``. The parameter can
    be a dictionary.
    
    Args:
        arr (list,array,dict):  Set of numerical values
    
    Returns:
        Index or key of the maximum entry in ``arr``
    """
    if isinstance(arr, dict):
        return max(arr.items(), key=operator.itemgetter(1))[0]
    else:
        return np.argmax(arr)

def logmexp(x):
    return np.log1p(-np.exp(x))

def logpexp(x):
    return np.log1p(np.exp(x))


def logsigmoid(x):
    """
    log(sigmoid(x)) = -log(1+exp(-x)) = -log1pexp(-x)
    """
    return -log1pexp(-x)


def log1pexp(x):
    """
    Numerically stable implementation of log(1+exp(x)) aka softmax(0,x).

    -log1pexp(-x) is log(sigmoid(x))

    Source:
    http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    if x <= -37:
        return exp(x)
    elif -37 <= x <= 18:
        return log1p(exp(x))
    elif 18 < x <= 33.3:
        return x + exp(-x)
    else:
        return x


def log1mexp(x):
    """
    Numerically stable implementation of log(1-exp(x))

    Note: function is finite for x < 0.

    Source:
    http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    if x >= 0:
        return nan
    else:
        a = abs(x)
        if 0 < a <= 0.693:
            return log(-expm1(-a))
        else:
            return log1p(-exp(-a))


def log_add(x, y):
    # implementation: need separate checks for inf because inf-inf=nan.
    if x == NEG_INF:
        return y
    elif y == NEG_INF:
        return x
    else:
        if y <= x:
            d = y-x
            r = x
        else:
            d = x-y
            r = y
        return r + log1pexp(d)


def log_minus(x, y):
    if x == y:
        return NEG_INF
    if y > x:
        if y-x > MACHINE_EPS:
            logging.warn("Using function log_minus for invalid values")
        return nan
    else:
        return x + log1mexp(y-x)

vectorized_log_minus = np.vectorize(log_minus)

vectorized_log_add_eps = np.vectorize(lambda x: log_add(x, LOG_MACHINE_EPS))

def log_add_old(a, b):
    # takes two log probabilities; equivalent to adding probabilities in log space
    if a == NEG_INF or b == NEG_INF:
        return max(a, b)
    smaller = min(a,b)
    larger = max(a,b)
    return larger + log1pexp(smaller - larger)

def log_minus_old(a, b):
    # takes two log probabilities; equivalent to subtracting probabilities in log space
    assert b <= a
    if a == b:
        return NEG_INF
    if b == NEG_INF:
        return a
    comp = a + log1mexp(-(a-b))
    return comp if not np.isnan(comp) else NEG_INF


def softmax(x, temperature=1.):
    return np.exp(log_softmax(x, temperature=temperature))

def log_softmax(x, temperature=1.):
    x = x/temperature
    # numerically stable log softmax
    shift_x = x - np.max(x)
    # mask invalid values (neg inf)
    b = (~np.ma.masked_invalid(shift_x).mask).astype(int)
    return shift_x - logsumexp(shift_x, b=b)

  
def binary_search(a, x): 
    i = bisect_left(a, x) 
    if i != len(a) and a[i] == x: 
        return i 
    else: 
        return -1

def perplexity(arr):
    if len(arr) == 0:
        return INF
    score = sum([s for s in arr])
    return 2**(-score/len(arr))


def prod(iterable):
    return reduce(operator.mul, iterable, 1.0)

# Functions for common access to numpy arrays, lists, and dicts
def common_viewkeys(obj):
    """Can be used to iterate over the keys or indices of a mapping.
    Works with numpy arrays, lists, and dicts. Code taken from
    http://stackoverflow.com/questions/12325608/iterate-over-a-dict-or-list-in-python
    """
    if isinstance(obj, dict):
        return obj.keys()
    else:
        return range(len(obj))


def common_iterable(obj):
    """Can be used to iterate over the key-value pairs of a mapping.
    Works with numpy arrays, lists, and dicts. Code taken from
    http://stackoverflow.com/questions/12325608/iterate-over-a-dict-or-list-in-python
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key, value
    else:
        for index, value in enumerate(obj):
            yield index, value


def common_get(obj, key, default):
    """Can be used to access an element via the index or key.
    Works with numpy arrays, lists, and dicts.
    
    Args:
        ``obj`` (list,array,dict):  Mapping
        ``key`` (int): Index or key of the element to retrieve
        ``default`` (object): Default return value if ``key`` not found
    
    Returns:
        ``obj[key]`` if ``key`` in ``obj``, otherwise ``default``
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    else:
        return obj[key] if key < len(obj) else default


def common_contains(obj, key):
    """Checks the existence of a key or index in a mapping.
    Works with numpy arrays, lists, and dicts.
    
    Args:
        ``obj`` (list,array,dict):  Mapping
        ``key`` (int): Index or key of the element to retrieve
    
    Returns:
        ``True`` if ``key`` in ``obj``, otherwise ``False``
    """
    if isinstance(obj, dict):
        return key in obj
    else:
        return key < len(obj)


# Miscellaneous


def get_path(tmpl, sub = 1):
    """Replaces the %d placeholder in ``tmpl`` with ``sub``. If ``tmpl``
    does not contain %d, return ``tmpl`` unmodified.
    
    Args:
        tmpl (string): Path, potentially with %d placeholder
        sub (int): Substitution for %d
    
    Returns:
        string. ``tmpl`` with %d replaced with ``sub`` if present
    """
    try:
        return tmpl % sub
    except TypeError:
        pass
    return tmpl


def split_comma(s, func=None):
    """Splits a string at commas and removes blanks."""
    if not s:
        return []
    parts = s.split(",")
    if func is None:
        return [el.strip() for el in parts]
    return [func(el.strip()) for el in parts]


def ngrams(sen, n):
    sen = sen.split(' ')
    output = []
    for i in range(len(sen)-n+1):
        output.append(tuple(sen[i:i+n]))
    return output

def distinct_ngrams(hypos, n):
    total_ngrams = 0
    distinct = []
    for h in hypos:
        all_ngrams = ngrams(h, n)
        total_ngrams += len(all_ngrams)
        distinct.extend(all_ngrams)
    
    if len(distinct) == 0:
        return 0
    return float(len(set(distinct)))/len(distinct)

def ngram_diversity(hypos):
    ds = [distinct_ngrams(hypos, i) for i in range(1,5)]
    return sum(ds)/4


MESSAGE_TYPE_DEFAULT = 1
"""Default message type for observer messages """


MESSAGE_TYPE_POSTERIOR = 2
"""This message is sent by the decoder after ``apply_predictors`` was
called. The message includes the new posterior distribution and the
score breakdown. 
"""


MESSAGE_TYPE_FULL_HYPO = 3
"""This message type is used by the decoder when a new complete 
hypothesis was found. Note that this is not necessarily the best hypo
so far, it is just the latest hypo found which ends with EOS.
"""


class Observer(object):
    """Super class for classes which observe (GoF design patten) other
    classes.
    """
    
    @abstractmethod
    def notify(self, message, message_type = MESSAGE_TYPE_DEFAULT):
        """Get a notification from an observed object.
        
        Args:
            message (object): the message sent by observed object
            message_type (int): The type of the message. One of the
                                ``MESSAGE_TYPE_*`` variables
        """
        raise NotImplementedError
    

class Observable(object):
    """For the GoF design pattern observer """
    
    def __init__(self):
        """Initializes the list of observers with an empty list """
        self.observers = []
    
    def add_observer(self, observer):
        """Add a new observer which is notified when this class fires
        a notification
        
        Args:
            observer (Observer): the observer class to add
        """
        self.observers.append(observer)
    
    def notify_observers(self, message, message_type = MESSAGE_TYPE_DEFAULT):
        """Sends the given message to all registered observers.
        
        Args:
            message (object): The message to send
            message_type (int): The type of the message. One of the
                                ``MESSAGE_TYPE_*`` variables
        """
        for observer in self.observers:
            observer.notify(message, message_type)

## Taken from https://gist.github.com/kilian-gebhardt/6f1db877797d69fa1df6aa936feea607

class MinMaxHeap(object):
    """
    Implementation of a Min-max heap following Atkinson, Sack, Santoro, and
    Strothotte (1986): https://doi.org/10.1145/6617.6621
    """
    def __init__(self, reserve=0):
        self.a = [None] * reserve
        self.size = 0

    def __len__(self):
        return self.size

    def __iter__(self):
        return self

    def __list__(self):
        return self.a
        
    def __next__(self):
        try:
            return self.popmin()
        except AssertionError:
            raise StopIteration

    def insert(self, key):
        """
        Insert key into heap. Complexity: O(log(n))
        """
        if len(self.a) < self.size + 1:
            self.a.append(key)
        insert(self.a, key, self.size)
        self.size += 1

    def peekmin(self):
        """
        Get minimum element. Complexity: O(1)
        """
        return peekmin(self.a, self.size)

    def peekmax(self):
        """
        Get maximum element. Complexity: O(1)
        """
        return peekmax(self.a, self.size)

    def popmin(self):
        """
        Remove and return minimum element. Complexity: O(log(n))
        """
        m, self.size = removemin(self.a, self.size)
        self.a.pop(-1)
        return m

    def popmax(self):
        """
        Remove and return maximum element. Complexity: O(log(n))
        """
        m, self.size = removemax(self.a, self.size)
        self.a.pop(-1)
        return m

    def replacemax(self, val):
        """
        Remove and return maximum element. Complexity: O(log(n))
        """
        replacemax(self.a, self.size, val)
        


def level(i):
    return (i+1).bit_length() - 1


def trickledown(array, i, size):
    if level(i) % 2 == 0:  # min level
        trickledownmin(array, i, size)
    else:
        trickledownmax(array, i, size)


def trickledownmin(array, i, size):
    if size > i * 2 + 1:  # i has children
        m = i * 2 + 1
        if i * 2 + 2 < size and array[i*2+2] < array[m]:
            m = i*2+2
        child = True
        for j in range(i*4+3, min(i*4+7, size)):
            if array[j] < array[m]:
                m = j
                child = False

        if child:
            if array[m] < array[i]:
                array[i], array[m] = array[m], array[i]
        else:
            if array[m] < array[i]:
                if array[m] < array[i]:
                    array[m], array[i] = array[i], array[m]
                if array[m] > array[(m-1) // 2]:
                    array[m], array[(m-1)//2] = array[(m-1)//2], array[m]
                trickledownmin(array, m, size)


def trickledownmax(array, i, size):
    if size > i * 2 + 1:  # i has children
        m = i * 2 + 1
        if i * 2 + 2 < size and array[i*2+2] > array[m]:
            m = i*2+2
        child = True
        for j in range(i*4+3, min(i*4+7, size)):
            if array[j] > array[m]:
                m = j
                child = False

        if child:
            if array[m] > array[i]:
                array[i], array[m] = array[m], array[i]
        else:
            if array[m] > array[i]:
                if array[m] > array[i]:
                    array[m], array[i] = array[i], array[m]
                if array[m] < array[(m-1) // 2]:
                    array[m], array[(m-1)//2] = array[(m-1)//2], array[m]
                trickledownmax(array, m, size)


def bubbleup(array, i):
    if level(i) % 2 == 0:  # min level
        if i > 0 and array[i] > array[(i-1) // 2]:
            array[i], array[(i-1) // 2] = array[(i-1)//2], array[i]
            bubbleupmax(array, (i-1)//2)
        else:
            bubbleupmin(array, i)
    else:  # max level
        if i > 0 and array[i] < array[(i-1) // 2]:
            array[i], array[(i-1) // 2] = array[(i-1) // 2], array[i]
            bubbleupmin(array, (i-1)//2)
        else:
            bubbleupmax(array, i)


def bubbleupmin(array, i):
    while i > 2:
        if array[i] < array[(i-3) // 4]:
            array[i], array[(i-3) // 4] = array[(i-3) // 4], array[i]
            i = (i-3) // 4
        else:
            return


def bubbleupmax(array, i):
    while i > 2:
        if array[i] > array[(i-3) // 4]:
            array[i], array[(i-3) // 4] = array[(i-3) // 4], array[i]
            i = (i-3) // 4
        else:
            return


def peekmin(array, size):
    assert size > 0
    return array[0]


def peekmax(array, size):
    assert size > 0
    if size == 1:
        return array[0]
    elif size == 2:
        return array[1]
    else:
        return max(array[1], array[2])


def removemin(array, size):
    assert size > 0
    elem = array[0]
    array[0] = array[size-1]
    # array = array[:-1]
    trickledown(array, 0, size - 1)
    return elem, size-1


def removemax(array, size):
    assert size > 0
    if size == 1:
        return array[0], size - 1
    elif size == 2:
        return array[1], size - 1
    else:
        i = 1 if array[1] > array[2] else 2
        elem = array[i]
        array[i] = array[size-1]
        # array = array[:-1]
        trickledown(array, i, size - 1)
        return elem, size-1

def replacemax(array, size, val):
    assert size > 0
    if size == 1:
        array[0] = val
    elif size == 2:
        array[1] = val
        bubbleup(array, 1)
    else:
        i = 1 if array[1] > array[2] else 2
        array[i] = array[size-1]
        trickledown(array, i, size)
        array[size-1] = val
        bubbleup(array, size-1)


def insert(array, k, size):
    array[size] = k
    bubbleup(array, size)


def minmaxheapproperty(array, size):
    for i, k in enumerate(array[:size]):
        if level(i) % 2 == 0:  # min level
            # check children to be larger
            for j in range(2 * i + 1, min(2 * i + 3, size)):
                if array[j] < k:
                    print(array, j, i, array[j], array[i], level(i))
                    return False
            # check grand children to be larger
            for j in range(4 * i + 3, min(4 * i + 7, size)):
                if array[j] < k:
                    print(array, j, i, array[j], array[i], level(i))
                    return False
        else:
            # check children to be smaller
            for j in range(2 * i + 1, min(2 * i + 3, size)):
                if array[j] > k:
                    print(array, j, i, array[j], array[i], level(i))
                    return False
            # check grand children to be smaller
            for j in range(4 * i + 3, min(4 * i + 7, size)):
                if array[j] > k:
                    print(array, j, i, array[j], array[i], level(i))
                    return False

    return True


def test(n):
    from random import randint
    a = [-1] * n
    l = []
    size = 0
    for _ in range(n):
        x = randint(0, 5 * n)
        insert(a, x, size)
        size += 1
        l.append(x)
        assert minmaxheapproperty(a, size)

    assert size == len(l)
    print(a)

    while size > 0:
        assert min(l) == peekmin(a, size)
        assert max(l) == peekmax(a, size)
        if randint(0, 1):
            e, size = removemin(a, size)
            assert e == min(l)
        else:
            e, size = removemax(a, size)
            assert e == max(l)
        l[l.index(e)] = l[-1]
        l.pop(-1)
        assert len(a[:size]) == len(l)
        assert minmaxheapproperty(a, size)

    print("OK")


def test_heap(n):
    from random import randint
    heap = MinMaxHeap(n)
    l = []
    for _ in range(n):
        x = randint(0, 5 * n)
        heap.insert(x)
        l.append(x)
        assert minmaxheapproperty(heap.a, len(heap))

    assert len(heap) == len(l)
    print(heap.a)

    while len(heap) > 0:
        assert min(l) == heap.peekmin()
        assert max(l) == heap.peekmax()
        if randint(0, 1):
            e = heap.popmin()
            assert e == min(l)
        else:
            e = heap.popmax()
            assert e == max(l)
        l[l.index(e)] = l[-1]
        l.pop(-1)
        assert len(heap) == len(l)
        assert minmaxheapproperty(heap.a, len(heap))

    print("OK")
    
    
