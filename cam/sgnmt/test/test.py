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

"""This module is the bridge between the command line configuration of
the decode.py script and the SGNMT software architecture consisting of
decoders, predictors, and output handlers. A common use case is to call
`create_decoder()` first, which reads the SGNMT configuration and loads
the right predictors and decoding strategy with the right arguments.
The actual decoding is implemented in `do_decode()`. See `decode.py`
to learn how to use this module.
"""

import logging
import codecs
import sys
import time
import traceback
import os
import uuid

from cam.sgnmt import utils
from cam.sgnmt.decoding.astar import AstarDecoder
from cam.sgnmt.decoding.beam import BeamDecoder
from cam.sgnmt.decoding.core import Hypothesis
from cam.sgnmt.decoding.dijkstra import DijkstraDecoder
from cam.sgnmt.decoding.dijkstra_time_sync import DijkstraTSDecoder
from cam.sgnmt.decoding.reference import ReferenceDecoder
from cam.sgnmt.decoding.sampling import SamplingDecoder
from cam.sgnmt.decoding.dfs import DFSDecoder, \
                                   SimpleDFSDecoder, \
                                   SimpleLengthDFSDecoder
from cam.sgnmt.decoding.greedy import GreedyDecoder
from cam.sgnmt.decoding.swor import BasicSworDecoder, MemEfficientSworDecoder
from cam.sgnmt.output import TextOutputHandler, \
                             NBestOutputHandler, \
                             NBestSeparateOutputHandler, \
                             NgramOutputHandler, \
                             TimeCSVOutputHandler, \
                             FSTOutputHandler, \
                             StandardFSTOutputHandler, \
                             ScoreOutputHandler

from cam.sgnmt.test.dummy_predictor import DummyPredictor
from cam.sgnmt.ui import get_args

args = None
"""This variable is set to the global configuration when 
base_init().
"""

def base_init(new_args):
    """This function should be called before accessing any other
    function in this module. It initializes the `args` variable on 
    which all the create_* factory functions rely on as configuration
    object, and it sets up global function pointers and variables for
    basic things like the indexing scheme, logging verbosity, etc.

    Args:
        new_args: Configuration object from the argument parser.
    """
    global args
    args = new_args
    # UTF-8 support
    if sys.version_info < (3, 0):
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
        logging.warn("SGNMT is tested with Python 3, but you are using "
                     "Python 2. Expect the unexpected or switch to >3.5.")
    # Set up logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
    logging.getLogger().setLevel(logging.INFO)
    if args.verbosity == 'debug':
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbosity == 'info':
        logging.getLogger().setLevel(logging.INFO)
    elif args.verbosity == 'warn':
        logging.getLogger().setLevel(logging.WARN)
    elif args.verbosity == 'error':
        logging.getLogger().setLevel(logging.ERROR)
    # Set reserved word IDs
    utils.switch_to_fairseq_indexing()
    args.max_len_factor=10
    

def add_predictors(decoder):
    """Adds all enabled predictors to the ``decoder``. This function 
    makes heavy use of the global ``args`` which contains the
    SGNMT configuration. Particularly, it reads out ``args.predictors``
    and adds appropriate instances to ``decoder``.
    TODO: Refactor this method as it is waaaay tooooo looong
    
    Args:
        decoder (Decoder):  Decoding strategy, see ``create_decoder()``.
            This method will add predictors to this instance with
            ``add_predictor()``
    """
    
    pred_weight = 1.0
    p = DummyPredictor(vocab_size=10)
    decoder.add_predictor("dummy", p, pred_weight)

def create_decoder():
    """Creates the ``Decoder`` instance. This specifies the search 
    strategy used to traverse the space spanned by the predictors. This
    method relies on the global ``args`` variable.
    
    TODO: Refactor to avoid long argument lists
    
    Returns:
        Decoder. Instance of the search strategy
    """
    # Create decoder instance and add predictors
    decoder = None
    try:
        if args.decoder == "greedy":
            decoder = GreedyDecoder(args)
        elif args.decoder == "beam":
            decoder = BeamDecoder(args)
        elif args.decoder == "dfs":
            decoder = DFSDecoder(args)
        elif args.decoder == "simpledfs":
            decoder = SimpleDFSDecoder(args)
        elif args.decoder == "simplelendfs":
            decoder = SimpleLengthDFSDecoder(args)
        elif args.decoder == "astar":
            decoder = AstarDecoder(args)
        elif args.decoder == "dijkstra":
            decoder = DijkstraDecoder(args)
        elif args.decoder == "dijkstra_ts":
            decoder = DijkstraTSDecoder(args)
        elif args.decoder == "reference":
            decoder = ReferenceDecoder(args)
        elif args.decoder == "sampling":
            decoder = SamplingDecoder(args)
        elif args.decoder == "basic_swor":
            decoder = BasicSworDecoder(args)
        elif args.decoder == "mem_swor":
            decoder = MemEfficientSworDecoder(args)
        else:
            logging.fatal("Decoder %s not available. Please double-check the "
                          "--decoder parameter." % args.decoder)
    except Exception as e:
        logging.fatal("An %s has occurred while initializing the decoder: %s"
                      " Stack trace: %s" % (sys.exc_info()[0],
                                            e,
                                            traceback.format_exc()))
    if decoder is None:
        sys.exit("Could not initialize decoder.")
    add_predictors(decoder)
    return decoder


def _generate_dummy_hypo(predictors):
    return Hypothesis([utils.UNK_ID], 0.0, [0.0]) 


def do_decode(decoder, 
              output_handlers, 
              src_sentences,
              trgt_sentences=None,
              num_log=1):
    """This method contains the main decoding loop. It iterates through
    ``src_sentences`` and applies ``decoder.decode()`` to each of them.
    At the end, it calls the output handlers to create output files.
    
    Args:
        decoder (Decoder):  Current decoder instance
        output_handlers (list):  List of output handlers, see
                                 ``create_output_handlers()``
        src_sentences (list):  A list of strings. The strings are the
                               source sentences with word indices to 
                               translate (e.g. '1 123 432 2')
    """
    if not decoder.has_predictors():
        logging.fatal("Terminated due to an error in the "
                      "predictor configuration.")
        return
    all_hypos = []
    
    start_time = time.time()
    logging.info("Start time: %s" % start_time)
    sen_indices = []
    diversity_metrics = []
    not_full = 0
    src_sentences = list(range(10))
    for sen_idx, src in enumerate(src_sentences):
        decoder.set_current_sen_id(sen_idx)
        logging.info("Next sentence (ID: %d): %s" % (sen_idx + 1, str(src)))
        decoder.apply_predictors_count = 0
        start_hypo_time = time.time()
        hypos = decoder.decode([src])
        if not hypos:
            logging.error("No translation found for ID %d!" % (sen_idx+1))
            logging.info("Stats (ID: %d): score=<not-found> "
                     "num_expansions=%d "
                     "time=%.2f" % (sen_idx+1,
                                    decoder.apply_predictors_count,
                                    time.time() - start_hypo_time))
            hypos = [_generate_dummy_hypo(decoder.predictors)]
        
        for logged_hypo in hypos[:num_log]:
            logging.info("Decoded (ID: %d): %s" % (
                    sen_idx+1,
                    logged_hypo.trgt_sentence))
            logging.info("Stats (ID: %d): score=%f "
                         "num_expansions=%d "
                         "time=%.2f " 
                         "perplexity=%.2f"% (sen_idx+1,
                                        logged_hypo.total_score,
                                        decoder.apply_predictors_count,
                                        time.time() - start_hypo_time,
                                        utils.perplexity(logged_hypo.score_breakdown)))

           

args = get_args()
base_init(args)
decoder = create_decoder()
do_decode(decoder, [], False, num_log=args.num_log)

