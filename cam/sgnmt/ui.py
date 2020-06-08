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

"""This module handles the user interface and contains subroutines
for parsing and verifying config files and command line arguments.
"""

import argparse
import logging
import os
import sys
import platform

from cam.sgnmt import utils
from cam import sgnmt

YAML_AVAILABLE = True
try:
    import yaml
except:
    YAML_AVAILABLE = False


def str2bool(v):
    """For making the ``ArgumentParser`` understand boolean values"""
    return v.lower() in ("yes", "true", "t", "1")


def run_diagnostics():
    """Check availability of external libraries."""
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    print("Checking SGNMT version.... %s%s%s" 
          % (OKGREEN, sgnmt.__version__, ENDC))
    if sys.version_info > (3, 0):
        print("Checking Python3.... %sOK (%s)%s" 
              % (OKGREEN, platform.python_version(), ENDC))
    else:
        print("Checking Python3.... %sNOT FOUND %s%s"
              % (FAIL, sys.version_info, ENDC))
        print("Please upgrade to Python 3!")
    if YAML_AVAILABLE:
        print("Checking PyYAML.... %sOK%s" % (OKGREEN, ENDC))
    else:
        print("Checking PyYAML.... %sNOT FOUND%s" % (FAIL, ENDC))
        logging.info("NOTOK: PyYAML is not available. That means that "
                     "--config_file cannot be used. Check the documentation "
                     "for further instructions.")
    try:
        import torch
        print("Checking PyTorch.... %sOK (%s)%s"
              % (OKGREEN, torch.__version__, ENDC))
    except ImportError:
        print("Checking PyTorch.... %sNOT FOUND%s" % (FAIL, ENDC))
        print("PyTorch is not available. This affects the following "
              "components: Predictors: fairseq, onmtpy. Check the "
              "documentation for further instructions.")
    try:
        import fairseq
        print("Checking fairseq.... %sOK (%s)%s"
              % (OKGREEN, fairseq.__version__, ENDC))
    except ImportError:
        print("Checking fairseq.... %sNOT FOUND%s" % (FAIL, ENDC))
        print("fairseq is not available. This affects the following "
              "components: Predictors: fairseq. Check the "
              "documentation for further instructions.")


def parse_args(parser):
    """http://codereview.stackexchange.com/questions/79008/parse-a-config-file-
    and-add-to-command-line-arguments-using-argparse-in-python """
    args = parser.parse_args()
    if args.config_file:
        if not YAML_AVAILABLE:
            logging.fatal("Install PyYAML in order to use config files.")
            return args
        paths = args.config_file
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        for path in utils.split_comma(paths):
            _load_config_file(arg_dict, path)
    return args


def _load_config_file(arg_dict, path):
    with open(path.strip()) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        for key, value in data.items():
            if key == "config_file":
                for sub_path in value.split(","):
                    _load_config_file(arg_dict, sub_path)
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value


def parse_param_string(param):
    """Parses a parameter string such as 'param1=x,param2=y'. Loads 
    config files if specified in the string. If ``param`` points to a
    file, load this file with YAML.
    """
    if not param:
        return {}
    if os.path.isfile(param):
        param = "config_file=%s" % param
    config = {}
    for pair in param.strip().split(","):
        (k,v) = pair.split("=", 1)
        if k == 'config_file':
            if not YAML_AVAILABLE:
                logging.fatal("Install PyYAML in order to use config files.")
            else:
                with open(v) as f:
                    data = yaml.load(f)
                    for config_file_key, config_file_value in data.items():
                        config[config_file_key] = config_file_value
        else:
            config[k] = v
    return config


def get_parser():
    """Get the parser object which is used to build the configuration
    argument ``args``. This is a helper method for ``get_args()``
    TODO: Decentralize configuration
    
    Returns:
        ArgumentParser. The pre-filled parser object
    """
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    
    ## General options
    group = parser.add_argument_group('General options')
    group.add_argument('--config_file', 
                        help="Configuration file in standard .ini format. NOTE:"
                        " Configuration file overrides command line arguments")
    group.add_argument("--run_diagnostics", default=False, action="store_true",
                       help="Run diagnostics and check availability of "
                       "external libraries.")
    group.add_argument("--verbosity", default="info",
                        choices=['debug', 'info', 'warn', 'error'],
                        help="Log level: debug,info,warn,error")
    group.add_argument("--min_score", default=float("-inf"), type=float,
                        help="Delete all complete hypotheses with total scores"
                        " smaller than this value")
    group.add_argument("--range", default="",
                        help="Defines the range of sentences to be processed. "
                        "Syntax is equal to HiFSTs printstrings and lmerts "
                        "idxrange parameter: <start-idx>:<end-idx> (both "
                        "inclusive, start with 1). E.g. 2:5 means: skip the "
                        "first sentence, process next 4 sentences. If this "
                        "points to a file, we grap sentence IDs to translate "
                        "from that file and delete the decoded IDs. This can "
                        "be used for distributed decoding.")
    group.add_argument("--src_test", default="",
                        help="Path to source test set. This is expected to be "
                        "a plain text file with one source sentence in each "
                        "line. Words need to be indexed, i.e. use word IDs "
                        "instead of their string representations.")
    group.add_argument("--trgt_test", default="",
                        help="Path to source test set. This is expected to be "
                        "a plain text file with one source sentence in each "
                        "line. Words need to be indexed, i.e. use word IDs "
                        "instead of their string representations.")
    group.add_argument("--indexing_scheme", default="fairseq",
                        choices=['t2t', 'fairseq'],
                        help="This parameter defines the reserved IDs.\n\n"
                        "* 't2t': unk: 3, <s>: 2, </s>: 1.\n"
                        "* 'fairseq': unk: 3, <s>: 0, </s>: 2.")
    group.add_argument("--ignore_sanity_checks", default=False, type='bool',
                       help="SGNMT terminates when a sanity check fails by "
                       "default. Set this to true to ignore sanity checks.")
    group.add_argument("--input_method", default="file",
                        choices=['dummy', 'file', 'shell', 'stdin'],
                        help="This parameter controls how the input to SGNMT "
                        "is provided. SGNMT supports three modes:\n\n"
                        "* 'dummy': Use dummy source sentences.\n"
                        "* 'file': Read test sentences from a plain text file"
                            "specified by --src_test.\n"
                        "* 'shell': Start SGNMT in an interactive shell.\n"
                        "* 'stdin': Test sentences are read from stdin\n\n"
                        "In shell and stdin mode you can change SGNMT options "
                        "on the fly: Beginning a line with the string '!sgnmt '"
                        " signals SGNMT directives instead of sentences to "
                        "translate. E.g. '!sgnmt config predictor_weights "
                        "0.2,0.8' changes the current predictor weights. "
                        "'!sgnmt help' lists all available directives. Using "
                        "SGNMT directives is particularly useful in combination"
                        " with MERT to avoid start up times between "
                        "evaluations. Note that input sentences still have to "
                        "be written using word ids in all cases.")
    group.add_argument("--log_sum",  default="log",
                        choices=['tropical', 'log'],
                        help="Controls how to compute the sum in the log "
                        "space, i.e. how to compute log(exp(l1)+exp(l2)) for "
                        "log values l1,l2.\n\n"
                        "* 'tropical': approximate with max(l1,l2)\n"
                        "* 'log': Use logsumexp in scipy")
    group.add_argument("--n_cpu_threads", default=-1, type=int,
                        help="Set the number of CPU threads for libraries like"
                        " Theano or TensorFlow for internal multithreading. "
                        "Also, see the OMP_NUM_THREADS environment variable.")
    group.add_argument("--single_cpu_thread", default=False, type='bool',
                        help="Synonym for --n_cpu_threads=1")
    
    ## Decoding options
    group = parser.add_argument_group('Decoding options')
    group.add_argument("--decoder", default="beam",
                        choices=['greedy',
                                 'beam',
                                 'dfs',
                                 'simpledfs',
                                 'simplelendfs',
                                 'astar',
                                 'dijkstra',
                                 'dijkstra_ts',
                                 'reference',
                                 'sampling'],
                        help="Strategy for traversing the search space which "
                        "is spanned by the predictors.\n\n"
                        "* 'greedy': Greedy decoding (similar to beam=1)\n"
                        "* 'beam': beam search like in Bahdanau et al, 2015\n"
                        "* 'dfs': Depth-first search. This should be used for "
                        "exact decoding or the complete enumeration of the "
                        "search space, but it cannot be used if the search "
                        "space is too large (like for unrestricted NMT) as "
                        "it performs exhaustive search. If you have not only "
                        "negative predictor scores, set --early_stopping to "
                        "false.\n"
                        "* 'simpledfs': Depth-first search which works with "
                        "only one predictor. Good for exhaustive search in "
                        "combination with --score_lower_bounds_file from a "
                        "previous (beam) run.\n"
                        "* 'simplelendfs': simpledfs variant with length-"
                        "dependent lower bounds.\n"
                        "* 'astar': A* search. The heuristic function is "
                        "configured using the --heuristics options.")
    group.add_argument("--beam", default=0, type=int,
                        help="Size of beam. Only used if --decoder is set to "
                        "'beam' or 'astar'. For 'astar' it limits the capacity"
                        " of the queue. Use --beam 0 for unlimited capacity.")
    group.add_argument("--sub_beam", default=0, type=int,
                        help="This denotes the maximum number of children of "
                        "a partial hypothesis in beam-like decoders. If zero, "
                        "this is set to --beam to reproduce standard beam "
                        "search.")
    group.add_argument("--allow_unk_in_output", default=True, type='bool',
                        help="If false, remove all UNKs in the final "
                        "posteriors. Predictor distributions can still "
                        "produce UNKs, but they have to be replaced by "
                        "other words by other predictors")
    group.add_argument("--max_len_factor", default=2.0, type=float,
                        help="Limits the length of hypotheses to avoid "
                        "infinity loops in search strategies for unbounded "
                        "search spaces. The length of any translation is "
                        "limited to max_len_factor times the length of the "
                        "source sentence.")
    group.add_argument("--early_stopping", default=False, type='bool',
                        help="Use this parameter if you are only interested in "
                        "the first best decoding result. This option has a "
                        "different effect depending on the used --decoder. For"
                        " the beam decoder, it means stopping decoding when "
                        "the best active hypothesis ends with </s>. If false, "
                        "do not stop until all hypotheses end with EOS. For "
                        "the dfs and restarting decoders, early stopping "
                        "enables admissible pruning of branches when the "
                        "accumulated score already exceeded the currently best "
                        "score. DO NOT USE early stopping in combination with "
                        "the dfs or restarting decoder when your predictors "
                        "can produce positive scores!")
    group.add_argument("--heuristics", default="",
                        help="Comma-separated list of heuristics to use in "
                        "heuristic based search like A*.\n\n"
                        "* 'predictor': Predictor specific heuristics. Some "
                        "predictors come with own heuristics - e.g. the fst "
                        "predictor uses the shortest path to the final state."
                        " Using 'predictor' combines the specific heuristics "
                        "of all selected predictors.\n"
                        "* 'greedy': Do greedy decoding to get the heuristic"
                        " costs. This is expensive but accurate.\n"
                        "* 'lasttoken': Use the single score of the last "
                        "token.\n"
                        "* 'stats': Collect unigram statistics during decoding"
                        "and compare actual hypothesis scores with the product"
                        " of unigram scores of the used words.\n"
                        "* 'scoreperword': Using this heuristic normalizes the"
                        " previously accumulated costs by its length. It can "
                        "be used for beam search with normalized scores, using"
                        " a capacity (--beam), no other heuristic, and setting"
                        "--decoder to astar.\n\n"
                        "Note that all heuristics are inadmissible, i.e. A* "
                        "is not guaranteed to find the globally best path.")
    group.add_argument("--low_decoder_memory", default=True, type='bool',
                        help="Some decoding strategies support modes which do "
                        "not change the decoding logic, but make use of the "
                        "inadmissible pruning parameters like max_expansions "
                        "to reduce memory consumption. This usually requires "
                        "some  computational overhead for cleaning up data "
                        "structures. Applicable to restarting and bucket "
                        "decoders.")
    group.add_argument("--collect_statistics", default="best",
                       choices=['best', 'full', 'all'],
                        help="Determines over which hypotheses statistics are "
                        "collected.\n\n"
                        "* 'best': Collect statistics from the current best "
                        "full hypothesis\n"
                        "* 'full': Collect statistics from all full hypos\n"
                        "* 'all': Collect statistics also from partial hypos\n"
                        "Applicable to the bucket decoder, the heuristic "
                        "of the bow predictor, and the heuristic 'stats'.")
    group.add_argument("--heuristic_scores_file", default="",
                       help="The bow predictor heuristic and the stats "
                       "heuristic sum up the unigram scores of words as "
                       "heuristic estimate. This option should point to a "
                       "mapping file from word-id to (unigram) score. If this "
                       "is empty, the unigram scores are collected during "
                       "decoding for each sentence separately according "
                       "--collect_statistics.")
    group.add_argument("--decoder_diversity_factor", default=-1.0, type=float,
                       help="If this is greater than zero, promote diversity "
                       "between active hypotheses during decoding. The exact "
                       "way of doing this depends on --decoder:\n"
                       "* The 'beam' decoder roughly follows the approach in "
                       "Li and Jurafsky, 2016\n"
                       "* The 'bucket' decoder reorders the hypotheses in a "
                       "bucket by penalizing hypotheses with the number of "
                       "expanded hypotheses from the same parent.")
    group.add_argument("--simplelendfs_lower_bounds_file", default="",
                        help="Path to a file with length dependent lower "
                        "lower bounds for the simplelendfs decoder. Each line "
                        "must be in the format <len1>:<lower-bound1> ... "
                        "<lenN>:<lower-boundN>.")
    group.add_argument("--reward_type", default=None,
                        choices=['bounded','max', None],
                        help="")
    group.add_argument("--epsilon", default=20.0, type=float,
                        help="give positive value")
    group.add_argument("--memory_threshold_coef", default=0, type=int,
                        help="total queue size will be set to `memory_threshold_coef`"
                         "* beam size. When capacity is exceeded, the worst scoring "
                         "hypothesis from the earliest time step will be discarded")
    group.add_argument("--gumbel", action='store_true',
                        help="Add gumbel RV to make beam search effectively"
                        "random sampling")
    group.add_argument('--fairseq_temperature', default=1., type=float, metavar='N',
                       help='temperature for generation')
    

    ## Output options
    group = parser.add_argument_group('Output options')
    group.add_argument("--nbest", default=0, type=int,
                        help="Maximum number of hypotheses in the output "
                        "files. Set to 0 to output all hypotheses found by "
                        "the decoder. If you use the beam or astar decoder, "
                        "this option is limited by the beam size.")
    group.add_argument("--output_path", default="sgnmt-out.%s",
                        help="Path to the output files generated by SGNMT. You "
                        "can use the placeholder %%s for the format specifier")
    group.add_argument("--outputs", default="",
                        help="Comma separated list of output formats: \n\n"
                        "* 'text': First best translations in plain text "
                        "format\n"
                        "* 'nbest': Moses' n-best format with separate "
                        "scores for each predictor.\n"
                        "* 'nbest_sep': nbest translations in plain text "
                        "output to individual files based off of 'output_path'\n"
                        "* 'fst': Translation lattices in OpenFST "
                        "format with sparse tuple arcs.\n"
                        "* 'sfst': Translation lattices in OpenFST "
                        "format with standard arcs (i.e. combined scores).\n"
                        "* 'timecsv': Generate CSV files with separate "
                        "predictor scores for each time step.\n"
                        "* 'ngram': MBR-style n-gram posteriors.\n\n"
                        "For extract_scores_along_reference.py, select "
                        "one of the following output formats:\n"
                        "* 'json': Dump data in pretty JSON format.\n"
                        "* 'pickle': Dump data as binary pickle.\n"
                        "The path to the output files can be specified with "
                        "--output_path")
    group.add_argument("--remove_eos", default=True, type='bool',
                        help="Whether to remove </S> symbol on output.")
    group.add_argument("--src_wmap", default="",
                        help="Path to the source side word map (Format: <word>"
                        " <id>). See --preprocessing and --postprocessing for "
                        "more details.")
    group.add_argument("--trg_wmap", default="",
                        help="Path to the source side word map (Format: <word>"
                        " <id>). See --preprocessing and --postprocessing for "
                        "more details.")
    group.add_argument("--wmap", default="",
                        help="Sets --src_wmap and --trg_wmap at the same time")
    group.add_argument("--preprocessing", default="id",
                        choices=['id','word', 'char', 'bpe', 'bpe@@'],
                        help="Preprocessing strategy for source sentences.\n"
                        "* 'id': Input sentences are expected in indexed "
                        "representation (321 123 456 4444 ...).\n"
                        "* 'word': Apply --src_wmap on the input.\n"
                        "* 'char': Split into characters, then apply "
                        "(character-level) --src_wmap.\n"
                        "* 'bpe': Apply Sennrich's subword_nmt segmentation \n"
                        "SGNMT style (as in $SGNMT/scripts/subword-nmt)\n"
                        "* 'bpe@@': Apply Sennrich's subword_nmt segmentation "
                        "with original default values (removing </w>, using @@"
                        " separator)\n")
    group.add_argument("--postprocessing", default="id",
                        choices=['id','word', 'bpe@@','wmap', 'char', 'subword_nmt', 'bart', 'bpe_'],
                        help="Postprocessing strategy for output sentences. "
                        "See --preprocessing for more.")
    group.add_argument("--bpe_codes", default="",
                        help="Must be set if preprocessing=bpe. Path to the "
                        "BPE codes file from Sennrich's subword_nmt.")
    
    ## Predictor options
    
    # General
    group = parser.add_argument_group('General predictor options')
    group.add_argument("--predictors", default="fairseq",
                        help="Comma separated list of predictors. Predictors "
                        "are scoring modules which define a distribution over "
                        "target words given the history and some side "
                        "information like the source sentence. If vocabulary "
                        "sizes differ among predictors, we fill in gaps with "
                        "predictor UNK scores.:\n\n"
                        "* 't2t': Tensor2Tensor predictor.\n"
                        "         Options: t2t_usr_dir, t2t_model, "
                        "t2t_problem, t2t_hparams_set, t2t_checkpoint_dir, "
                        "pred_src_vocab_size, pred_trg_vocab_size\n"
                        "checkpoint_dir, lexnizza_min_id\n"
                        "* 'fairseq': fairseq predictor.\n"
                        "         Options: fairseq_path, fairseq_user_dir, "
                        "fairseq_lang_pair, n_cpu_threads")    
    
    
    # Neural predictors
    group = parser.add_argument_group('Neural predictor options')
    group.add_argument("--gnmt_alpha", default=0.0, type=float,
                       help="If this is greater than zero and the combination "
                       "scheme is set to length_norm, use Google-style length "
                       " normalization (Wu et al., 2016) rather than simply "
                       "dividing by translation length.")
    group.add_argument("--fairseq_path", default="",
                       help="Points to the model file (*.pt) for the fairseq "
                       "predictor. Like --path in fairseq-interactive.")
    group.add_argument("--fairseq_user_dir", default="",
                       help="fairseq user directory for additional models.")
    group.add_argument("--fairseq_lang_pair", default="",
                       help="Language pair such as 'en-fr' for fairseq. Used "
                       "to load fairseq dictionaries")

            
    return parser


def get_args():
    """Get the arguments for the current SGNMT run from both command
    line arguments and configuration files. This method contains all
    available SGNMT options, i.e. configuration is not encapsulated e.g.
    by predictors. 
    
    Returns:
        object. Arguments object like for ``ArgumentParser``
    """ 
    parser = get_parser()
    args = parse_args(parser)
    
    # Legacy parameter names
    if args.single_cpu_thread:
        args.n_cpu_threads = 1
    return args


def validate_args(args):
    """Some rudimentary sanity checks for configuration options.
    This method directly prints help messages to the user. In case of fatal
    errors, it terminates using ``logging.fatal()``
    
    Args:
        args (object):  Configuration as returned by ``get_args``
    """
    # Validate --range
    if args.range and args.input_method == 'shell':
        logging.warn("The --range parameter can lead to unexpected "
                     "behavior in 'shell' mode.")
        
    # Some common pitfalls
    sanity_check_failed = False
    if args.input_method == 'dummy' and args.max_len_factor < 10:
        logging.warn("You are using the dummy input method but a low value "
                     "for max_len_factor (%d). This means that decoding will "
                     "not consider hypotheses longer than %d tokens. Consider "
                     "increasing max_len_factor to the length longest relevant"
                     " hypothesis" % (args.max_len_factor, args.max_len_factor))
        sanity_check_failed = True
    if "fairseq" in args.predictors and args.indexing_scheme != "fairseq":
        logging.warn("You are using the fairseq predictor, but indexing_scheme "
                     "is not set to fairseq.")
        sanity_check_failed = True
    if args.preprocessing != "id" and not args.wmap and not args.src_wmap:
        logging.warn("Your preprocessing method needs a source wmap.")
        sanity_check_failed = True
    if args.postprocessing != "id" and not args.wmap and not args.trg_wmap:
        logging.warn("Your postprocessing method needs a target wmap.")
        sanity_check_failed = True
    if sanity_check_failed and not args.ignore_sanity_checks:
        raise AttributeError("Sanity check failed (see warnings). If you want "
            "to proceed despite these warnings, use --ignore_sanity_checks.")

