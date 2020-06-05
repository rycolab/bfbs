# PMI Decoding
 You'll need to install fairseq (latest version should be fine) in order to work with the models already trained. For scoring, install sacrebleu. Both can be done using the default packages in pip.
 Unzip the model checkpoints and place them in `data/ckpts` 

 To run Dijkstra's with a normal conditional LM, use the command:

 ```
 python decode.py  --fairseq_path data/ckpts/cond_model.pt --fairseq_lang_pair de-en --src_wmap data/wmaps/wmap.bpe.de --trg_wmap data/wmaps/wmap.bpe.en --input_method file --src_test data/valid.de --preprocessing word --n_cpu_threads 30 --postprocessing bpe@@ --decoder dijkstra_ts 
 ```
 note that this probably won't finish since it takes up a huge amount of memory. To run beam search with k=5, use the command: 
 
 ```
 python decode.py  --fairseq_path data/ckpts/cond_model.pt --fairseq_lang_pair de-en --src_wmap data/wmaps/wmap.bpe.de --trg_wmap data/wmaps/wmap.bpe.en --input_method file --src_test data/valid.de --preprocessing word --n_cpu_threads 30 --postprocessing bpe@@ --decoder beam --beam 5 
 ```

To run dijkstra_ts with PMI and a unigram model as the marginal LM, use the command:
 ```
 python decode.py  --fairseq_path data/ckpts/cond_model.pt --fairseq_lang_pair de-en --src_wmap data/wmaps/wmap.bpe.de --trg_wmap data/wmaps/wmap.bpe.en --input_method file --src_test data/valid.de --preprocessing word --n_cpu_threads 30 --postprocessing bpe@@ --decoder dijkstra_ts --subtract-uni --lmbd 0.2
 ```

 Note that lmbda is the interpolation parameter (i.e. lmbda 0.2 -> log P(y|x) - 0.2log P(y)). To run dijkstra_ts with PMI and a NN as the marginal LM, use the command:

 ```
 python decode.py  --fairseq_path data/ckpts/cond_model.pt --fairseq_lang_pair de-en --src_wmap data/wmaps/wmap.bpe.de --trg_wmap data/wmaps/wmap.bpe.en --input_method file --src_test data/valid.de --preprocessing word --n_cpu_threads 30 --postprocessing bpe@@ --decoder dijkstra_ts --subtract-marg --marg_path data/ckpts/lm.pt --lmbd 0.2
 ```

You can run any of the decoders in the library and they should work with PMI (no promises that they'll finish running...). For example, you can run DFS by setting `--decoder dfs` or use regular beam search with `--decoder beam --beam <k>`.

### Scoring
 For scoring, append the arguments `--outputs text --output_path <file_name>.txt` and then detokenize the text using the moses detokenizer script (copied to `scripts/detokenizer.perl` for ease)

 ```
 cat <file_name>.txt | perl scripts/detokenizer.perl -threads 8 -l en > out
 ```

 The detokenized valid/test files for IWSLT14 de-en are located in the `data` folder already. You can run sacrebleu to score with:

 ```
 cat out | sacrebleu data/valid.detok.en
 ```

 If you want to decode on different data sets, lmk and I can send the tokenization scripts and bpe codes I used for training the models.

## SGNMT


SGNMT is an open-source framework for neural machine translation (NMT) and other sequence prediction
tasks. The tool provides a flexible platform which allows pairing NMT with various other models such 
as language models, length models, or bag2seq models. It supports rescoring both n-best lists and lattices.
A wide variety of search strategies is available for complex decoding problems. 

SGNMT is compatible with the following NMT toolkits:

-  [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) ([TensorFlow](https://www.tensorflow.org/))
-  [fairseq](https://github.com/pytorch/fairseq) ([PyTorch](https://pytorch.org/))

Old SGNMT versions (0.x) are compatible with:

- [(extended) TF seq2seq tutorial](https://github.com/ehasler/tensorflow) ([TensorFlow](https://www.tensorflow.org/))
- [Blocks](http://blocks.readthedocs.io/en/latest/) ([Theano](http://deeplearning.net/software/theano/))


Features:

- Syntactically guided neural machine translation (NMT lattice rescoring)
- n-best list rescoring with NMT
- Integrating external n-gram posterior probabilities used in MBR
- Ensemble NMT decoding
- Forced NMT decoding
- Integrating language models
- Different search algorithms (beam, A\*, depth first search, greedy...)
- Target sentence length modelling
- Bag2Sequence models and decoding algorithms
- Joint decoding with word- and subword/character-level models
- Hypothesis recombination
- Heuristic search
- ...

### Documentation

Please see the [full SGNMT documentation](http://ucam-smt.github.io/sgnmt/html/) for more information.

### Contributors

- Felix Stahlberg, University of Cambridge
- Eva Hasler, SDL Research
- Danielle Saunders, University of Cambridge

### Citing

If you use SGNMT in your work, please cite the following paper:

Felix Stahlberg, Eva Hasler, Danielle Saunders, and Bill Byrne.
SGNMT - A Flexible NMT Decoding Platform for Quick Prototyping of New Models and Search Strategies.
In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP 17 Demo Session)*, September 2017. Copenhagen, Denmark.
[arXiv](https://arxiv.org/abs/1707.06885)


