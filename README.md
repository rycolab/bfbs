# SWOR Decoding

Library based on SGNMT: https://github.com/ucam-smt/sgnmt. See their [docs](http://ucam-smt.github.io/sgnmt/html/) for setting up a fairseq model to work with the library.

## Dependencies 

```
fairseq
sacrebleu
subword-nmt
scipy
numpy
cython
```

To compile the datastructure classes, run:
```
python setup.py install
```

Tokenization and detokenization should be performed with the [mosesdecoder library](https://github.com/moses-smt/mosesdecoder.git). tokenizer.perl and detokenizer.perl have been copied to the `scripts` folder for ease.

## Getting Started
We recommend starting with the pretrained models available from fairseq. Download any of the models from, e.g., their NMT examples, unzip, and place model checkpoints in `data/ckpts`. You'll have to preprocess the dictionary files to a format that the library expects (see [SGNMT fairseq tutorial](http://ucam-smt.github.io/sgnmt/html/tutorial_pytorch.html) for one-line command). Additionally, if the model uses BPE, you'll have to preprocess the input file to put words in byte pair format. A file named `bpecodes` should be included in the fairseq files if this is the case. Example:

```
cat input_file.txt | perl scripts/tokenizer.perl -threads 8 -l en > out
subword-nmt apply-bpe -c bpecodes -i out -o input_file_bpe.txt
```

Alternatively, one can play around with the toy model in the test scripts. Outputs are not meaningful but it is deterministic and useful for debugging.

### Beam Search

Basic beam search can be performed on a fairseq model translating from German to English on the IWSLT dataset as follows:

```
 python decode.py  --fairseq_path data/ckpts/model.pt --fairseq_lang_pair de-en --src_wmap data/wmaps/wmap.de --trg_wmap data/wmaps/wmap.en --input_file data/input_file_bpe.txt --preprocessing word --postprocessing bpe@@ --decoder beam --beam 10 
 ```

A faster version, best first beam search, simply changes the decoder:

```
 python decode.py  --fairseq_path data/ckpts/model.pt --fairseq_lang_pair de-en --src_wmap data/wmaps/wmap.de --trg_wmap data/wmaps/wmap.en --input_file data/valid.de --preprocessing word --postprocessing bpe@@ --decoder dijkstra_ts --beam 10 
 ```

By default, both decoders only return the best solution. Set `--early_stopping False` if you want the entire set.

A basic example of outputs can be seen when using the test suite:

 ```
 python test.py --decoder beam --beam 10 
 ```

 Additionally, you can run
 ```
 python decode.py --help
 ```
 to see descriptions of all available arguments.
 
### Sampling without Replacement
 To run SWOR decoding with the gumbel-max trick, use the command:

 ```
 python decode.py  --fairseq_path data/ckpts/model.pt --fairseq_lang_pair de-en --src_wmap data/wmaps/wmap.de --trg_wmap data/wmaps/wmap.en --input_file data/valid.de --preprocessing word --postprocessing bpe@@ --decoder dijkstra --beam 10 --gumbel --temperature 0.1
 ```
 where `--beam 10` would lead to 10 samples. For gumbel sampling, you should get the same results using the beam decoder.
 
 ```
 python decode.py  --fairseq_path data/ckpts/model.pt --fairseq_lang_pair de-en --src_wmap data/wmaps/wmap.de --trg_wmap data/wmaps/wmap.en --input_file data/valid.de --preprocessing word --postprocessing bpe@@ --decoder beam --beam 10 --gumbel --temperature 0.1
 ```
 For other sampling schemes, remove the `--gumbel` flag and set the decoder to `sampling, nucleus_sampling`.
 
The test suite can likewise be used by changing the decoder flag.


### Outputs

To see all outputs, set `--num_log <n>` for however many outputs (per input) you'd like to see. To write all outputs to files, set `--outputs nbest_sep --output_path <path_prefix>`. You'll then get a file of samples for each position (not each input!). To just write the first/best output to a file, use `--outputs text --output_path <path>`

### Scoring
 Scoring is not integrated into the library but can be performed afterwards. Make sure you use the arguments `--outputs text --output_path <file_name>.txt` during decoding and then detokenize the text using the mosesdecoder detokenizer script (copied to `scripts/detokenizer.perl` for ease). Given a (detokenized) baseline, you can then run sacrebleu to calculate BLEU. For example:

 ```
 cat <output_file_name>.txt | perl scripts/detokenizer.perl -threads 8 -l en | sacrebleu data/valid.detok.en
 ```

