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

## Getting Started
It's recommended to start with the pretrained models available from fairseq. Download any of the models from their NMT examples, unzip, and place model checkpoints in `data/ckpts`. 

Alternatively, one can play around with the toy model in the test scripts.
 
### SWOR
 To run SWOR decoding with the gumbel-max trick on a fairseq model, use the command:

 ```
 python decode.py  --fairseq_path data/ckpts/model.pt --fairseq_lang_pair de-en --src_wmap data/wmaps/wmap.bpe.de --trg_wmap data/wmaps/wmap.bpe.en --src_test data/valid.de --preprocessing word --postprocessing bpe@@ --decoder dijkstra --beam 10 --gumbel --temperature 0.1
 ```
 where `--beam 10` would lead to 10 samples. For gumbel sampling, you should get the same results using the beam decoder.
 
 ```
 python decode.py  --fairseq_path data/ckpts/model.pt --fairseq_lang_pair de-en --src_wmap data/wmaps/wmap.bpe.de --trg_wmap data/wmaps/wmap.bpe.en --src_test data/valid.de --preprocessing word --postprocessing bpe@@ --decoder beam --beam 10 --gumbel --temperature 0.1
 ```
 For other sampling schemes, remove the `--gumbel` flag and set the decoder to one of `sampling, basic_swor, mem_swor`.

 A basic example of outputs can be seen when using the test suite:

 ```
 python test.py --decoder beam --beam 10 --gumbel --temperature 0.1
 ```


### Beam Search

Basic beam search can be performed as follows:

```
 python decode.py  --fairseq_path data/ckpts/model.pt --fairseq_lang_pair de-en --src_wmap data/wmaps/wmap.bpe.de --trg_wmap data/wmaps/wmap.bpe.en --src_test data/valid.de --preprocessing word --postprocessing bpe@@ --decoder beam --beam 10 --early_stopping True
 ```

A faster version simply changes the decoder:

```
 python decode.py  --fairseq_path data/ckpts/model.pt --fairseq_lang_pair de-en --src_wmap data/wmaps/wmap.bpe.de --trg_wmap data/wmaps/wmap.bpe.en --src_test data/valid.de --preprocessing word --postprocessing bpe@@ --decoder dijkstra_ts --beam 10 --early_stopping True
 ```

Use the `--early_stopping True` flag if you only want the best solution.

The test suite can likewise be used by changing the decoder flag.

### Outputs

To see all outputs, set `--num_log <n>` for however many outputs (per input) you'd like to see. To write all outputs to files, set `--outputs nbest_sep --output_path <path_prefix>`. You'll then get a file of samples for each position (not each input!). To just write the first/best output to a file, use `--outputs text --output_path <path>`

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



