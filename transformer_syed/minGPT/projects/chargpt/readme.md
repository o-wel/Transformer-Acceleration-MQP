# chargpt

chargpt trains a character-level language model.

We support three settings: 1 convenience setting and 2 "benchmark" settings that have acedemic literature results:

- a user specified `input.txt` file that we train an LM on (e.g. get tiny-shakespear (1.1MB of data) [here](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt))
- TODO [text8](http://mattmahoney.net/dc/textdata.html): also derived from Wikipedia text but all XML is removed and is lowercased to only 26 characters of
- TODO [enwik8](http://prize.hutter1.net) benchmark ("Hutter Prize"), first 100M bytes of a Wikipedia XML dump, with 205 unique tokensEnglish plus spaces

## Expanded literature dataset

From the repository root, prepare Tiny Shakespeare plus Lewis Carroll's
[Alice's Adventures in Wonderland](https://www.gutenberg.org/ebooks/11)
(public domain in the USA):

```sh
python projects/chargpt/prepare_data.py --preset=literature
python projects/chargpt/chargpt.py --data.input_path=data/literature/input.txt --trainer.max_iters=2000
```

The preparation script caches original downloads, removes Gutenberg's header and
footer from the training copy, normalizes line endings, and records source URLs,
SHA-256 hashes, sizes, and the combined character vocabulary in `sources.json`.
The originals, including Gutenberg's distribution information, remain in
`*.raw.txt`. Your existing root `input.txt` is untouched. Individual cleaned
`shakespeare.txt` and `alice.txt` files are also available in `data/literature/`.
You can select any UTF-8 text file using `--data.input_path=path/to/text.txt`.

This is next-character prediction on a mixture of drama and narrative prose,
not chatbot instruction training. The two works are concatenated without
oversampling; the larger Shakespeare corpus therefore contributes more examples.
The existing script reports training loss, not held-out validation performance.

charGPT constructs a new model with `GPT(config.model)`, so this command trains
all weights from scratch. For later LoRA adaptation, first load a trained charGPT
checkpoint and preserve its exact character-to-ID mapping, then enable LoRA
before building the trainer. Adding characters or rebuilding a sorted vocabulary
can change token IDs and make an existing checkpoint incompatible. Pretrained
GPT-2 uses a different tokenizer and cannot be substituted into this character
training pipeline directly.

## Everyday language (recommended)

```sh
python projects/chargpt/prepare_data.py
python projects/chargpt/chargpt.py --data.input_path=data/everyday/input.txt --system.work_dir=out/chargpt-everyday --trainer.max_iters=10000
```

The default `everyday` preset uses Google's Taskmaster-1 spoken-dialogue corpus:
human role-play conversations about food orders, rides, movies, reservations,
and auto repairs. We retain only utterance text and `User:` / `Assistant:` roles,
with blank lines between conversations. Conversations with missing speaker labels
are excluded. This is human-created task dialogue,
not unrestricted chat or synthetic model-generated text. The training objective
still predicts every next character, including both speakers' turns.

For narrative variety, it adds up to 200,000 characters each from *The Adventures
of Sherlock Holmes*, *Pride and Prejudice*, and *Alice's Adventures in Wonderland*.
The prose cap lets modern dialogue dominate without repeating or oversampling it.
Shakespeare is available in the separate `literature` preset. Both presets retain
raw downloads, cleaned source files, and a source/hash/vocabulary manifest.

Taskmaster-1 is by Bill Byrne, Karthik Krishnamoorthi, Chinnadhurai Sankar,
Arvind Neelakantan, Amit Dubey, Kyu-Young Kim, and Andy Cedilnik, Google LLC,
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
[Source and methodology](https://github.com/google-research-datasets/Taskmaster/tree/master/TM-1-2019).
The upstream attribution is saved as `TASKMASTER_ATTRIBUTION.md`; our changes
extract utterances, normalize whitespace, and add speaker labels. Gutenberg
source URLs are recorded in `sources.json`; original notices remain in raw files.

Use a new model/run directory for the expanded vocabulary. More varied training
text can improve everyday phrasing, but this small character model is not a
ChatGPT-like assistant. The example iteration count is a starting point, not a
verified convergence target. Output quality needs actual training and evaluation.
