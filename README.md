# N-gram Music Completion

A statistical language-modeling project that treats musical notes as tokens and completes short symbolic sequences. It compares **unigram, bigram and trigram models**, smoothing choices and perplexity on a small course music corpus.

**Python · pandas · NumPy · n-gram language models**

## What it explores

- Tokenization of note sequences, including a mode that ignores accidentals.
- N-gram probability estimates with Lidstone smoothing.
- Smoothing-parameter selection using leave-one-out cross-validation.
- Sequence completion and perplexity comparisons.

The output is a sequence of note tokens, rather than a synthesized audio file.

## Run

The original `music.csv` is included beside the script, at the path expected by the entry point:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe ngram_music.py
```

The program prints the two experiment variants, completed sequences and comparison values. The [original Persian report](docs/report-fa.pdf) contains saved examples and result tables that can be read without running the code.

## Files and context

| Path | Purpose |
| --- | --- |
| [`ngram_music.py`](ngram_music.py) | Model, preprocessing, parameter search and experiment entry point |
| [`music.csv`](music.csv) | Small course note-sequence corpus |
| [`docs/report-fa.pdf`](docs/report-fa.pdf) | Original explanation and recorded results |

Developed for natural-language-processing coursework at the University of Isfahan. This project connects statistical NLP with a personal interest in music.
