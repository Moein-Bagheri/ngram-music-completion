# Melodies as language

**Persian melodies read as language.** Notes become tokens, and n-gram models try to finish the phrase: a small experiment in where counting stops being enough.

*Python · NumPy · n-gram language models* · [Report in Persian](docs/report-fa.pdf)

## What I did

- Worked with a course corpus of 41 Persian songs as note sequences, each tagged with its dastgah (shur, mahur and others).
- Implemented unigram, bigram and trigram models with Lidstone smoothing, with and without accidentals.
- Tuned the smoothing by leave-one-out cross-validation over all 41 songs, and compared Laplace and Jeffreys–Perks smoothing.
- Completed five melody openings with every model and scored them by perplexity.

## What I found

| Model | Best α (leave-one-out) | Mean perplexity |
| --- | ---: | ---: |
| Unigram | 5.0 | 11.03 |
| Bigram | 1.0 | 8.45 |
| Trigram | 0.7 | 9.29 |

- **Context helps, up to a point.** Bigrams beat unigrams on unseen songs, but trigrams fall back: 41 songs are too few to see most three-note contexts.
- **The likeliest note isn't music.** Greedy completion repeats the commonest note (C C C C…) or falls into loops (C D C D…).
- **Perplexity isn't the whole story.** The report suggests judging a continuation by whether it stays in the song's dastgah.

## Try it

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe ngram_music.py
```

It prints both experiment variants, the completed sequences and their perplexities. The output is note tokens, not audio.

## Layout

| Path | What's there |
| --- | --- |
| [`ngram_music.py`](ngram_music.py) | Models, smoothing search and experiments |
| [`music.csv`](music.csv) | The 41-song corpus |
| [`docs/report-fa.pdf`](docs/report-fa.pdf) | Method, results and analysis, in Persian |

Natural language processing course project, University of Isfahan, 2026.
