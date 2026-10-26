# Fake News Detection

Classifying news articles as reliable or unreliable on [FakeNewsCorpus](https://github.com/several27/FakeNewsCorpus),
with an evaluation setup that measures how much of the score is real signal and how much is the model
recognising the publishing website.

This is a rebuild of my 2019 group project for the Data Science course at the University of Copenhagen
(with Niels-Christian, Simon and Ayman).
The original used notebooks, a PostgreSQL database, TF-IDF with linear models, and CNN/LSTM models on
GloVe embeddings. This version is a tested Python package with a fine-tuned transformer and a stricter
evaluation.

## The problem with the obvious setup

FakeNewsCorpus labels come from [OpenSources](https://github.com/OpenSourcesGuide/opensources), which
rates *websites*, not articles. Every article from a site gets that site's label. If you split articles
at random, the same sites appear in train and test, and a model can score well by learning each site's
boilerplate ("Getty Images", "Advertisements", share-button text) without reading the article.

So every model here is evaluated three ways:

1. **Random split**: articles shuffled into train/val/test (80/10/10). The usual, optimistic number.
2. **Domain split**: whole websites held out (`StratifiedGroupKFold` by domain). The test set only has
   sites the model has never seen. This is the honest in-corpus number.
3. **LIAR**: trained on FakeNewsCorpus, tested on the [LIAR](https://sites.cs.ucsb.edu/~william/data/liar_dataset.zip)
   test set (1,283 PolitiFact-rated political statements). Different source, different length, labels
   per claim instead of per site.

## Results

<!-- results:start -->
| Model | Split | FNC acc | FNC macro F1 | FNC AUC | LIAR acc | LIAR macro F1 | LIAR AUC |
|---|---|---|---|---|---|---|---|
| tfidf_logreg | random | 0.875 | 0.875 | 0.952 | 0.521 | 0.519 | 0.535 |
| roberta-base | random | 0.923 | 0.923 | 0.980 | 0.557 | 0.529 | 0.533 |
| tfidf_logreg | domain | 0.794 | 0.794 | 0.878 | 0.548 | 0.532 | 0.538 |
| roberta-base | domain | 0.836 | 0.835 | 0.913 | 0.578 | 0.523 | 0.554 |
| tfidf_logreg (trained on LIAR) | - | - | - | - | 0.633 | 0.617 | 0.662 |
<!-- results:end -->

FNC = FakeNewsCorpus test split. LIAR labels: pants-fire / false / barely-true = FAKE,
half-true / mostly-true / true = REAL. The last row is a TF-IDF model trained on LIAR's own training set,
as a reference for what is reachable on LIAR.

**What this shows**

- Holding out websites costs both models 8 to 9 points of accuracy (TF-IDF 0.875 to 0.794, RoBERTa
  0.923 to 0.836). That gap is site recognition, not content understanding.
- RoBERTa beats TF-IDF by about 4 points on both splits, trained on a quarter of the data. On the domain
  split its validation score peaks after epoch 1 and drops in epoch 2 while training loss keeps falling,
  so it starts to memorise the training sites quickly; the epoch-1 checkpoint is the one evaluated.
- The TF-IDF model's strongest features confirm it: "advertisements", "breakpoint", "phaser" (leftover
  page markup) and "infowars" push towards FAKE; "getty images", "reuters", "ap" push towards REAL.
- Nothing trained on FakeNewsCorpus transfers to LIAR: macro F1 around 0.52 and AUC around 0.54 for
  every model, while a simple model trained on LIAR itself gets 0.62 macro F1. Site-level labels on long
  articles teach style, not claim truthfulness.

## Data

| | |
|---|---|
| Source | FakeNewsCorpus v1.0 (`news_cleaned_2018_02_13.csv`, 29 GB) |
| Sample | 200,000 articles, 100,000 per class, from 377 domains |
| FAKE | fake, conspiracy, junksci, hate, unreliable |
| REAL | reliable, political |
| Dropped | satire, bias, clickbait, rumor, state, unknown (ambiguous) |

The corpus is streamed in 200k-row chunks. Each domain keeps at most 1,000 articles, picked by smallest
content hash (a bottom-k sample), so a few huge sites can't dominate. Exact duplicates and articles under
200 characters are removed, and text is cut to 5,000 characters.

The 2019 version counted bias, clickbait, rumor and unknown as REAL. Those are dropped here because they
are neither reliable nor fake.

## Models

- **TF-IDF + logistic regression**: word uni- and bigrams, sublinear TF, 200k features.
  Text cleaning (lowercase, URL/e-mail masking, tokenizer) is ported from the original project.
- **RoBERTa-base**: fine-tuned for 2 epochs on 50,000 training articles (first 256 tokens), AdamW with
  linear warmup and decay, bf16 autocast, best epoch picked on validation macro F1. One RTX 3090.

## Usage

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras

# put news_cleaned_2018_02_13.csv in data/raw/fakenewscorpus/ and the LIAR tsv files in data/raw/liar/
uv run fakenews prepare                       # stream and sample the corpus (~3 min)
uv run fakenews baseline --split domain       # TF-IDF + LogReg
uv run fakenews transformer --split domain --max-train 50000
uv run fakenews liar-reference
uv run fakenews report                        # results/RESULTS.md

uv run python -m fakenews.app                 # Gradio demo
```

The demo shows both models' probability and which terms pushed the TF-IDF model's decision.

## Layout

```
src/fakenews/
  config.py        paths and label mappings
  data.py          streaming sampler, LIAR loader, random/domain splits
  preprocess.py    text cleaning
  baseline.py      TF-IDF + logistic regression
  transformer.py   PyTorch fine-tuning loop
  evaluate.py      accuracy, macro F1, ROC AUC
  cli.py           command line
  app.py           Gradio demo
tests/             pytest on synthetic data (no downloads needed)
results/           JSON metrics per run and RESULTS.md
```

## Limitations

- Labels are per website, so even the domain split measures "does this read like an unreliable outlet",
  not "is this claim false". The demo says so.
- The corpus was scraped in 2017 to 2018 and is heavy on US politics.
- RoBERTa sees only the first 256 tokens and was trained on a 50k subsample.

## License

MIT
