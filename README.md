# 11611-NLP — Spring 2026

Homework assignments for **CMU 11-611: Natural Language Processing**, Spring 2026.

---

## Repository Structure

```
11611-NLP-26Spring/
├── Assignment1-LanguageID/      # Language Identification
├── Assignment2/                 # N-gram LMs & Text Classification
├── Assignment3/                 # Machine Translation Evaluation
└── Assignment4/                 # Direct Preference Optimization (DPO)
```

---

## Assignments

### Assignment 1 — Language Identification
Identifies the language of text samples using character-level or word-level features.

- **Data:** `train.tsv`, `test.tsv` — tab-separated files with text and language labels.

---

### Assignment 2 — N-gram Language Models & Text Classification
Builds n-gram language models and a neural encoder-based text classifier.

- **Key files:**
  - `ngram_lm.py` — N-gram language model implementation
  - `encoder_classifier.py` — Encoder-based text classifier
  - `utils.py` — Shared utilities
  - `HW2.ipynb` — Main notebook
  - `glove.6B.50d.txt` — Pre-trained GloVe embeddings (50-dimensional)
- **Data:**
  - `data/bbc/` — BBC news articles across 5 categories (business, entertainment, politics, sport, tech)
  - `data/lyrics/` — Song lyrics from artists including Taylor Swift, Billie Eilish, Ed Sheeran, and Green Day
  - `data/train/` and `data/test/` — Train/test splits for lyrics classification

---

### Assignment 3 — Machine Translation Evaluation
Evaluates machine translation output using automatic metrics.

- **Key files:**
  - `hw3_starter_S26.ipynb` — Starter notebook
  - `test-DIST.json` — Test distribution data
  - `test-translations.txt` — Translation outputs to evaluate
  - `hw3_writeup.pdf` — Written report

---

### Assignment 4 — Direct Preference Optimization (DPO)
Implements DPO, an RLHF-alternative algorithm for aligning language models from human preference data.

- **Key files:**
  - `submission.py` — Core implementation (log-probabilities, DPO loss)
  - `train.py` — Training script
  - `data.py` — Data pipeline
  - `requirements.txt` — Python dependencies
- **Data:** `data/uf_small_{train,val,test}.jsonl` — UltraFeedback preference dataset (small split)
- **Tests:** `tests_public/` — Public test suite covering data pipeline, log-probability computation, and DPO loss

---

## Setup

```bash
# Clone the repo
git clone https://github.com/althea-yuquan-chen/11611-NLP.git
cd 11611-NLP

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies (for Assignment 4)
pip install -r Assignment4/requirements.txt
```

---

## Running Tests (Assignment 4)

```bash
cd Assignment4
pytest tests_public/
```

---

## Course

[11-611 Natural Language Processing](https://reecursion.github.io/11411-11611-nlp-website/) — Language Technologies Institute, Carnegie Mellon University.

---

## Author

Yuquan (Althea) Chen
