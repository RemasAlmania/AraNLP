
# Arabic Word Embeddings + Text Classification (AraNLP Bootcamp - SDAIA)

## Overview
This project trains Arabic Word2Vec embeddings using a Twitter 
dataset, then uses these embeddings to build a sentiment classification model 
in PyTorch.

## Dataset
- **Name:** Arabic Sentiment Twitter Corpus
- **Source:** HuggingFace (`arbml/Arabic_Sentiment_Twitter_Corpus`)
- **Size:** 47,000 Arabic tweets
- **Labels:** 0 = Negative, 1 = Positive

## Pipeline
1. **Preprocessing** — clean Arabic tweets (remove URLs, mentions, 
   normalize alef, remove diacritics, tokenize)
2. **Word2Vec Training** — train Skip-gram and CBOW embeddings 
   from scratch using gensim
3. **Sentence Representation** — average word vectors per sentence
4. **Classification** — feedforward neural network in PyTorch

## Experiments

| Configuration | Accuracy |
|---|---|
| Skip-gram (vector_size=100, window=5) | 74.94% |
| CBOW (vector_size=100, window=5) | 72.09% |

Skip-gram outperformed CBOW, consistent with literature showing Skip-gram performs better on medium-sized datasets.

## Requirements
pip install gensim datasets pyarabic torch scikit-learn

## How to Run
1. Open `arabic_w2v_classification.ipynb` in Google Colab
2. Run all cells in order
3. Results will be printed at the end of each experiment

