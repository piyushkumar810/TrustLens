from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Minimal download guard for common NLTK resources used in this project
for package in [
    "punkt_tab",
    "stopwords",
    "wordnet",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "words",
]:
    try:
        nltk.data.find(package if package != "punkt_tab" else "tokenizers/punkt_tab")
    except LookupError:
        nltk.download(package)

nltk_stopwords = set(stopwords.words("english"))
ps = PorterStemmer()
wnl = WordNetLemmatizer()
spacy_nlp = spacy.blank("en")


def clean_text(text: str) -> str:
    text = str(text or "")
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def pos_tag_example(text: str):
    tokens = word_tokenize(clean_text(text))
    return nltk.pos_tag(tokens)[:10]


def lemmatize_tokens(tokens):
    result = []
    for token in tokens:
        if not token:
            continue
        lemma = wnl.lemmatize(token, pos="v")
        if len(lemma) > 1:
            result.append(lemma)
    return result


def preprocess(text: str):
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    filtered = [t for t in tokens if t not in nltk_stopwords and len(t) > 2]
    stemmed = [ps.stem(t) for t in filtered]
    lemmas = lemmatize_tokens(filtered)
    pos = nltk.pos_tag(filtered)

    doc = spacy_nlp(cleaned)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return {
        "cleaned": cleaned,
        "tokens": filtered,
        "stemmed": stemmed,
        "lemmas": lemmas,
        "pos_tags": pos[:12],
        "entities": entities[:10],
    }


def build_ngram_model(reviews_df: pd.DataFrame):
    text_corpus = " ".join(reviews_df["text_"].dropna().astype(str).tolist())
    tokens = word_tokenize(clean_text(text_corpus))
    bigrams = Counter(zip(tokens, tokens[1:]))
    trigrams = Counter(zip(tokens, tokens[1:], tokens[2:]))
    return {"unigram_count": len(tokens), "bigram_count": len(bigrams), "trigram_count": len(trigrams)}


def run_step5() -> None:
    reviews = pd.read_csv(DATA_DIR / "reviews.csv")
    reviews["text_"] = reviews["text_"].fillna("")

    result_rows = []
    for text in reviews["text_"].head(200):
        result = preprocess(text)
        result_rows.append(result)

    reviews["preprocessed"] = reviews["text_"].apply(lambda t: preprocess(t)["lemmas"])
    processed_path = DATA_DIR / "reviews_preprocessed.csv"
    reviews.to_csv(processed_path, index=False)

    example_text = "Great quality product from a trusted brand. This shirt fits perfectly and feels premium."
    example_summary = preprocess(example_text)

    report = {
        "example_text": example_text,
        "example_result": example_summary,
        "sample_count": len(result_rows),
        "ngram_summary": build_ngram_model(reviews),
    }

    with open(REPORTS_DIR / "step5_nlp_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Saved processed reviews to:", processed_path)
    print("Example preprocessing output:", example_summary["lemmas"][:20])
    print("N-gram summary:", report["ngram_summary"])
    print("POS sample:", pos_tag_example(example_text))


if __name__ == "__main__":
    run_step5()
