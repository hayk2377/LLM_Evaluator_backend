from collections import Counter
from typing import Iterable
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize


def _count_syllables(word: str) -> int:
    """Heuristic syllable counter suitable for FK grade (no heavy deps)."""
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    # Silent 'e' adjustment
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _flesch_kincaid_grade(num_sentences: int, num_words: int, num_syllables: int) -> float:
    if num_sentences <= 0 or num_words <= 0:
        return 0.0
    return 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59


def _repetition_penalty(tokens: list[str]) -> float:
    """Percentage of repeated n-grams in the text (0..100).

    Uses trigrams when available; if no trigram repeats and text is short,
    falls back to bigrams to avoid always-zero on brief outputs.
    """
    def _ratio_for_n(n: int) -> float:
        if len(tokens) < n:
            return 0.0
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        total = len(ngrams)
        if total == 0:
            return 0.0
        unique = len(set(ngrams))
        repeats = total - unique
        return (repeats / total) * 100.0

    r3 = _ratio_for_n(3)
    if r3 > 0.0 or len(tokens) >= 50:
        return r3
    # For short texts where trigrams rarely repeat, check bigrams too
    return _ratio_for_n(2)


def calculate_metrics(prompt: str, response: str) -> dict:
    """Compute objective metrics for a prompt-response pair.

    Returns a dict with: lexical_diversity, query_coverage, flesch_kincaid_grade, repetition_penalty.
    """
    text = (response or "")
    tokens = word_tokenize(text.lower())
    if not tokens:
        return {
            "lexical_diversity": 0.0,
            "query_coverage": 0.0,
            "flesch_kincaid_grade": 0.0,
            "repetition_penalty": 0.0,
        }

    # Unique types vs tokens as a percentage
    lexical_diversity = (len(set(tokens)) / len(tokens)) * 100.0

    # Coverage over prompt keywords (non-stopwords)
    prompt_words = set(word_tokenize((prompt or "").lower()))
    stop_words = set(stopwords.words("english"))
    prompt_keywords = prompt_words - stop_words
    response_words = set(tokens)
    covered = prompt_keywords.intersection(response_words)
    query_coverage = (len(covered) / len(prompt_keywords) * 100.0) if prompt_keywords else 100.0

    # Flesch-Kincaid Grade Level
    sentences = sent_tokenize(text)
    num_sentences = max(len(sentences), 1)
    num_words = len(tokens)
    num_syllables = sum(_count_syllables(w) for w in tokens)
    fk_grade = _flesch_kincaid_grade(num_sentences, num_words, num_syllables)

    # Repetition penalty with trigram-first, bigram fallback for short texts
    repetition = _repetition_penalty(tokens)

    return {
        "lexical_diversity": lexical_diversity,
        "query_coverage": query_coverage,
        "flesch_kincaid_grade": fk_grade,
        "repetition_penalty": repetition,
    }
