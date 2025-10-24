import csv
import math
import random
from typing import List, Optional
from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm

# --- NLTK Setup ---
# Ensure NLTK data is available
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# --- Metrics Calculation ---
def calculate_metrics(prompt: str, response: str):
    """Calculates a dictionary of metrics for a given prompt and response."""
    # Ensure inputs are strings
    prompt = str(prompt)
    response = str(response)

    # Tokenize the response
    words = word_tokenize(response.lower())
    if not words:
        return {
            "lexical_diversity": 0.0,
            "query_coverage": 0.0,
            "flesch_kincaid_grade": 0.0,
            "repetition_penalty": 0.0,
        }

    # 1. Lexical Diversity
    lexical_diversity = (len(set(words)) / len(words)) * 100 if words else 0

    # 2. Query Coverage
    prompt_words = set(word_tokenize(prompt.lower()))
    stop_words = set(stopwords.words('english'))
    prompt_keywords = prompt_words - stop_words
    response_words = set(words)
    
    covered_keywords = prompt_keywords.intersection(response_words)
    query_coverage = (len(covered_keywords) / len(prompt_keywords)) * 100 if prompt_keywords else 100

    # 3. Structural Depth (Sentence Count)
    sentence_count = len(sent_tokenize(response))

    # 4. Complexity Proxy (Average Word Length)
    total_chars = sum(len(word) for word in words)
    avg_word_length = total_chars / len(words) if words else 0

    # Reuse logic from app/evaluation/metrics.py: FK grade and repetition penalty
    def _count_syllables(word: str) -> int:
        word = word.lower()
        vowels = "aeiouy"
        count = 0
        prev = False
        for ch in word:
            is_vowel = ch in vowels
            if is_vowel and not prev:
                count += 1
            prev = is_vowel
        if word.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)

    def _flesch_kincaid_grade(num_sentences: int, num_words: int, num_syllables: int) -> float:
        if num_sentences <= 0 or num_words <= 0:
            return 0.0
        return 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59

    def _repetition_penalty(tokens: list[str], n: int = 3) -> float:
        if len(tokens) < n:
            return 0.0
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        total = len(ngrams)
        unique = len(set(ngrams))
        repeats = total - unique
        return (repeats/total)*100.0 if total > 0 else 0.0

    num_sentences = max(len(sent_tokenize(response)), 1)
    num_words = len(words)
    num_syllables = sum(_count_syllables(w) for w in words)
    fk_grade = _flesch_kincaid_grade(num_sentences, num_words, num_syllables)
    rep_penalty = _repetition_penalty(words, n=3)

    return {
        "lexical_diversity": round(lexical_diversity, 2),
        "query_coverage": round(query_coverage, 2),
        "flesch_kincaid_grade": round(fk_grade, 2),
        "repetition_penalty": round(rep_penalty, 2),
    }

# --- Helpers for variation ---
def _clip(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def perturb_metrics(base: dict, temperature: float, top_p: float) -> dict:
    """Create plausible variation in metrics based on temperature and top_p.

    Intuition (synthetic but useful for visualization):
    - Higher temperature -> more lexical diversity, slightly fewer on-topic words
    - Higher top_p (broader sampling) -> more diversity but can reduce coverage a bit
    - Sentence count drifts with temperature
    - Avg word length drifts slightly with top_p
    """
    t = temperature
    p = top_p

    # Normalize around 0.5 for simple linear effects
    dt = t - 0.5
    dp = p - 0.5

    # Noise terms for realism
    n_ld = random.uniform(-2.0, 2.0)
    n_qc = random.uniform(-2.0, 2.0)
    n_fk = random.uniform(-0.6, 0.6)
    n_rp = random.uniform(-3.0, 3.0)

    # Start from base
    ld = base["lexical_diversity"]
    qc = base["query_coverage"]
    # For new metrics we use the base values (fk grade, repetition) as a starting point
    # and then perturb.
    fk = base["flesch_kincaid_grade"]
    rp = base["repetition_penalty"]

    # Apply synthetic effects
    ld2 = ld * (1.0 + 0.25 * dt + 0.15 * dp) + n_ld
    qc2 = qc * (1.0 - 0.20 * dt - 0.10 * dp) + n_qc
    # FK grade tends to increase with higher temperature and slightly with top_p
    fk_baseline = 2.0 + 8.0 * (0.5 + 0.25 * dt + 0.15 * dp)  # ~2..10 baseline
    fk2 = max(fk, fk_baseline) * (1.0 + 0.10 * dt + 0.05 * dp) + n_fk
    # Repetition penalty tends to decrease with temperature (more variety), increase with lower top_p
    # Provide a baseline even if base rp=0 so the grid shows variation
    rp_baseline = 5.0 + 15.0 * (0.5 - dp) - 8.0 * dt  # ~0..20 depending on params
    rp2 = max(rp, rp_baseline) + n_rp

    # Clip to reasonable ranges
    ld2 = round(_clip(ld2, 0.0, 100.0), 2)
    qc2 = round(_clip(qc2, 0.0, 100.0), 2)
    fk2 = round(_clip(fk2, 0.0, 18.0), 2)
    rp2 = round(_clip(rp2, 0.0, 100.0), 2)

    return {
        "lexical_diversity": ld2,
        "query_coverage": qc2,
    "flesch_kincaid_grade": fk2,
    "repetition_penalty": rp2,
    }


# --- Main Script ---
def generate_mock_data(
    num_prompts: int = 60,
    grid_temps: Optional[List[float]] = None,
    grid_top_ps: Optional[List[float]] = None,
    output_file: str = "mock_data.csv",
    model_name: str = "gemini-2.0-flash-lite",
):
    """Generate a grid of parameter variations per prompt to enable heatmaps.

    For each prompt, we iterate over a small grid of (temperature, top_p) and
    perturb the base metrics to correlate with the parameters.
    """

    if grid_temps is None:
        grid_temps = [0.1, 0.3, 0.5, 0.7, 0.9]
    if grid_top_ps is None:
        grid_top_ps = [0.2, 0.4, 0.6, 0.8, 1.0]

    print("Downloading 'databricks/databricks-dolly-15k' dataset...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split=f"train[:{num_prompts}]")

    headers = [
        "prompt",
        "model",
        "temperature",
        "top_p",
        "lexical_diversity",
        "query_coverage",
        "flesch_kincaid_grade",
        "repetition_penalty",
    ]

    total_rows = len(grid_temps) * len(grid_top_ps) * num_prompts
    print(f"Generating {total_rows} rows into '{output_file}'...")

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for item in tqdm(dataset, total=num_prompts):
            prompt = item["instruction"]
            response = item["response"]
            if not prompt or not response:
                continue

            base = calculate_metrics(prompt, response)

            for t in grid_temps:
                for p in grid_top_ps:
                    m = perturb_metrics(base, t, p)
                    writer.writerow(
                        {
                            "prompt": prompt,
                            "model": model_name,
                            "temperature": round(t, 2),
                            "top_p": round(p, 2),
                            **m,
                        }
                    )

    print(f"\nSuccessfully generated '{output_file}' with grid-variant mock data.")


if __name__ == "__main__":
    random.seed(42)
    generate_mock_data()
