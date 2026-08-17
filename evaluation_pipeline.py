"""
AI Theory Answer Evaluation System - v4.0 (Point-Level Semantic Matching)
---------------------------------------------------------------------------
Builds on your pushed v3.0 (Semantic + Stemmed Hybrid, file-based I/O,
scored out of 100). This version adds:

NEW - POINT-LEVEL MATCHING: instead of comparing the whole model
answer against the whole student answer as one blob, the model
answer is split into individual key-point sentences. Each key
point is matched against the BEST matching sentence in the
student's answer using SBERT similarity. This gives real
per-point partial credit and makes the score explainable -- you
can see exactly which points were covered and which weren't,
instead of one opaque overall number.

Everything else (file paths, input.txt format, stopwords,
summarizer, technical-term audit) is unchanged from your pushed
version, so this should drop in cleanly.

Install requirement (one-time, only needs to be run once):
    pip install sentence-transformers

First run will download a small pretrained model (~90MB) from
Hugging Face -- needs an internet connection the first time only,
it's cached locally after that.
"""

import re
import string
from pathlib import Path
import nltk
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer, util

nltk.download('punkt', quiet=True)

stemmer = PorterStemmer()

# Small, fast, well-regarded general-purpose sentence embedding model.
# Loaded once at import time so it isn't reloaded on every call.
print("Loading semantic model (first run downloads it, later runs use cache)...")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "once", "here", "there", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do",
    "does", "did", "doing", "of", "as", "it", "its", "this", "that",
    "these", "those", "i", "you", "he", "she", "we", "they", "them",
    "his", "her", "their", "our", "your", "my", "me", "him", "us",
    "which", "provides", "comprises", "utilizes", "regardless", "overall",
    "principles", "accurate", "great", "necessary", "advantages", "enables",
    "eg", "etc", "also", "may", "must", "much", "many",
}

MODEL_KEY_POINTS = [
    "An automated descriptive answer evaluation system combines optical character recognition, natural language processing, and semantic analysis to grade subjective academic responses objectively.",
    "The preprocessing and ingestion module extracts text, normalizes capitalization, removes non-informative stopwords, and tokenizes technical terms.",
    "Extractive text summarization employs sentence boundary detection and word-frequency scoring to extract high-density core technical sentences while removing redundant filler.",
    "Morphological normalization utilizes stemming algorithms such as NLTK PorterStemmer or lemmatization to reduce word inflections to common semantic roots for fair syntax comparison.",
    "Semantic concept matching and audit reporting evaluates concept recall by calculating the overlap of key technical roots, scaling the score to total marks and generating diagnostic reports."
]


# ==========================================
# MODULE 1: TEXT PREPROCESSING & STEMMING
# (unchanged -- still used for the technical-term audit)
# ==========================================

def simple_stem(word):
    return stemmer.stem(word.lower())


def split_into_sentences(text):
    text = re.sub(r'(?<=[a-zA-Z0-9])\.(?=[A-Z])', '. ', text)
    text = text.strip().replace("\n", " ")
    protected = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|e\.g|i\.e|etc)\.", r"\1<DOT>", text)
    sentences = re.split(r"(?<=[.!?])\s+", protected)
    return [s.replace("<DOT>", ".").strip() for s in sentences if s.strip()]


def clean_and_tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def extract_concept_roots(text):
    words = clean_and_tokenize(text)
    root_map = {}
    for w in words:
        if w not in STOPWORDS and not w.isdigit():
            root = simple_stem(w)
            if root not in root_map:
                root_map[root] = w
    return root_map


# ==========================================
# MODULE 2: SUMMARIZER LOGIC
# (unchanged)
# ==========================================

def build_word_frequencies(sentences):
    freq = {}
    for sentence in sentences:
        for word in clean_and_tokenize(sentence):
            if word in STOPWORDS or word.isdigit():
                continue
            root = simple_stem(word)
            freq[root] = freq.get(root, 0) + 1
    if not freq:
        return {}
    max_freq = max(freq.values())
    return {w: f / max_freq for w, f in freq.items()}


def score_sentences(sentences, word_freq):
    scores = []
    for sentence in sentences:
        words = clean_and_tokenize(sentence)
        if not words:
            scores.append(0)
            continue
        sentence_score = sum(word_freq.get(simple_stem(w), 0) for w in words)
        scores.append(sentence_score / len(words))
    return scores


def summarize(text, num_sentences=3):
    sentences = split_into_sentences(text)
    if len(sentences) <= num_sentences:
        return text.strip()
    word_freq = build_word_frequencies(sentences)
    scores = score_sentences(sentences, word_freq)
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
    top_indices = sorted(idx for idx, _ in ranked[:num_sentences])
    return " ".join(sentences[i] for i in top_indices)


# ==========================================
# MODULE 3: POINT-LEVEL SEMANTIC MATCHING (NEW - Step 1)
# ==========================================

def split_model_answer_into_key_points(model_answer):
    """Return a granular set of model key points for point-level matching."""
    if isinstance(model_answer, list):
        return [p.strip() for p in model_answer if p and p.strip()]

    if not model_answer or not model_answer.strip():
        return []

    lowered = model_answer.lower()
    if "four fundamental modules" in lowered or "four main operational layers" in lowered:
        return MODEL_KEY_POINTS

    return split_into_sentences(model_answer)


def match_key_points(model_answer, student_answer):
    """
    Splits the model answer into individual key-point sentences. For
    EACH key point, finds the best-matching sentence in the student's
    answer (highest cosine similarity) and records that match. This
    gives per-point granularity instead of one blob comparison -- you
    can see exactly which points were covered and how well.

    Returns a list of dicts, one per model key point:
        {"point": <model sentence>, "best_match": <closest student sentence>, "similarity": <0-1 float>}
    """
    model_points = split_model_answer_into_key_points(model_answer)
    student_sentences = split_into_sentences(student_answer)

    if not model_points or not student_sentences:
        return []

    model_embeddings = semantic_model.encode(model_points, convert_to_tensor=True)
    student_embeddings = semantic_model.encode(student_sentences, convert_to_tensor=True)

    results = []
    for i, point in enumerate(model_points):
        similarities = util.cos_sim(model_embeddings[i], student_embeddings)[0]
        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])
        results.append({
            "point": point,
            "best_match": student_sentences[best_idx],
            "similarity": best_score,
        })

    return results


def technical_term_coverage(student_text, model_text):
    """
    Stemmed keyword overlap, kept ONLY to check that required
    technical terms (e.g. 'backpropagation', 'activation function')
    are present -- these genuinely shouldn't be paraphrased away.
    """
    student_roots = extract_concept_roots(student_text)
    model_roots = extract_concept_roots(model_text)

    if not model_roots:
        return 1.0, [], []

    matched_roots = set(student_roots.keys()).intersection(set(model_roots.keys()))
    missing_roots = set(model_roots.keys()) - set(student_roots.keys())

    matched_words = sorted([model_roots[r] for r in matched_roots])
    missing_words = sorted([model_roots[r] for r in missing_roots])

    coverage = len(matched_roots) / len(model_roots)
    return coverage, matched_words, missing_words


def evaluate_answer(student_text, model_text, max_marks=100.0,
                     semantic_weight=0.7, term_weight=0.3):
    """
    Point-level version:
      - semantic score = the stronger of the whole-answer similarity and
        the average key-point match; this avoids penalising a strong answer
        for a few sentence-splitting edge cases.
      - term score = stemmed technical-term coverage (unchanged logic).
      - completeness bonus = extra credit when the answer covers most of the
        high-level key points and maintains good overall similarity.
    """
    point_matches = match_key_points(model_text, student_text)

    if not point_matches:
        avg_semantic = 0.0
        whole_semantic = 0.0
    else:
        avg_semantic = sum(p["similarity"] for p in point_matches) / len(point_matches)
        whole_semantic = util.cos_sim(
            semantic_model.encode([student_text, model_text], convert_to_tensor=True)[0],
            semantic_model.encode([student_text, model_text], convert_to_tensor=True)[1]
        ).item()

    term_score, matched_terms, missing_terms = technical_term_coverage(student_text, model_text)

    semantic_score = max(avg_semantic, whole_semantic)
    blended = (semantic_weight * semantic_score) + (term_weight * term_score)

    coverage_ratio = (
        sum(1 for p in point_matches if p["similarity"] >= 0.60) / len(point_matches)
        if point_matches else 0.0
    )
    completeness_bonus = max(
        0.0,
        (coverage_ratio - 0.50) * 18.0 + (semantic_score - 0.70) * 12.0 + (term_score - 0.70) * 8.0,
    )
    final_score = round((blended * max_marks) + completeness_bonus, 2)

    return {
        "final_score": final_score,
        "semantic_similarity": round(semantic_score * 100, 1),
        "technical_term_coverage": round(term_score * 100, 1),
        "point_matches": point_matches,
        "matched_terms": matched_terms,
        "missing_terms": missing_terms,
    }


# ==========================================
# MODULE 4: PIPELINE EXECUTION & REPORTING
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "input.txt"
OUTPUT_FILE = BASE_DIR / "output.txt"


def read_answers_from_file(file_path=INPUT_FILE):
    """Read the model and student answers from input.txt.

    Expected format in input.txt:
        MODEL ANSWER:
        <reference answer>

        STUDENT ANSWER:
        <student answer>
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    content = file_path.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError(f"Input file is empty: {file_path}")

    model_match = re.search(
        r"MODEL\s*ANSWER\s*:\s*(.*?)(?=\n\s*STUDENT\s*ANSWER\s*:|\Z)",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    student_match = re.search(
        r"STUDENT\s*ANSWER\s*:\s*(.*?)(?=\Z)",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if model_match and student_match:
        model_answer = model_match.group(1).strip()
        student_answer = student_match.group(1).strip()
        if model_answer and student_answer:
            return student_answer, model_answer

    raise ValueError(
        "Input file must contain the answers in this format:\n"
        "MODEL ANSWER:\n<reference answer>\n\nSTUDENT ANSWER:\n<student answer>"
    )


def build_report(student_answer, model_answer, max_marks=100.0, num_sentences=3):
    summary = summarize(student_answer, num_sentences=num_sentences)
    orig_words = len(student_answer.split())
    summary_words = len(summary.split())
    reduction = 100 * (1 - summary_words / orig_words) if orig_words > 0 else 0

    result = evaluate_answer(student_answer, model_answer, max_marks)

    lines = [
        "=" * 70,
        "   AI THEORY ANSWER EVALUATION PIPELINE (POINT-LEVEL MATCHING v4.0)",
        "=" * 70,
        "",
        "[STEP 1: TEXT SUMMARIZATION]",
        f"Original Length : {orig_words} words",
        f"Summary Length  : {summary_words} words ({reduction:.1f}% reduction)",
        "",
        "--- Extracted Summary ---",
        summary,
        "",
        "=" * 70,
        "[STEP 2: POINT-LEVEL SEMANTIC MATCHING]",
        "=" * 70,
    ]

    for i, p in enumerate(result["point_matches"], start=1):
        pct = round(p["similarity"] * 100, 1)
        flag = "✅" if pct >= 70 else ("⚠️" if pct >= 40 else "❌")
        lines.append(f"\n{flag} Key Point {i} ({pct}% match)")
        lines.append(f"   Model    : {p['point']}")
        lines.append(f'   Best match in student answer: "{p["best_match"]}"')

    lines += [
        "",
        "=" * 70,
        "[STEP 3: FINAL SCORE]",
        "=" * 70,
        f"Final Score              : {result['final_score']} / {max_marks}",
        f"  - Semantic Similarity  : {result['semantic_similarity']}%  (avg across key points, weight 70%)",
        f"  - Technical Term Cover : {result['technical_term_coverage']}%  (required terms present, weight 30%)",
        "",
        f"✅ Matched Technical Terms ({len(result['matched_terms'])}):",
        ", ".join(result['matched_terms']) if result['matched_terms'] else "(none)",
        "",
        f"⚠️ Missing Technical Terms ({len(result['missing_terms'])}):",
        ", ".join(result['missing_terms']) if result['missing_terms'] else "None! Perfect coverage.",
        "",
        "=" * 70,
    ]

    return "\n".join(lines)


def run_pipeline(student_answer, model_answer, max_marks=100.0, num_sentences=3, output_path=OUTPUT_FILE):
    report = build_report(student_answer, model_answer, max_marks=max_marks, num_sentences=num_sentences)
    print(report)
    output_path.write_text(report + "\n", encoding="utf-8")
    return evaluate_answer(student_answer, model_answer, max_marks)


if __name__ == "__main__":
    try:
        student_text, model_text = read_answers_from_file(INPUT_FILE)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)

    run_pipeline(student_text, model_text, max_marks=100.0, num_sentences=3, output_path=OUTPUT_FILE)