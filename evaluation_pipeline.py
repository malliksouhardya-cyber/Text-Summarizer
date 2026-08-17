"""
AI Theory Answer Evaluation System - v3.0 (Semantic + Stemmed Hybrid)
----------------------------------------------------------------------
Adds SBERT (Sentence-BERT) semantic similarity on top of the existing
stemmed keyword pipeline.

Why this exists:
Root-matching (stemming) only fixes word-FORM differences (plural,
tense). It still can't tell that two DIFFERENT sentences mean the
SAME thing. If your professor writes "an optimization algorithm
updates the weights" and you write "the network adjusts its weights
to learn", stemming sees almost no overlap -- even though you said
the same thing in your own words.

SBERT fixes this: it converts a whole sentence/paragraph into a
vector that captures its MEANING, not just its words. Comparing two
such vectors with cosine similarity tells you how close in meaning
two answers are, regardless of phrasing, sentence structure, or
vocabulary choice.

Final score = a blend of:
  - Semantic similarity  (does the overall MEANING match? - the main score)
  - Technical term coverage (are the required technical terms present at all?)

This mirrors how a human evaluator actually grades: you can explain
in your own words, but you can't skip the technical terms.

Install requirement (one-time, only needs to be run once):
    pip install sentence-transformers

First run will download a small pretrained model (~90MB) from
Hugging Face -- needs an internet connection the first time only,
it's cached locally after that.
"""

import re
import string
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
}


# ==========================================
# MODULE 1: TEXT PREPROCESSING & STEMMING
# (unchanged from your v2.1 -- still used for the technical-term audit)
# ==========================================

def simple_stem(word):
    return stemmer.stem(word.lower())


def split_into_sentences(text):
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
# (unchanged from your v2.1)
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
# MODULE 3: SEMANTIC SCORING (NEW)
# ==========================================

def semantic_similarity(student_text, model_text):
    """
    Encodes both texts into embeddings and returns their cosine
    similarity (0 to 1). This is meaning-based, not word-based --
    paraphrasing, synonyms, and different sentence structure don't
    hurt the score as long as the underlying idea matches.
    """
    embeddings = semantic_model.encode([student_text, model_text], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return similarity


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


def evaluate_answer(student_text, model_text, max_marks=10.0,
                     semantic_weight=0.7, term_weight=0.3):
    """
    Blended score:
      - semantic_weight (default 70%) rewards matching the MEANING,
        regardless of phrasing -- this is what lets you use your own
        words freely.
      - term_weight (default 30%) rewards having the required
        technical terms present, since those shouldn't be paraphrased.

    Tune the weights based on how strict you want term-matching to
    be relative to overall understanding.
    """
    sem_score = semantic_similarity(student_text, model_text)
    term_score, matched_terms, missing_terms = technical_term_coverage(student_text, model_text)

    blended = (semantic_weight * sem_score) + (term_weight * term_score)
    final_score = round(blended * max_marks, 2)

    return {
        "final_score": final_score,
        "semantic_similarity": round(sem_score * 100, 1),
        "technical_term_coverage": round(term_score * 100, 1),
        "matched_terms": matched_terms,
        "missing_terms": missing_terms,
    }


# ==========================================
# MODULE 4: PIPELINE EXECUTION & REPORTING
# ==========================================

def run_pipeline(student_answer, model_answer, max_marks=10.0, num_sentences=3):
    print("=" * 65)
    print("   AI THEORY ANSWER EVALUATION PIPELINE (SEMANTIC + STEMMED v3.0)")
    print("=" * 65)

    summary = summarize(student_answer, num_sentences=num_sentences)
    orig_words = len(student_answer.split())
    summary_words = len(summary.split())
    reduction = 100 * (1 - summary_words / orig_words) if orig_words > 0 else 0

    print("\n[STEP 1: TEXT SUMMARIZATION]")
    print(f"Original Length : {orig_words} words")
    print(f"Summary Length  : {summary_words} words ({reduction:.1f}% reduction)")
    print("\n--- Extracted Summary ---")
    print(summary)

    result = evaluate_answer(student_answer, model_answer, max_marks)

    print("\n" + "=" * 65)
    print("[STEP 2: SEMANTIC + TECHNICAL-TERM SCORING]")
    print("=" * 65)
    print(f"Final Score              : {result['final_score']} / {max_marks}")
    print(f"  - Semantic Similarity  : {result['semantic_similarity']}%  (meaning match, weight 70%)")
    print(f"  - Technical Term Cover : {result['technical_term_coverage']}%  (required terms present, weight 30%)")

    print(f"\n✅ Matched Technical Terms ({len(result['matched_terms'])}):")
    print(", ".join(result['matched_terms']) if result['matched_terms'] else "(none)")

    print(f"\n⚠️ Missing Technical Terms ({len(result['missing_terms'])}):")
    print(", ".join(result['missing_terms']) if result['missing_terms'] else "None! Perfect coverage.")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    student_text = (
        "An Artificial Neural Network (ANN) is a computational model inspired by the human "
        "brain's biological neural networks. An ANN consists of three main layers: the input "
        "layer, hidden layers, and the output layer. The input layer receives raw data signals, "
        "such as image pixels or numerical features. Hidden layers process these inputs through "
        "weighted connections and non-linear activation functions like ReLU or Sigmoid. During the "
        "forward pass, predictions are generated by transferring signals across layers. "
        "Backpropagation is the core learning algorithm used to update weights by calculating "
        "the gradient of the loss function. Optimization algorithms like Adam or Stochastic Gradient "
        "Descent adjust these weights to minimize overall prediction errors. Deep neural networks "
        "contain multiple hidden layers, enabling them to learn highly complex patterns in "
        "unstructured data."
    )

    model_text = (
        "An Artificial Neural Network (ANN) is a computational model inspired by the human brain. "
        "It consists of three primary layers: an input layer that receives data features, one or "
        "more hidden layers that transform information using weighted connections and non-linear "
        "activation functions (such as ReLU or Sigmoid), and an output layer that produces the "
        "final prediction. Key mechanisms include forward propagation for generating output and "
        "backpropagation for updating network weights using optimization algorithms like Gradient "
        "Descent to minimize loss. Deep neural networks use multiple hidden layers to automatically "
        "extract hierarchical representations from complex, unstructured data."
    )

    run_pipeline(student_text, model_text, max_marks=10.0, num_sentences=3)