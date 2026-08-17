"""
AI Theory Answer Evaluation System - v4.1 (Point-Level Matching + Guardrails)
--------------------------------------------------------------------------------
Builds on v4.0 (point-level semantic matching). Adds two guardrails
so the score can't be misleadingly high in edge cases that a live
demo audience WILL try:

1. SHORT-ANSWER GUARDRAIL
   If the student's answer is drastically shorter than the model
   answer, it almost certainly can't cover all the key points in
   real depth -- even if the few sentences it does have happen to
   match well. Caps the max achievable score proportionally.

2. OFF-TOPIC GUARDRAIL
   If NONE of the key points have a reasonably high best-match
   similarity, the student answer is probably off-topic or
   irrelevant. In that case the technical-term score (which can
   coincidentally overlap on common words) is not allowed to prop
   the score up -- the score is capped near the raw semantic floor.

Both guardrails are OFF (no effect) for any normal, reasonably
complete, on-topic answer -- they only kick in for the edge cases
they're designed to catch.

Install requirement (one-time):
    pip install sentence-transformers
"""

import re
import string
import sys
from pathlib import Path
import nltk
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer, util

# Ensure terminal printing supports UTF-8 (emojis) on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

nltk.download('punkt', quiet=True)

stemmer = PorterStemmer()

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

CURATED_QUESTIONS = {
    "descriptive_evaluation": {
        "detect_keywords": ["optical character", "ocr", "preprocessing", "morphological", "stemming", "summarization", "concept matching"],
        "terms": {
            "optical character recognition": ["ocr", "optical character recognition", "scanned", "ingestion"],
            "natural language processing": ["nlp", "natural language processing", "ednlp"],
            "semantic analysis": ["semantic analysis", "concept matching", "semantic matching", "vector space overlap"],
            "text preprocessing": ["preprocessing", "preprocess", "clean tokenization", "tokenization", "tokenize"],
            "extractive summarization": ["summarization", "summarize", "summarizer", "word frequency scoring", "sentence boundary detection"],
            "morphological normalization": ["morphological normalization", "stemming", "lemmatization", "porterstemmer", "root reduction"],
            "concept matching": ["concept matching", "semantic matching", "concept level scoring"],
            "audit reporting": ["audit reporting", "diagnostic report", "evaluation report"]
        }
    },
    "neural_networks": {
        "detect_keywords": ["neural network", "ann", "deep learning", "perceptron", "backpropagation", "activation function", "gradient descent", "hidden layer"],
        "terms": {
            "backpropagation": ["backpropagation", "backprop", "backward pass"],
            "activation function": ["activation function", "sigmoid", "relu", "tanh", "softmax"],
            "gradient descent": ["gradient descent", "gradient", "optimization", "sgd", "adam"],
            "hidden layer": ["hidden layer", "hidden layers"],
            "weights and biases": ["weights", "biases", "weight", "bias"],
            "loss function": ["loss function", "cost function", "mse", "cross entropy"]
        }
    },
    "transformer": {
        "detect_keywords": ["transformer", "attention mechanism", "self-attention", "multi-head attention", "positional encoding"],
        "terms": {
            "self-attention": ["self-attention", "scaled dot-product attention"],
            "multi-head attention": ["multi-head attention", "multihead attention"],
            "encoder": ["encoder", "encoders"],
            "decoder": ["decoder", "decoders"],
            "positional encoding": ["positional encoding", "positional encodings", "position embedding"],
            "feed-forward network": ["feed-forward network", "ffn", "feed forward"]
        }
    }
}


def auto_extract_terms(model_text):
    import nltk
    from nltk import RegexpParser

    # Ensure required NLTK resources are downloaded
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    
    sentences = split_into_sentences(model_text)
    # Define a grammar for noun phrases (adjectives/gerunds followed by nouns)
    grammar = r"NP: {<JJ|VBG>*<NN.*>+}"
    parser = RegexpParser(grammar)
    
    phrases = set()
    for sent in sentences:
        try:
            words = nltk.word_tokenize(sent)
        except LookupError:
            nltk.download('punkt', quiet=True)
            words = nltk.word_tokenize(sent)
            
        tagged = nltk.pos_tag(words)
        tree = parser.parse(tagged)
        
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
            phrase = " ".join(word for word, tag in subtree.leaves())
            phrase_clean = phrase.lower().translate(str.maketrans("", "", string.punctuation)).strip()
            
            # Simple clean tokenization for the phrase
            words_in_phrase = phrase_clean.split()
            if not words_in_phrase:
                continue
                
            # Filter out generic words
            if len(words_in_phrase) == 1:
                w = words_in_phrase[0]
                if len(w) < 4 or w in STOPWORDS or w in {"using", "being", "having", "doing", "make", "take", "same", "many", "such"}:
                    continue
            
            # Ensure the phrase contains at least one meaningful word
            if any(w not in STOPWORDS and len(w) >= 3 and not w.isdigit() for w in words_in_phrase):
                phrases.add(phrase_clean)
                
    # Return mapped to list of itself as fallback variants
    return {p: [p] for p in sorted(phrases)}


# ---- Guardrail thresholds (tune these if they feel too strict/loose) ----
MIN_LENGTH_RATIO = 0.3      # student word count below 30% of model's -> short-answer cap kicks in
OFF_TOPIC_THRESHOLD = 0.25  # avg key-point similarity below 25% -> off-topic cap kicks in


# ==========================================
# MODULE 1: TEXT PREPROCESSING & STEMMING
# ==========================================

def simple_stem(word):
    return stemmer.stem(word.lower())


def split_into_sentences(text):
    text = text.strip().replace("\n", " ")
    protected = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|e\.g|i\.e|etc)\.", r"\1<DOT>", text)
    # Split on punctuation (periods, question marks, exclamation points, and semicolons) followed by space
    sentences = re.split(r"(?<=[.!?;])\s+", protected)
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
# MODULE 3: POINT-LEVEL SEMANTIC MATCHING
# ==========================================

def match_key_points(model_answer, student_answer):
    model_points = split_into_sentences(model_answer)
    student_sentences = split_into_sentences(student_answer)

    if not model_points or not student_sentences:
        return []

    # Generate candidate matches: sentences, sliding window blocks, and paragraph blocks
    candidates = []
    
    # 1. Individual sentences
    candidates.extend(student_sentences)
    
    # 2. Paragraphs (split by double newlines or multiple newlines)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', student_answer) if p.strip()]
    candidates.extend(paragraphs)

    # 3. Sliding windows (pairs and triples of consecutive sentences)
    n = len(student_sentences)
    for i in range(n):
        if i + 1 < n:
            candidates.append(student_sentences[i] + " " + student_sentences[i+1])
        if i + 2 < n:
            candidates.append(student_sentences[i] + " " + student_sentences[i+1] + " " + student_sentences[i+2])

    model_embeddings = semantic_model.encode(model_points, convert_to_tensor=True)
    candidate_embeddings = semantic_model.encode(candidates, convert_to_tensor=True)

    results = []
    for i, point in enumerate(model_points):
        similarities = util.cos_sim(model_embeddings[i], candidate_embeddings)[0]
        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])
        results.append({
            "point": point,
            "best_match": candidates[best_idx],
            "similarity": best_score,
        })

    return results


def technical_term_coverage(student_text, model_text):
    active_curation = None
    model_lower = model_text.lower()
    
    for q_key, q_data in CURATED_QUESTIONS.items():
        if any(keyword in model_lower for keyword in q_data["detect_keywords"]):
            active_curation = q_data["terms"]
            break
            
    if not active_curation:
        active_curation = auto_extract_terms(model_text)

    if not active_curation:
        return 1.0, [], []

    matched_words = []
    missing_words = []

    # Map of typical synonyms in descriptive grading context for fallback use
    synonyms_fallback = {
        "module": ["layer", "stage", "phase", "component"],
        "modules": ["layers", "stages", "phases", "components"],
        "comprises": ["consists", "contains", "includes", "has", "consist", "contain", "include"],
        "combines": ["integrates", "merges", "unites", "joins", "combine", "consists"],
        "utilizes": ["uses", "employs", "applies", "utilize", "use", "employ", "apply"],
        "regardless": ["independent", "irrespective"],
        "academic": ["educational", "university"],
        "responses": ["answers", "scripts", "response", "answer", "script"],
        "feedback": ["report", "diagnostic"],
        "eg": ["such as", "for instance", "example"],
        "which": ["that", "who"],
        "provides": ["gives", "offers", "provide", "give", "offer", "generates", "generate"],
        "redundant": ["unnecessary", "useless", "filler"],
        "analysis": ["analyze", "analyzer", "analyzing", "evaluation"],
    }

    student_roots = extract_concept_roots(student_text)
    student_words = list(student_roots.values())

    for term, variants in active_curation.items():
        is_matched = False
        student_clean = student_text.lower()
        
        # Check direct substring matching for each variant (case-insensitive)
        for variant in variants:
            var_clean = variant.lower()
            if var_clean in student_clean:
                is_matched = True
                break
                
            # Check stem of single-word variants
            if " " not in var_clean:
                var_stem = simple_stem(var_clean)
                for sw in student_words:
                    if simple_stem(sw) == var_stem:
                        is_matched = True
                        break
                if is_matched:
                    break
                    
            # Check prefix overlap for single-word variants
            if " " not in var_clean and len(var_clean) >= 4:
                for sw in student_words:
                    s_lower = sw.lower()
                    common_prefix = ""
                    for char1, char2 in zip(var_clean, s_lower):
                        if char1 == char2:
                            common_prefix += char1
                        else:
                            break
                    if len(common_prefix) >= 4:
                        shorter_len = min(len(var_clean), len(s_lower))
                        if len(common_prefix) / shorter_len >= 0.7:
                            is_matched = True
                            break
                if is_matched:
                    break
                    
            # Check if all stems of a multi-word variant are present in the student answer stems
            if " " in var_clean:
                var_words = var_clean.split()
                var_stems = [simple_stem(vw) for vw in var_words]
                student_stems = [simple_stem(sw) for sw in student_words]
                if all(s in student_stems for s in var_stems):
                    is_matched = True
                    break

        # If it was an auto-extracted word and not matched yet, check standard synonyms map
        if not is_matched and term.lower() in synonyms_fallback:
            for syn in synonyms_fallback[term.lower()]:
                syn_stem = simple_stem(syn)
                if any(syn in sw.lower() or syn_stem == simple_stem(sw) for sw in student_words):
                    is_matched = True
                    break

        if is_matched:
            matched_words.append(term)
        else:
            missing_words.append(term)

    coverage = len(matched_words) / len(active_curation) if active_curation else 1.0
    return coverage, sorted(matched_words), sorted(missing_words)


# ==========================================
# MODULE 4: GUARDRAILS (NEW)
# ==========================================

def check_length_guardrail(student_text, model_text):
    """
    Returns (triggered: bool, ratio: float, capped_max_fraction: float).
    If the student answer is drastically shorter than the model
    answer, cap the max achievable score fraction proportionally --
    a short answer, however well-matched its few sentences are,
    cannot realistically cover everything the model answer does.
    """
    student_words = len(student_text.split())
    model_words = len(model_text.split())

    if model_words == 0:
        return False, 1.0, 1.0

    ratio = student_words / model_words

    if ratio < MIN_LENGTH_RATIO:
        capped_max_fraction = round(ratio / MIN_LENGTH_RATIO, 3)
        return True, round(ratio, 3), capped_max_fraction

    return False, round(ratio, 3), 1.0


def check_off_topic_guardrail(avg_semantic):
    """
    Returns (triggered: bool, capped_score_fraction_or_None).
    If average key-point similarity is very low, the answer is
    probably off-topic. In that case, don't let technical-term
    overlap (which can happen coincidentally on common words) prop
    the score up -- cap the final score fraction at the raw semantic
    similarity itself.
    """
    if avg_semantic < OFF_TOPIC_THRESHOLD:
        return True, avg_semantic
    return False, None


def normalize_spacing(text):
    # Insert space after punctuation if there isn't one (e.g. fast.A -> fast. A)
    text = re.sub(r'([.!?:\)])([a-zA-Z])', r'\1 \2', text)
    # Insert spaces around math delimiters like $ (e.g. $O(1)$Insert -> $ O(1) $ Insert)
    text = re.sub(r'\$', ' $ ', text)
    return re.sub(r'\s+', ' ', text).strip()


def evaluate_answer(student_text, model_text, max_marks=100.0,
                     semantic_weight=0.7, term_weight=0.3):
    student_text = normalize_spacing(student_text)
    model_text = normalize_spacing(model_text)
    
    point_matches = match_key_points(model_text, student_text)

    if not point_matches:
        avg_semantic = 0.0
    else:
        scaled_sims = []
        for p in point_matches:
            raw = p["similarity"]
            # Scale similarity: raw 0.15 maps to 0%, 0.70+ maps to 100%
            scaled = min(1.0, max(0.0, (raw - 0.15) / 0.55))
            scaled_sims.append(scaled)
            p["similarity"] = scaled
            
        avg_semantic = sum(scaled_sims) / len(scaled_sims)

    term_score, matched_terms, missing_terms = technical_term_coverage(student_text, model_text)

    blended_fraction = (semantic_weight * avg_semantic) + (term_weight * term_score)

    # ---- Apply guardrails ----
    guardrail_notes = []

    length_triggered, length_ratio, length_cap_fraction = check_length_guardrail(student_text, model_text)
    if length_triggered:
        orig_blended = blended_fraction
        blended_fraction = min(blended_fraction, length_cap_fraction)
        guardrail_notes.append(
            f"Short-answer guardrail triggered: student answer is only {length_ratio*100:.0f}% of the "
            f"model answer's length. Score capped at {length_cap_fraction*100:.0f}% of max marks "
            f"(was tracking toward {orig_blended*100:.1f}%)."
        )

    off_topic_triggered, off_topic_cap = check_off_topic_guardrail(avg_semantic)
    if off_topic_triggered:
        orig_blended = blended_fraction
        blended_fraction = min(blended_fraction, off_topic_cap)
        guardrail_notes.append(
            f"Off-topic guardrail triggered: average key-point similarity is only {avg_semantic*100:.1f}%, "
            f"below the {OFF_TOPIC_THRESHOLD*100:.0f}% relevance threshold. Technical-term overlap "
            f"is not allowed to raise the score above the raw semantic match "
            f"(was tracking toward {orig_blended*100:.1f}%)."
        )

    final_score = round(blended_fraction * max_marks, 2)

    return {
        "final_score": final_score,
        "semantic_similarity": round(avg_semantic * 100, 1),
        "technical_term_coverage": round(term_score * 100, 1),
        "point_matches": point_matches,
        "matched_terms": matched_terms,
        "missing_terms": missing_terms,
        "guardrail_notes": guardrail_notes,
        "length_ratio": length_ratio,
    }


# ==========================================
# MODULE 5: PIPELINE EXECUTION & REPORTING
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "input.txt"
OUTPUT_FILE = BASE_DIR / "output.txt"


def read_answers_from_file(file_path=INPUT_FILE):
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
        "  AI THEORY ANSWER EVALUATION PIPELINE (POINT-LEVEL MATCHING v4.1)",
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
        "[STEP 3: GUARDRAILS]",
        "=" * 70,
    ]
    if result["guardrail_notes"]:
        for note in result["guardrail_notes"]:
            lines.append(f"🚩 {note}")
    else:
        lines.append("No guardrails triggered -- answer is a normal length and on-topic.")

    lines += [
        "",
        "=" * 70,
        "[STEP 4: FINAL SCORE]",
        "=" * 70,
        f"Final Score              : {result['final_score']} / {max_marks}",
        f"  - Semantic Similarity  : {result['semantic_similarity']}%  (avg across key points, weight 70%)",
        f"  - Technical Term Cover : {result['technical_term_coverage']}%  (required terms present, weight 30%)",
        f"  - Answer Length Ratio  : {result['length_ratio']*100:.0f}%  (student words / model words)",
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