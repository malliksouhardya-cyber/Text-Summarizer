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


def evaluate_answer(student_text, model_text, max_marks=100.0,
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

def run_pipeline(student_answer, model_answer, max_marks=100.0, num_sentences=3):
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
        """Artificial Intelligence and Its Impact on Society

Artificial intelligence, commonly known as AI, has become one of the most important technologies of the modern world. It is a branch of computer science that focuses on creating machines and software capable of performing tasks that normally require human intelligence. These tasks can include understanding language, recognizing images, solving problems, making predictions, learning from information, and making decisions. Over the past few years, AI has developed rapidly and has become a part of many people's daily lives. From smartphones and search engines to education, healthcare, transportation, and business, artificial intelligence is changing the way people work, communicate, learn, and solve problems.

One of the most common uses of AI can be found in smartphones and digital applications. Voice assistants can understand spoken commands and provide answers to questions. Many applications use AI to recommend music, movies, videos, products, and other content based on a user's interests. Search engines also use intelligent algorithms to understand what people are looking for and provide relevant results. These technologies save time and make digital services more convenient. Although people may not always notice it, artificial intelligence is often working in the background of many technologies they use every day.

Education is another field where artificial intelligence can provide significant benefits. Students can use AI-powered learning tools to understand difficult concepts, practice questions, summarize information, and receive explanations. Teachers can also use AI to prepare educational materials, organize information, and develop learning activities. Intelligent systems can analyze a student's performance and identify subjects where additional practice may be useful. This can make education more personalized. However, students should use AI as a learning assistant rather than depending on it completely. Developing independent thinking, creativity, communication skills, and problem-solving abilities remains extremely important.

Healthcare is also being influenced by artificial intelligence. Modern healthcare produces large amounts of information, including medical records, research data, and diagnostic images. AI systems can process large amounts of information quickly and help medical professionals identify patterns. Researchers can use AI to study diseases, analyze scientific data, and explore possible treatments. AI can also support administrative tasks in hospitals and healthcare organizations. However, healthcare decisions are extremely important, so AI systems should be carefully tested and used under appropriate human supervision. Technology should support healthcare professionals rather than replace responsible human judgment.

Artificial intelligence is also transforming businesses and workplaces. Companies can use AI to analyze large datasets, understand customer preferences, automate repetitive tasks, and improve their services. Customer-support systems can answer frequently asked questions and help users find information. Businesses can also use AI to identify patterns in sales and predict future demand. By automating certain repetitive activities, AI can allow employees to focus on more creative and complex responsibilities. At the same time, automation may change the nature of some jobs. Therefore, workers may need to learn new skills and adapt to changing technologies.

Transportation is another area where AI has an important role. Navigation applications can analyze traffic information and recommend suitable routes. Transportation companies can use AI to improve scheduling, logistics, and resource management. Researchers are also developing increasingly advanced driver-assistance and autonomous transportation technologies. These systems are designed to improve efficiency and safety, but they require extensive testing because transportation involves significant real-world risks. Human responsibility and appropriate safety standards remain important as these technologies continue to develop.

Despite its many benefits, artificial intelligence also creates several challenges. One major concern is privacy. AI systems often depend on large amounts of data, and some of that data may contain personal information. Organizations must therefore handle data responsibly and take appropriate steps to protect people's privacy. People should also understand what information they are sharing with digital services. Strong security practices and clear policies can help reduce the risks associated with improper data use.

Another challenge is that AI systems can sometimes produce inaccurate or biased results. Artificial intelligence learns from data, and if the training data contains errors or unfair patterns, the resulting system may reproduce those problems. For this reason, developers need to carefully test AI systems and evaluate their performance. Human oversight is especially important when AI is used for decisions that can significantly affect people's lives. Developers should also try to make AI systems as transparent and understandable as possible.

The rapid development of AI also raises questions about employment. Automation may reduce the need for people to perform certain repetitive tasks, while creating new opportunities in areas such as software development, data analysis, AI research, cybersecurity, and technology management. The workplace of the future may therefore require different skills from those needed today. Education and continuous learning can help people adapt to these changes. Instead of viewing technology only as a replacement for human workers, it can also be understood as a tool that allows people to perform their work more effectively.

Ethics is another important part of artificial intelligence. Developers and organizations need to consider how their systems might affect individuals and society. AI should be developed and used responsibly, with attention to fairness, privacy, safety, transparency, and accountability. Governments, researchers, technology companies, educational institutions, and communities all have a role to play in creating responsible approaches to AI. Clear guidelines and appropriate regulations can help encourage innovation while reducing potential risks.

Artificial intelligence is also changing scientific research and innovation. Scientists can use AI to process complex information, identify patterns, and develop models that would be difficult to create manually. AI can assist researchers in areas such as climate science, astronomy, engineering, biology, and many other disciplines. As computing power and available data continue to increase, AI may become an even more valuable research tool. However, scientific conclusions still need careful verification because AI-generated results are not automatically correct.

For young people, understanding artificial intelligence is becoming increasingly valuable. Students who learn programming, mathematics, data analysis, communication, and critical thinking can develop skills that are useful in an AI-driven world. It is also important to understand the limitations of AI. An AI system may generate an answer that sounds convincing but contains incorrect information. Therefore, people should learn to verify important information rather than accepting every AI-generated response without question.

The future of artificial intelligence will depend largely on how humans choose to develop and use it. AI has the potential to improve productivity, support scientific discoveries, assist education, improve services, and solve many complex problems. At the same time, privacy concerns, misinformation, bias, security, employment changes, and ethical questions must be addressed carefully. The goal should not simply be to create increasingly powerful AI systems, but to create systems that are useful, reliable, safe, and beneficial to society. With responsible development, human oversight, proper education, and thoughtful regulation, artificial intelligence can become a powerful tool that works alongside people and contributes positively to the future."""
    )

    model_text = (
       """Artificial Intelligence (AI) is a branch of computer science that enables machines and software to perform tasks that normally require human intelligence, such as learning, problem-solving, understanding language, recognizing images, and making decisions. AI has become an important part of modern society and is used in smartphones, education, healthcare, business, transportation, and scientific research.

In education, AI can help students understand difficult concepts, practice questions, summarize information, and receive personalized assistance. Teachers can use AI to prepare educational materials and analyze student performance. In healthcare, AI can process large amounts of medical information, identify patterns, assist researchers, and support healthcare professionals. However, human supervision is necessary because AI-generated results may not always be accurate.

Businesses use AI to analyze data, understand customer preferences, automate repetitive tasks, improve customer service, and predict future demand. In transportation, AI can analyze traffic, improve logistics, recommend routes, and support the development of advanced driver-assistance systems. AI is also useful in scientific research because it can process complex datasets and help researchers identify patterns.

Despite its advantages, AI has several challenges. Privacy is an important concern because AI systems may require large amounts of personal data. AI can also produce inaccurate or biased results if the data used to develop the system contains errors or unfair patterns. Automation may change some jobs while creating new opportunities in areas such as software development, data analysis, cybersecurity, and AI research.

Therefore, AI should be developed and used responsibly. Important principles include fairness, privacy, safety, transparency, accountability, and human oversight. Students should develop skills such as programming, critical thinking, communication, and data analysis to prepare for an AI-driven future. Overall, artificial intelligence has great potential to benefit society, but its risks must be carefully managed so that it remains a safe, reliable, and useful technology."""
    )

    run_pipeline(student_text, model_text, max_marks=100.0, num_sentences=3)