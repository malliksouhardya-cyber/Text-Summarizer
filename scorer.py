"""
Answer Scorer Prototype (Built From Scratch)
---------------------------------------------
Scores a student's answer against a model/reference answer using
keyword overlap. No pretrained models or ML libraries are used —
this is a simple, inspectable baseline you can later swap for
embeddings (SBERT) once the pipeline logic is proven.

How it works (the algorithm):
1. Clean and tokenize both the student answer and the model answer.
2. Remove stopwords from both, leaving only "content" words.
3. Find which model-answer words also appear in the student's answer
   (matched concepts) and which don't (missing concepts).
4. Score = (matched concepts / total concepts in model answer) * total_marks.
5. Print a breakdown: score, matched concepts, missing concepts.

Note: this rewards keyword presence, not sentence-level correctness
or logical flow. It's a fast MVP scorer — good enough to demo the
pipeline end-to-end before upgrading to semantic (embedding-based)
scoring.
"""

import nltk
import re
from nltk.corpus import stopwords

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

stop_words = set(stopwords.words('english'))


def clean_and_tokenize(text):
    """Lowercase, strip non-letters, split into words, drop stopwords."""
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    return [w for w in words if w not in stop_words]


def score_answer(student_text, model_text, total_marks):
    student_words = set(clean_and_tokenize(student_text))
    model_words = set(clean_and_tokenize(model_text))

    if not model_words:
        return 0, set(), set()

    matched = student_words & model_words
    missing = model_words - student_words

    coverage = len(matched) / len(model_words)
    score = round(coverage * total_marks, 1)

    return score, matched, missing


def print_report(score, total_marks, matched, missing):
    print("\n" + "=" * 55)
    print("SCORING REPORT")
    print("=" * 55)
    print(f"Score: {score} / {total_marks}")
    print(f"\nMatched concepts ({len(matched)}):")
    print(", ".join(sorted(matched)) if matched else "  (none)")
    print(f"\nMissing concepts ({len(missing)}):")
    print(", ".join(sorted(missing)) if missing else "  (none)")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    print("Enter the MODEL ANSWER (reference/key answer):")
    model_answer = input()

    print("\nEnter the STUDENT ANSWER:")
    student_answer = input()

    print("\nEnter the TOTAL MARKS for this question:")
    total_marks = float(input())

    score, matched, missing = score_answer(student_answer, model_answer, total_marks)
    print_report(score, total_marks, matched, missing)