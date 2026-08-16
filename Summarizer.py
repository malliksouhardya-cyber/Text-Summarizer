"""
Text Summarizer Prototype (Built From Scratch)
------------------------------------------------
An extractive summarizer: it scores each sentence in the input text based on
word-frequency importance, then picks the top-N highest scoring sentences to
form the summary. No pretrained models or ML libraries are used — just pure
Python logic — so it's easy to inspect, modify, and "train" (tune weights,
add features, swap scoring rules) yourself.

How it works (the algorithm):
1. Split text into sentences.
2. Split text into words, remove common "stopwords" (the, is, and, etc.),
   and count how often each remaining word appears.
3. Score each sentence by summing the frequency scores of the words it
   contains (normalized by sentence length so long sentences don't win
   just because they have more words).
4. Sort sentences by score, pick the top N, and reassemble them in their
   original order to form the final summary.

This is the same core idea behind classic algorithms like Luhn's method.
You can extend it later (e.g. add TF-IDF, position weighting, or plug in
your own trained model) since every step is transparent and editable.
"""

import re
import string

# A small built-in stopword list so this script has zero external
# dependencies. Feel free to expand this list.
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


def split_into_sentences(text):
    """
    Splits raw text into a list of sentences.
    Uses a simple regex on sentence-ending punctuation (. ! ?) while
    trying to avoid breaking on common abbreviations like 'Mr.' or 'e.g.'.
    """
    text = text.strip().replace("\n", " ")
    # Basic abbreviation guard: temporarily protect a few common ones
    protected = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|e\.g|i\.e|etc)\.", r"\1<DOT>", text)
    sentences = re.split(r"(?<=[.!?])\s+", protected)
    sentences = [s.replace("<DOT>", ".").strip() for s in sentences if s.strip()]
    return sentences


def clean_and_tokenize(sentence):
    """Lowercases, strips punctuation, and splits a sentence into words."""
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    words = sentence.split()
    return words


def build_word_frequencies(sentences):
    """
    Builds a dictionary of {word: frequency_score} across the whole text,
    ignoring stopwords, then normalizes scores to a 0-1 range so the
    most frequent meaningful word has a score of 1.0.
    """
    freq = {}
    for sentence in sentences:
        for word in clean_and_tokenize(sentence):
            if word in STOPWORDS or word.isdigit():
                continue
            freq[word] = freq.get(word, 0) + 1

    if not freq:
        return {}

    max_freq = max(freq.values())
    for word in freq:
        freq[word] = freq[word] / max_freq

    return freq


def score_sentences(sentences, word_freq):
    """
    Scores each sentence by summing the frequency scores of its words,
    normalized by the number of words in the sentence (so short sentences
    packed with important words score well, and long sentences aren't
    automatically favored just for being long).
    """
    scores = []
    for sentence in sentences:
        words = clean_and_tokenize(sentence)
        if not words:
            scores.append(0)
            continue
        sentence_score = sum(word_freq.get(w, 0) for w in words)
        normalized_score = sentence_score / len(words)
        scores.append(normalized_score)
    return scores


def summarize(text, num_sentences=3):
    """
    Main summarizer function.
    Returns a summary string made of the top `num_sentences` sentences,
    kept in their original order.
    """
    sentences = split_into_sentences(text)

    if len(sentences) <= num_sentences:
        return text.strip()  # Nothing to trim; text is already short.

    word_freq = build_word_frequencies(sentences)
    scores = score_sentences(sentences, word_freq)

    # Pair each sentence with its score and original position
    ranked = sorted(
        enumerate(scores), key=lambda pair: pair[1], reverse=True
    )
    top_indices = sorted(idx for idx, _ in ranked[:num_sentences])

    summary = " ".join(sentences[i] for i in top_indices)
    return summary


def get_user_input():
    """Collects text and desired summary length from the user via console."""
    print("=" * 60)
    print("TEXT SUMMARIZER (extractive, built from scratch)")
    print("=" * 60)
    print("\nPaste or type the text you want to summarize.")
    print("When finished, press Enter on an empty line:\n")

    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    text = "\n".join(lines)

    num_sentences = 3  # Default summary length; change this value to adjust.

    return text, num_sentences


def main():
    text, num_sentences = get_user_input()

    if not text.strip():
        print("\nNo text provided. Exiting.")
        return

    summary = summarize(text, num_sentences)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(summary)
    print("=" * 60)

    original_words = len(text.split())
    summary_words = len(summary.split())
    if original_words > 0:
        reduction = 100 * (1 - summary_words / original_words)
        print(
            f"\nOriginal: {original_words} words  ->  "
            f"Summary: {summary_words} words  "
            f"({reduction:.1f}% shorter)"
        )


if __name__ == "__main__":
    main()