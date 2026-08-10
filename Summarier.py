import nltk
import numpy as np
import networkx as nx
import re
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab') # Added to resolve LookupError

stop_words = stopwords.words('english')

def sentence_similarity(sent1, sent2):
    words = list(set(sent1 + sent2))
    word_index = {word: i for i, word in enumerate(words)}  # NEW: build once

    vector1 = [0] * len(words)
    vector2 = [0] * len(words)

    for word in sent1:
        if word not in stop_words:
            vector1[word_index[word]] += 1   # CHANGED: dict lookup instead of .index()

    for word in sent2:
        if word not in stop_words:
            vector2[word_index[word]] += 1   # CHANGED: dict lookup instead of .index()

    if sum(vector1) == 0 or sum(vector2) == 0:
        return 0

    return 1 - cosine_distance(vector1, vector2)

# Create similarity matrix
def build_similarity_matrix(sentences):
    matrix = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                matrix[i][j] = sentence_similarity(sentences[i], sentences[j])

    return matrix

# Generate summary
def generate_summary(text, top_n=3):

    # Split into sentences
    original_sentences = nltk.sent_tokenize(text)

    # Clean sentences
    clean_sentences = []
    for sentence in original_sentences:
        sentence = re.sub(r'[^a-zA-Z]', ' ', sentence)
        clean_sentences.append(sentence.lower().split())

    # Build similarity matrix
    similarity_matrix = build_similarity_matrix(clean_sentences)

    # Apply PageRank
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)

    # Rank sentences
    ranked_sentences = sorted(
        ((scores[i], s) for i, s in enumerate(original_sentences)),
        reverse=True
    )

    # Print summary
    print("\n===== SUMMARY =====\n")
    for i in range(min(top_n, len(ranked_sentences))):
        print(ranked_sentences[i][1])

# Main Program
if __name__ == "__main__":

    print("Enter the paragraph to summarize:")
    text = input()

    generate_summary(text, 3)