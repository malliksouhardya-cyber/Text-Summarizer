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
