
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def summarize_text(text, num_sentences=3):
    sentences = sent_tokenize(text)
    
    if len(sentences) <= num_sentences:
        return text  # Nothing to summarize

    # Convert sentences to vectors
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences)

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(vectors)

    # Rank sentences based on sum of similarities
    scores = similarity_matrix.sum(axis=1)
    ranked_sentences = [sentences[i] for i in np.argsort(scores)[-num_sentences:]]

    # Return summary
    return ' '.join(ranked_sentences)

if __name__ == "__main__":
    print("=== TEXT SUMMARIZER ===")
    user_input = input("Paste your article/text here:\n\n")
    summary = summarize_text(user_input, num_sentences=3)
    print("\n--- Summary ---\n")
    print(summary)
