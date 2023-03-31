import os
import string
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter


def most_common_words(fake_path, true_path):
    # Set up empty lists to hold the text contents of each file
    fake_news_text = []
    true_news_text = []

    # Loop through all files in the Fake folder and add their contents to the fake_news_text list
    for filename in os.listdir(fake_path):
        if filename.endswith(".txt"):
            with open(os.path.join(fake_path, filename), "r", encoding="utf-8") as f:
                text = f.read()
                fake_news_text.append(text)

    # Loop through all files in the True folder and add their contents to the true_news_text list
    for filename in os.listdir(true_path):
        if filename.endswith(".txt"):
            with open(os.path.join(true_path, filename), "r", encoding="utf-8") as f:
                text = f.read()
                true_news_text.append(text)

    # Tokenize the text into words and remove punctuation and stop words
    stop_words = set(stopwords.words("english"))
    fake_news_words = []
    true_news_words = []

    for text in fake_news_text:
        tokens = nltk.word_tokenize(text.lower())
        fake_news_words.extend([word for word in tokens if word.isalpha() and word not in stop_words])

    for text in true_news_text:
        tokens = nltk.word_tokenize(text.lower())
        true_news_words.extend([word for word in tokens if word.isalpha() and word not in stop_words])

    # Count the frequency of each word and store the 20 most common words in each case in a list
    fake_news_word_counts = Counter(fake_news_words)
    true_news_word_counts = Counter(true_news_words)

    fake_news_top_words = [word for word, count in fake_news_word_counts.most_common(20)]
    true_news_top_words = [word for word, count in true_news_word_counts.most_common(20)]

    # Create a set of all the most common words in both cases
    common_words = set(fake_news_top_words).union(set(true_news_top_words))

    # Create a dictionary of word frequencies in both cases
    word_freqs = {}
    for word in common_words:
        fake_news_count = fake_news_word_counts.get(word, 0)
        true_news_count = true_news_word_counts.get(word, 0)
        word_freqs[word] = [fake_news_count, true_news_count]

    # Pickle the word frequency dictionary for later use
    with open("word_freqs.pickle", "wb") as f:
        pickle.dump(word_freqs, f)

    # Return the most common words in each case as two separate lists
    return fake_news_top_words, true_news_top_words


def main():
    fake_path = os.path.join(".", "Fake")
    true_path = os.path.join(".", "True")

    fake_news_top_words, true_news_top_words = most_common_words(fake_path, true_path)

    # Train your ML model here using the word_freqs dictionary

if __name__ == "__main__":
    main()
