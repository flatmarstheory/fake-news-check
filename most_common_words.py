import os
import string
import nltk
from nltk.corpus import stopwords
from collections import Counter

nltk.download("stopwords")

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

    # Calculate the total word count and the frequency of each word in each case
    fake_news_word_counts = Counter(fake_news_words)
    true_news_word_counts = Counter(true_news_words)

    fake_news_total_words = sum(fake_news_word_counts.values())
    true_news_total_words = sum(true_news_word_counts.values())

    # Calculate the average weightage of each word in each case and store the 20 most common words in each case in a list
    fake_news_top_words = []
    fake_news_top_counts = []
    true_news_top_words = []
    true_news_top_counts = []

    for word, count in fake_news_word_counts.most_common():
        weightage = count / fake_news_total_words
        if word not in true_news_word_counts:
            fake_news_top_words.append(word)
            fake_news_top_counts.append(weightage)
        else:
            true_news_weightage = true_news_word_counts[word] / true_news_total_words
            if weightage > true_news_weightage:
                fake_news_top_words.append(word)
                fake_news_top_counts.append(weightage)

    for word, count in true_news_word_counts.most_common():
        weightage = count / true_news_total_words
        if word not in fake_news_word_counts:
            true_news_top_words.append(word)
            true_news_top_counts.append(weightage)
        else:
            fake_news_weightage = fake_news_word_counts[word] / fake_news_total_words
            if weightage > fake_news_weightage:
                true_news_top_words.append(word)
                true_news_top_counts.append(weightage)

    # Return the most common words in each case as two separate lists
    return fake_news_top_words[:20], fake_news_top_counts[:20], true_news_top_words[:20], true_news_top_counts[:20]
