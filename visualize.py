import os
from most_common_words import most_common_words
from related_words import related_words
import nltk
nltk.download('punkt')
import matplotlib.pyplot as plt


fake_news_top_words, fake_news_top_counts, true_news_top_words, true_news_top_counts = most_common_words("Fake", "True")

fig, ax = plt.subplots()
ax.barh(range(len(fake_news_top_words)), fake_news_top_counts, align="center", color="red", label="Fake News")
ax.barh(range(len(true_news_top_words)), true_news_top_counts, align="center", color="blue", label="True News")
ax.set_yticks(range(len(fake_news_top_words)))
ax.set_yticklabels(fake_news_top_words)
ax.invert_yaxis()
ax.set_xlabel("Word Count")
ax.set_title("Most Popular Buzzwords in Fake News vs. True News")
ax.legend()
plt.show()

G = related_words("Fake", "True")
