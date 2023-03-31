import string
import nltk
from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the tokenizer
with open('news_classifier_tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_json = f.read()
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

# Load the model
model = tf.keras.models.load_model('news_classifier.h5')

# Get a headline from the user as input
headline = input("Enter a headline: ")

# Clean the headline
stop_words = set(stopwords.words("english"))
headline = "".join([word.lower() for word in headline if word.lower() not in string.punctuation])
tokens = nltk.word_tokenize(headline)
clean_headline = [word for word in tokens if word not in stop_words]

# Convert the headline to a sequence of word indices
headline_seq = tokenizer.texts_to_sequences([clean_headline])

# Pad the sequence to a fixed length
headline_padded = pad_sequences(headline_seq, maxlen=1000)

# Use the model to predict the fakeness index score
prediction = model.predict(headline_padded)[0][0]
fakeness_score = 1 - prediction

# Output the prediction and the fakeness score out of 10
if prediction >= 0.5:
    print("The headline is true news.")
else:
    print("The headline is fake news.")
print("Fakeness score out of 10:", round(fakeness_score * 10, 2))
