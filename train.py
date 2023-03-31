# Pandas is used for data manipulation
import pandas as pd
# Neural network library
import tensorflow as tf
# Keras is used for creating the neural network
from tensorflow import keras
# Numpy is used for working with arrays
import numpy as np
# Matplotlib is used for plotting graphs
import matplotlib.pyplot as plt
# Sklearn is used for splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
# Sklearn is used for calculating the accuracy of the model
from sklearn.metrics import accuracy_score
# Sklearn is used for calculating the confusion matrix
from sklearn.metrics import confusion_matrix
# Sklearn is used for calculating the precision and recall
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Flatten, GlobalMaxPooling1D
# For making clouds
from word_cloud.word_cloud_generator import WordCloud
import matplotlib.pyplot as plt
from IPython.core.display import HTML


# Create an empty dataframe
df = pd.DataFrame(columns=['text', 'label'])

# First folder contains the txt files of news articles for the true news
# Iterate through the files in the folder
# Read the file and store the text in a list
# Append the list to a dataframe
# Add a label column to the dataframe and set the value to 1
path = "True/"
for filename in os.listdir(path):
    with open(os.path.join(path, filename), "r", encoding='utf-8') as f:
        text = f.read()
        df = df.append({'text': text, 'label': 1}, ignore_index=True)

# Second folder contains the txt files of news articles for the fake news
# Iterate through the files in the folder
# Read the file and store the text in a list
# Append the list to the same dataframe with a label of 0
path = "Fake/"
for filename in os.listdir(path):
    with open(os.path.join(path, filename), "r", encoding='utf-8') as f:
        text = f.read()
        df = df.append({'text': text, 'label': 0}, ignore_index=True)

ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "said","you", "your", "yours", "yourself",
    "yourselves"])

# Clouds
# Create a word cloud of the true news
true_text = df[df['label'] == 1]['text'].values
# true_text = ' '.join(true_text)

# initialize WordCloud
true_wordcloud=WordCloud(stopwords=ENGLISH_STOP_WORDS)

# get html code
embed_code=true_wordcloud.get_embed_code(text=true_text,random_color=True,topn=40)

# display
HTML(embed_code)

# Create a word cloud of the fake news
fake_text = df[df['label'] == 0]['text'].values
fake_text = ' '.join(fake_text)
fake_wordcloud = WordCloud().generate(fake_text)

# Plot the word clouds
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(true_wordcloud, interpolation='bilinear')
ax1.axis("off")
ax1.set_title('True News')
ax2.imshow(fake_wordcloud, interpolation='bilinear')
ax2.axis("off")
ax2.set_title('Fake News')
plt.show()


# Text preprocessing
max_words = 20000
max_len = 300
tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
tokenizer.fit_on_texts(df['text'].values)

X = tokenizer.texts_to_sequences(df['text'].values)
X = pad_sequences(X, maxlen=max_len)

# Define the target variable 'y' as the 'label' column of the DataFrame
y = df['label'].astype(float)

# Use the preprocessed text for train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# Create tensorflow datasets from the preprocessed text data
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train.values))
train_ds = train_ds.shuffle(buffer_size=len(X_train)).batch(32)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val.values))
val_ds = val_ds.batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test.values))
test_ds = test_ds.batch(32)

# Update the model architecture
model = keras.Sequential([
    # Embedding layer
    Embedding(max_words, 64, input_length=max_len),
    # GlobalMaxPooling1D layer
    GlobalMaxPooling1D(),
    # Dense layer with 256 neurons and relu activation function
    keras.layers.Dense(256, activation='relu'),
    # Batch normalization layer
    keras.layers.BatchNormalization(),
    # Dropout layer
    keras.layers.Dropout(0.5),
    # Dense layer with 128 neurons and relu activation function
    keras.layers.Dense(128, activation='relu'),
    # Batch normalization layer
    keras.layers.BatchNormalization(),
    # Dropout layer
    keras.layers.Dropout(0.5),
    # Dense layer with 64 neurons and relu activation function
    keras.layers.Dense(64, activation='relu'),
    # Batch normalization layer
    keras.layers.BatchNormalization(),
    # Dropout layer
    keras.layers.Dropout(0.5),
    # Dense layer with 32 neurons and relu activation function
    keras.layers.Dense(32, activation='relu'),
    # Batch normalization layer
    keras.layers.BatchNormalization(),
    # Output layer with sigmoid activation function
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(),
)

# Define early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the model with early stopping
history = model.fit(
    train_ds,
    epochs=100,
    validation_data=val_ds,
    callbacks=[early_stopping]
)

# Evaluate the model
y_test_pred = (model.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy:", accuracy)

# Make predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1}")
