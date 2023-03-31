import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Define constants
MAX_SEQUENCE_LENGTH = 1000  # Maximum sequence length of text data
MAX_NUM_WORDS = 20000  # Maximum number of words in the vocabulary
EMBEDDING_DIM = 100  # Dimension of the word embedding vector
BATCH_SIZE = 32  # Batch size for training
NUM_EPOCHS = 10  # Number of epochs to train for

# Load the text data
fake_dir = "Fake"
true_dir = "True"
fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(".txt")]
true_files = [os.path.join(true_dir, f) for f in os.listdir(true_dir) if f.endswith(".txt")]

texts = []
labels = []
for fname in fake_files:
    with open(fname, encoding="utf8") as f:
        texts.append(f.read())
    labels.append(0)  # Label 0 for fake news

for fname in true_files:
    with open(fname, encoding="utf8") as f:
        texts.append(f.read())
    labels.append(1)  # Label 1 for true news

# Tokenize the text data and create sequences
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad the sequences to a fixed length
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Shuffle the data and split into training and validation sets
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = np.array(labels)[indices]

num_validation_samples = int(0.2 * data.shape[0])
x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

# Save the tokenizer
tokenizer_json = tokenizer.to_json()
with open('news_classifier_tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(tokenizer_json)

# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(x_val, y_val))

# Save the model
model.save("news_classifier.h5")

# Load the model
model = tf.keras.models.load_model("news_classifier.h5")

# Load the tokenizer
with open('news_classifier_tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_json = f.read()
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

# Get input from the user
article = input("Enter a news article:\n")


# Convert the input to a sequence of word indices
article_seq = tokenizer.texts_to_sequences([article])

# Pad the sequence to a fixed length
article_padded = pad_sequences(article_seq, maxlen=MAX_SEQUENCE_LENGTH)

# Use the model to predict the fakeness index score
prediction = model.predict(article_padded)[0][0]

# Print the fakeness index score
print("Fakeness index score:", prediction)
