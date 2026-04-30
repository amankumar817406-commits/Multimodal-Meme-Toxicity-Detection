import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load dataset
df = pd.read_csv("data/jigsaw-toxic-comment.csv")

# Create binary label
df['label'] = df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].max(axis=1)

texts = df['comment_text'].astype(str)
labels = df['label']

# Tokenization
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=100)

# Model
model = Sequential()
model.add(Embedding(10000, 128))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(X, labels, epochs=3, batch_size=64)

# Prediction function
def predict_text(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=100)
    pred = model.predict(pad)[0][0]
    return pred
