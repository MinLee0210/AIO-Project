import pandas as pd
import re
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATA_PATH = './data/poem_dataset.csv'
df = pd.read_csv(DATA_PATH, index_col=0)
df.head(10)

def text_normalize(text):
    text = text.lower()
    text = text.strip()
    text = re.sub(r'[^\w\s\n]', '', text)
    text = text.replace('\n\n', '\n')
    text = '\n'.join(['<start>' + line + '<end>' for line in text.split('\n') 
                      if line!='' and len(line.split())==5])
    return text

df['content'] = df['content'].apply(lambda p: text_normalize(p))
corpus = df['content'].to_numpy()


X = []
y = []

for idx, row in df.iterrows():
    lines = row['content'].split('\n')
    lines = [line for line in lines if line!=' ']
    for idx in range(0, len(lines) -1):
        input_sentence = lines[idx]
        output_sentence = lines[idx+1]

        X.append(input_sentence)
        y.append(output_sentence)


# Tokenization
VOCAB_SIZE = 50000

tokenizer = Tokenizer(num_words=VOCAB_SIZE, filters=' ', oov_token='<OOV>')
tokenizer.fit_on_texts(corpus)

VOCAB_SIZE = len(tokenizer.word_index) + 1


# Padding
MAX_SEQ_LEN = 7
X_sequences = tokenizer.texts_to_sequences(X)
X_padded_sequences = pad_sequences(X_sequences, maxlen=MAX_SEQ_LEN, 
                                   truncating='pre', padding='post')


def prepare_output_sequence(y_sequence):
    y_inputs = pad_sequences([y_seq[:-1] for y_seq in y_sequence], maxlen=MAX_SEQ_LEN, 
                             truncating='pre', padding='post')
    y_outputs = pad_sequences([y_seq[1:] for y_seq in y_sequence], maxlen=MAX_SEQ_LEN,
                              truncating='pre', padding='post')
    return y_inputs, y_outputs

y_sequences = tokenizer.texts_to_sequences(y)
y_inputs, y_outputs = prepare_output_sequence(y_sequences)


n_samples = len(X_padded_sequences)
train_len = int(n_samples*0.7)
val_len = int(n_samples*0.2)
test_len = n_samples - train_len - val_len

# Shuffle

np.random.seed(1)
idxs = np.arange(n_samples)
idxs = np.random.permutation(idxs)

X_padded_sequences = X_padded_sequences[idxs]
y_inputs = y_inputs[idxs]
y_outputs = y_outputs[idxs]

# Splitting dataset into train/val/test
X_train_seq, y_train_input, y_train_output = X_padded_sequences[:train_len], y_inputs[:train_len], y_outputs[:train_len]
X_val_seq, y_val_input, y_val_output = X_padded_sequences[train_len:train_len+val_len], y_inputs[train_len:train_len+val_len], y_outputs[train_len:train_len+val_len]
X_test_seq, y_test_input, y_test_output = X_padded_sequences[train_len+val_len:], y_inputs[train_len+val_len:], y_outputs[train_len+val_len:]


BATCH_SIZE = 32

train_ds = tf.data.Dataset.from_tensor_slices(((X_train_seq, y_train_input), y_train_output)).batch(BATCH_SIZE)
val_ds = tf.data.Dataset.from_tensor_slices(((X_val_seq, y_val_input), y_val_output)).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices(((X_test_seq, y_test_input), y_test_output)).batch(BATCH_SIZE)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)