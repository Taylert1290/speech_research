import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from keras.utils import to_categorical
from keras.utils import pad_sequences
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dropout


class DataSetup(object):
    def __init__(self, max_length, adding_padding=True, post_padding=True):
        self.max_length = max_length
        self.add_padding = adding_padding
        self.post_padding = post_padding

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"s\b", " ", text)
        text = re.sub(r"[^a-zA-Z]", " ", text)
        text = [t for t in text.split() if len(t) > 3 and t != " "]
        return " ".join(text)

    def character_map(self, corpus):
        clean_corpus = [self.preprocess(doc) for doc in corpus]
        characters = sorted(list(set(" ".join(clean_corpus))))
        char_to_id = dict((c, i) for i, c in enumerate(characters) if len(c) > 0)
        id_to_char = dict((char_to_id[c], c) for c in char_to_id)
        return char_to_id, id_to_char

    def build_data(self, corpus):
        X, y = [], []
        char_mapping = self.character_map(corpus=corpus)[0]
        for doc in tqdm(corpus):
            clean_text = self.preprocess(doc)
            for i in range(0, len(clean_text) - self.max_length, 1):
                char_sequence_x = clean_text[i : i + self.max_length]
                char_sequence_y = clean_text[i + self.max_length]
                X.append([char_mapping[char] for char in char_sequence_x])
                y.append(char_mapping[char_sequence_y])
        n_patterns = len(X)
        X = np.reshape(X, (n_patterns, self.max_length, 1))
        # one hot encode the output variable
        y = to_categorical(y)
        return X, y


class TextGeneration(object):
    def __init__(self, max_len, vocab_size, embedding_size):
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

    def model(self):
        model = Sequential()
        model.add(LSTM(self.embedding_size, input_shape=(self.max_len, 1)))
        model.add(Dropout(0.2))
        model.add(Dense(self.vocab_size, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam")
        return model

    def fit(self, X, y):
        model = self.model()
        filepath = "../saved_models/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(
            filepath, monitor="loss", verbose=1, save_best_only=True, mode="min"
        )
        callbacks_list = [checkpoint]
        model.fit(X, y, epochs=10, batch_size=128, callbacks=callbacks_list)

    def predict(self):
        pass


if __name__ == "__main__":
    df = pd.read_csv("../text_data/Corona_NLP_test.csv")
    text = df["OriginalTweet"]
    ds = DataSetup(max_length=100)
    X, y = ds.build_data(text)
    print(len(X))
    char_to_id, id_to_char = ds.character_map(corpus=text)
    model = TextGeneration(max_len=100, vocab_size=27, embedding_size=256)
    model.fit(X, y)
    # start = np.random.randint(0, len(X) - 1)
    # pattern = X[start]
    # append_pattern = [value[0] for value in pattern]
    # print("Seed:")
    # print("\"", ''.join([id_to_char[value[0]] for value in pattern]), "\"")
    # model = keras.models.load_model('weights-improvement-03-2.0079.hdf5')
    # for i in range(10):
    #     x = np.reshape(pattern, (1, len(pattern), 1))
    #     prediction = model.predict(x, verbose=0)
    #     index = np.argmax(np.array(prediction))
    #     result = id_to_char[index]
    #     seq_in = [id_to_char[value[0]] for value in pattern]
    #     append_pattern.append(index)
    #     pattern = pattern[1:len(pattern)]
    # print("\"", ''.join([id_to_char[value] for value in append_pattern]), "\"")
