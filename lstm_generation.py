import numpy as np
import sys

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import np_utils

# from tensorflow.python.keras import backend as k

def char_to_int(text_list):
    return dict((c, i) for i, c in enumerate(text_list))

def int_to_char(text_list):
    return dict((i, c) for i, c in enumerate(text_list))

alice_file = "alice.txt"
dracula_file = "dracula.txt"
jekyll_hyde_file = "jekyll_and_hyde.txt"

raw_text_alice = open(alice_file).read()
raw_text_dracula = open(dracula_file).read()
raw_text_jekyll = open(jekyll_hyde_file).read()

raw_text_alice = raw_text_alice.lower()
raw_text_dracula = raw_text_dracula.lower()
raw_text_jekyll = raw_text_jekyll.lower()

# Can be refactored into a function
alice = sorted(list(set(raw_text_alice)))
alice_char_to_int = char_to_int(alice)
dracula = sorted(list(set(raw_text_dracula)))
dracula_char_to_int = char_to_int(dracula)
jekyll_hyde = sorted(list(raw_text_jekyll))
jekyll_hyde_char_to_int = char_to_int(jekyll_hyde)

# Can be refactored into a function
alice_chars = len(raw_text_alice)
alice_vocab = len(alice)
dracula_chars = len(raw_text_dracula)
dracula_vocab = len(dracula)
jekyll_hyde_chars = len(raw_text_jekyll)
jekyll_hyde_vocab = len(jekyll_hyde)

# Can be refactored into a function
sequence_length = 10
aliceX = []
aliceY = []
for i in range(0, alice_chars - sequence_length, 1):
    sequence_in = raw_text_alice[i: i+sequence_length]
    sequence_out = raw_text_alice[i + sequence_length]
    aliceX.append([alice_char_to_int[char] for char in sequence_in])
    aliceY.append(alice_char_to_int[sequence_out])

number_of_patterns = len(aliceX)

wonderlandX = np.reshape(aliceX, (number_of_patterns, sequence_length, 1))
wonderlandX = wonderlandX / float(alice_vocab)
wonderlandy = np_utils.to_categorical(aliceY)

model = Sequential()
model.add(LSTM(256, input_shape = (wonderlandX.shape[1], wonderlandX.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(wonderlandy.shape[1], activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# model = Sequential()
# model.add(LSTM(256, input_shape = (wonderlandX.shape[1], wonderlandX.shape[2]), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(256))
# model.add(Dropout(0.2))
# model.add(Dense(wonderlandy.shape[1], activation = 'softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam')

# alice_model_file = "biggs/weights-improvement- - .hdf5" #fix with actual weight file
# alice_model_load = model.load_weights(alice_model_file)

alice_model_file = "wedge/weights-improvement-50-1.4290.hdf5"
alice_model_load = model.load_weights(alice_model_file)

alice_compile = model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
alice_int_to_char = int_to_char(alice)

start = np.random.randint(0, len(aliceX)-1)
alice_pattern = aliceX[start]
print("Initialize with a seed value: ")
print("\"", ''.join([alice_int_to_char[value] for value in alice_pattern]), "\"")

for i in range(1000):
    alice_x = np.reshape(alice_pattern, (1, len(alice_pattern), 1))
    alice_x = alice_x / float(alice_vocab)
    alice_prediction = model.predict(alice_x, verbose=0)
    alice_index = np.argmax(alice_prediction)
    alice_result = alice_int_to_char[alice_index]
    alice_sequence_in = [alice_int_to_char[value] for value in alice_pattern]
    sys.stdout.write(alice_result)
    alice_pattern.append(alice_index)
    alice_pattern = alice_pattern[1:len(alice_pattern)]

print("\n So it has been written.")
