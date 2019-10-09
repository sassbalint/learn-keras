'''
#This example demonstrates the use of Convolution1D for text classification.

Gets to 0.89 test accuracy after 2 epochs. </br>
90s/epoch on Intel i5 2.4Ghz CPU. </br>
10s/epoch on Tesla K40 GPU.
'''
from __future__ import print_function

import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb

# set parameters:
num_words = 10000 # only use top N words
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2

data_size = 5000 # max = 25000

# mennyit akarunk a végén látni
data_head = 3
text_head = 50

index_from = 3 # a spec jelek miatt kell asszem...

print(' * Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words, index_from=index_from)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print(' * Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print(' * Truncate data...')
x_train = x_train[:data_size]
y_train = y_train[:data_size]
x_test = x_test[:data_size]
y_test = y_test[:data_size]
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print()
print(' * Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(num_words,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

print(' * Compile...')
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(' * Fit...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

# Evaluate your performance in one line:
print(' * Evaluate your performance in one line:')
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=batch_size)
print( loss_and_metrics )

# get original text from imdb dataset
# https://stackoverflow.com/questions/42821330
word_to_id = imdb.get_word_index()
# valójában miért kell ez a hekkelés? és miért működik? XXX
word_to_id = {k:(v + index_from) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
word_to_id["<UNUSED>"] = 3
id_to_word = {value:key for key,value in word_to_id.items()}

print()
print(' * New data:')
for data_item in x_test[:data_head]:
  print('>>>')
  text = [id_to_word[id] + '-' + str(id) for id in data_item if id != 0]
  print(' '.join(text[:text_head] + ['...']))

print(' * Gold:')
print( y_test[:data_head] )
print(' * Predictions on new data:')
classes = model.predict(x_test, batch_size=batch_size)
print( classes[:data_head] )

