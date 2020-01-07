from __future__ import print_function

import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import LSTM

from keras.preprocessing.text import Tokenizer
import random
import csv
import keras_metrics as km

# set parameters:
num_words = 20000 # only use top N words -- nem beleértve a spec tokeneket!
maxlen = 400 # !!!ezt alább felülírom!!!
batch_size = 512
embedding_dims = 50
conv1d_filters = 250
conv1d_kernel_size = 3
hidden_dims = 250
epochs = 2

data_size = 1000000 # :)

train_test_split = 0.8

# mennyit akarunk a végén látni
data_head = 10
text_head = 50

# spec jelek kezelése
# "<PAD>" ----- ennek mindenképp fenntartja a Tokenizer a 0-t => 0. legyen!
# "<UNK>" ----- oov_token param beteszi, hagyomány szerint a 2-es => 2. legyen!
# "<START>" --- hagyomány szerint 1-es
# "<UNUSED>" -- hagyomány szerint 3-as
# szóval:
# "<PAD>" alapból, "<UNK>" az oov_token param miatt bennevan!
# => (spec tokenek száma - 2) db
#    új spec tokennel kell a word_index-et kiegészíteni! :)
spec_tokens = [ "<PAD>", "<START>", "<UNK>", "<UNUSED>" ]
spec_tokens_to_add = len( spec_tokens ) - 2
num_words += len( spec_tokens )

print(' * Loading data...')

# -----

#reader = csv.reader( open('petofi-arany-data/toy-data'), delimiter='\t' )
#reader = csv.reader( open('petofi-arany-data/data'), delimiter='\t' )
#reader = csv.reader( open('petofi-arany-data/utonevek.data'), delimiter='\t' )
reader = csv.reader( open('petofi-arany-data/enhu.data'), delimiter='\t' )

# https://stackoverflow.com/questions/37138491
data = list( reader )

print(' * Data size:', len(data))
print(' * Truncate data to', data_size)
data = data[:data_size]

data = [ [ x[0], int(x[1]) ] for x in data ]

print( data[:data_head] )

random.seed( 76 ) # így van a tesztben ilyen is, olyan is :)
random.shuffle( data )

print(' * Shuffled data:')
print( data[:data_head] )

train_test_split_index = round(len(data) * train_test_split)
train = data[:train_test_split_index]
test = data[train_test_split_index:]

print(' * Train:')
print( train[:data_head] )
print(' * Test:')
print( test[:data_head] )

# https://stackoverflow.com/questions/25806614
texts = [ t[0] for t in data ]

# nagy nehezen kiderült: van a dologra szabványos keras izé! :)
# tök nem trivi a doksiból... ezek mind kellettek hozzá:
# https://shirinsplayground.netlify.com/2019/01/text_classification_keras_data_prep
# https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/text.py
# plusz még az imdb.load_data forrását is nézegettem
# https://stackoverflow.com/questions/51956000
# ... vajon miért hívja ezt a cuccot "Tokenizer"-nek? :)
tokenizer = Tokenizer( num_words=num_words, char_level=True, oov_token="<UNK>" )
tokenizer.fit_on_texts( texts )

# hozzátesszük a spec tokeneket
tokenizer.word_index = {k:(v + spec_tokens_to_add) for k,v in tokenizer.word_index.items()}
for i, spec_token in enumerate( spec_tokens ):
  tokenizer.word_index[spec_token] = i

# https://stackoverflow.com/questions/28704526
print( list(tokenizer.word_index.items())[:data_head] )

sequences = tokenizer.texts_to_sequences( texts )

maxlen = max( map( len, sequences ) ) # ! :)

print()
print(' * Seqs:')
print( sequences[:data_head] )

x_train = sequences[:train_test_split_index]
y_train = [ t[1] for t in train ]
x_test  = sequences[train_test_split_index:]
y_test  = [ t[1] for t in test ]

print(' * x_train:')
print( x_train[:data_head] )
print(' * y_train:')
print( y_train[:data_head] )
print(' * x_test:')
print( x_test[:data_head] )
print(' * y_test:')
print( y_test[:data_head] )

# -----

print()
print(len(data), 'sequences')
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print(' * Pad sequences (samples x "time") maxlen =', maxlen)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('x_train:', x_train[:data_head])
print('x_test:', x_test[:data_head])

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
# word group filters of size filter_length: f1=71.5%
model.add(Conv1D(conv1d_filters,
                 conv1d_kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# XXX mi van, ha Conv1D helyett LSTM-et nyomok? :) f1=71% (2 x Dense esetén)
#model.add(LSTM(256,
#               dropout=0.3,
#               recurrent_dropout=0.3))

# We add a vanilla hidden layer: f1=70.6%
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
# XXX benyomok még egy réteget, akkor vajh mi lesz? :) f1=71.5%
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

print(' * Compile...')
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', km.precision(), km.recall(), km.f1_score()])
# https://stackoverflow.com/questions/43076609
# itt kell beírni, hogy milyen mértékeket szeretnék
# elvileg nem kell km-et használni, de másképp nekem nem megy...

print( model.summary() )

print(' * Fit...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

# Evaluate your performance on test
print(' * Evaluate performance on test:')
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=batch_size)
for l, m in zip( model.metrics_names, loss_and_metrics ):
  print( l, m )

# get original text
# https://stackoverflow.com/questions/42821330
id_to_word = {value:key for key,value in tokenizer.word_index.items()}

print()
print(' * New data:')
for data_item in x_test[:data_head]:
  print('>>>')
  text = [id_to_word[id] + '-' + str(id) for id in data_item]
  print(' '.join(text[:text_head] + ['...']))

print(' * Gold:')
print( y_test[:data_head] )
print(' * Predictions on new data:')
classes = model.predict(x_test, batch_size=batch_size)
print( classes[:data_head] )

predicted = list( int(x[0]) for x in classes.round() )

misclassified = 0
for i, (y, p ) in enumerate( zip( y_test, predicted ) ):
  if y != p:
    text = [id_to_word[id] for id in x_test[i] if id != 0] # XXX ~ copy-paste!
    print( i, y, p, ''.join(text) )
    misclassified += 1

print( " * Misclassified:", misclassified )

#print( y_test )
#print( list( int(x[0]) for x in classes.round() ) )

