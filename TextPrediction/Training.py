import Predictions.TrainingSequences as ts

from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint


retVal = ts.getEncodedSequences()
sequences = retVal[0]
mapping = retVal[1]
#print(sequences)

#vocab size
vocab = len(mapping)
sequences = np.array(sequences)
# Creating x, y

x, y = sequences[:, :-1], sequences[:, -1]
#one hot encode (bleh) y
y = to_categorical(y, num_classes=vocab)

#create and train validationsets
x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=0.1, random_state=42) # What??

print('Train shape: ', x_tr.shape, 'Val shape: ', x_val.shape)

"""
Embedding layer of Keras to lean a 50 dimentsion embedding for each character.
GRU layer as the base model which has 150 timesteps
Dense layer with a softmax activation for prediction
"""
model = Sequential()
model.add(Embedding(vocab, 50, input_length=30, trainable=True))
model.add(GRU(159, recurrent_dropout=0.1, dropout=0.1))
model.add(Dense(vocab, activation='softmax'))
print(model.summary())

# compile model
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
# fit the model
model.fit(x_tr, y_tr, epochs=100, verbose=2, validation_data=(x_val, y_val))

def generateSequence(model, mapping, seqLen, seedText, noOfChars):
    inputText = seedText
    #generate fixed no of chars
    for _ in range(noOfChars):
        #encoode characters as int
        encoded = [mapping[char] for chat in inputText]

        #truncate sequencs to a fixed length
        encoded = pad_sequences([encoded], maxlen=seqLen, truncating='pre')

        # predict character
        pchar = model.predict_classes(encoded, verbose=0)

        #reverse map the int to the char
        outputChar = ''
        for char, index in mapping.items():
            if index == pchar:
                outputChar = char
                break

        # append to input to look fancy
        inputText += char

    return inputText

testInput = 'Gender Stereotypes are '
testOutput = generateSequence(model, mapping, 30, testInput.lower(), 15)
print('Ip : ', testInput)
print('Op : ', testOutput)