"""
Creating training sequences from our proprocessed data.
And then encoding each character
"""

import Predictions.Preprocessing as pp

data_text = pp.getPreprocessedData() #ambiguous nomenclature
#print(data_text)

def createSequence(text):
    length = 30
    sequences = list()
    for i in range(length, len(text)):
        #select sequence of tokens
        seq = text[i-length:i+1]
        #store that
        sequences.append(seq)
    print('Total sequences: %d' % len(sequences))
    return sequences

mySequences = createSequence(data_text)
print(mySequences)

# Character mapping index
chars = sorted(list(set(data_text)))
mapping = dict((c, i) for i, c in enumerate(chars))

def encodeSequence(seq):
    sequences = list()
    for line in seq:
        # int encode line
        encodedSeq = [mapping[char] for char in line]
        # store
        sequences.append(encodedSeq)
    return sequences

def getEncodedSequencesAndMapping():
    retVal = []
    myEncodedSeqs = encodeSequence(mySequences)
    retVal.append(myEncodedSeqs)
    retVal.append(mapping)
    return retVal


"""
And now you gotta split the data into training and validation sets. This is so that while training, you can keep 
a track of how good the language model is working with unseen data.
"""