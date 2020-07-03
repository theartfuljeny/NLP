"""
So I'm gonna attempt to generate word vectors using Word2Vec
"""
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings(action='ignore')

import gensim
from gensim.models import word2vec
import numpy as np
import matplotlib.pyplot as plt

#Read ze file
sample = open("C://JENY//University//Master Project//TestText.txt", "r", encoding="utf8")
s = sample.read()

#replacing escape characted with a space - Why? Idk man
f = s.replace("\n", " ")

data = []

# Iterate through each sentence in the file
for i in sent_tokenize(f): # oh dang wtf
    temp = []
    #tokenize  the sentence into words
    for j in word_tokenize(i):
        temp.append((j.lower())) # what does this do?

    data.append(temp)

# Create CBOW model
model1 = gensim.models.Word2Vec(data, min_count = 1, size = 100, window = 5)

# Printing
op1 = model1.similarity('alice', 'wonderland')
op2 = model1.similarity('alice', 'machines') # does it have to be a word in the vocab? Does the training do fine tuning or new training??

print("Calculatin cosine similarity between the words alice and wonderland", op1)
print("Calculatin cosine similarity between the words alice and machines", op2)

# Create SKIPGram Model
model2 = gensim.models.Word2Vec(data, min_count = 1, size = 100, window = 5, sg = 1) # by default sg = 0 which is CBOW

op3 = model2.similarity('alice', 'wonderland')
op4 = model2.similarity('alice', 'machines')

print("Same comparison with Skipgram : ", op3)
print("Same comparison with Skipgram : ", op4)

#prints learned vocabulary of tokens
learnedWords = list(model1.wv.vocab)
print(learnedWords)

#embedded vector for a certain word:
embeddingForWord = model1['alice']
#print(embeddingForWord)

# Can save the model, typically as a binary file
# model1.wv.save_word2vec_format('myModel.bin') # add argument binary=False if you want to save in ASCII
# Can load the model again as :
# savedModel = gensim.models.Word2Vec.load('myModel.bin')



def findSomethingSimilar(model):
    a = model['alice']
    #print(a)

    """
        Okay I actually  made this function to try an understand how this 'model' variable is stored but imma put it of for now
        So please look at it again. Actually, I think you will end up looking at it again.
        
        You could probably find it on the Word2Vec paper publication.
    """
    pass

# Function to plote ze embedding
def displayMyEmbedding(model, word, size):

    arr = np.empty((0, size), dtype = 'f')
    word_labels = [word]

    close_words = model.similar_by_word(word)

    arr = np.append(arr, np.array([model[word]]), axis = 0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis = 0)

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords) # plotting scatterplot using matplotlib, which is our embedding I guess?

    for label, x, y, in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy = (x, y), xytext = (0, 0), textcoords = 'offset points')
        plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
        plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
        plt.show()

#displayMyEmbedding(model1, 'alice', 100)

