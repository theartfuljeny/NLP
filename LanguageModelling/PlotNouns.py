"""
Previous attempt didn't plot correctly so this is ze zecond attempt. <3

Okay, so it worked, but it picked up every single word in the vocabulary including the prepositions, aticles, etc.
So you gotta find a way for it to not do any of that.

Continuation : Attempting to plot only the noun phrases in the
"""

from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings(action='ignore')

import gensim
from gensim.models import word2vec
import matplotlib.pyplot as plt
from textblob import TextBlob
from textblob.taggers import NLTKTagger

#Read ze file
sample = open("C://JENY//University//Master Project//EnglishPaper.txt", "r", encoding="utf8")
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
model = gensim.models.Word2Vec(data, min_count = 1, size = 100, window = 5)

X = model[model.wv.vocab]
words = list(model.wv.vocab)

#*******************************#

# Naive noun plotting
nltkTagger = NLTKTagger()
blob = TextBlob(s, pos_tagger=nltkTagger)
allTags = blob.pos_tags

nouns = []

# remove non nouns
for n in allTags:
    if str(n[1]) == 'NN':
        nouns.append(n)
# print('NOUN LIST: ', nouns)

# Remove duplicates
nounFrequencies = {}
for n in nouns:
    if n[0] in nounFrequencies:
        val = nounFrequencies.get(n[0])
        val = val + 1
        nounFrequencies.update({n[0] : val})
    else:
        nounFrequencies.update({n[0] : 1})

# print(nounFrequencies)
# Sort by frequency
sortedNounFrequencies = {k: v for k, v in sorted(nounFrequencies.items(), key = lambda item: item[1], reverse=True)}
# print(sortedNounFrequencies)
keysToIgnore = []
for k in sortedNounFrequencies.keys():
    val = sortedNounFrequencies.get(k)
    if val < 3:
        keysToIgnore.append(k)
    if len(k) < 3:
        keysToIgnore.append(k)

# Can't directly delete from dictionary in a loop, that's why all the drama
for elemt in keysToIgnore:
    if elemt in sortedNounFrequencies.keys():
        sortedNounFrequencies.pop(elemt)

# print(sortedNounFrequencies)
myNouns = list(sortedNounFrequencies.keys())
# refine also the words list, like remove words and shit
print('My Nouns:', myNouns)

for elemnt in words:
    if elemnt not in myNouns:
        #print(elemnt)
        words.remove(elemnt)


testX = model[myNouns]
testwords = myNouns

#*******************************#

pca = PCA(n_components=2)
result = pca.fit_transform(testX)

plt.scatter(result[:,0], result[:, 1])
for i, word in enumerate(testwords):
    plt.annotate(word, xy = (result[i, 0], result[i, 1]))
plt.show()


def getW2VModelOfText():
    return model