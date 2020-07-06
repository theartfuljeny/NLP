"""
Reading a file to do some preprocesing on it. Like removing punctuations and prepositions that are really small.
Which is naive because prepositions are important too but we're just making a start here.

There's a method that returns your preprocessed data.
"""

import re
data = open("C://JENY//University//Master Project//EnglishPaper.txt", "r", encoding="utf8")
data_text = data.read()

#print(data_text)

def text_cleaner(text):
    #lower case text
    newString = text.lower()
    newString = re.sub(r"'s\b", "", newString)

    #remove punctuations
    newString = re.sub("[^a-zA-Z]", " ", newString)
    longWords = []
    # remove short words (less than 3)
    for i in newString.split():
        if len(i) >= 3:
            longWords.append(i)
    return (" ".join(longWords)).strip()

def getPreprocessedData():
    data_new = text_cleaner(data_text)
    #print(data_new)
    return data_new

#getPreprocessedData()