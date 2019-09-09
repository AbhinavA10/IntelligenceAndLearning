### An extractive summarizer, using the TextRank Method


### Main idea of spacy summarizer:
### - do text preprocessing (remove stopwords, punctuation)
### - create frequency table of words
### - score each depending on words contained and the frequency table
### - build summary by joining sentences that are above a score limit

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

f = open("Ghandhi.txt","r")
document1 = f.read()

# Text processing: 
stopwords = list(STOP_WORDS)
#print (stopwords)
# print (punctuation)

nlp = spacy.load('en')
docx = nlp(document1) # tokenizes document

# Word freq table
# - making dictonary of words and their counts, using non-stop words
word_frequencies = {}#empty dict
for word in docx:
    if word.text not in stopwords:
        if word.text not in word_frequencies.keys():
            word_frequencies[word.text] = 1 # add to dict if not there yet
        else:
            word_frequencies[word.text]+=1

# finding the weighted freq
max_freq = max(word_frequencies.values())
#print (max_freq)
for word in word_frequencies.keys():
    word_frequencies[word] = (word_frequencies[word]/max_freq)
#print (word_frequencies)

#finding the score of the sentence
sentence_list=[sentence for sentence in docx.sents]
sentence_scores={}
for sentence in sentence_list:
    for word in sentence:
        if word.text.lower() in word_frequencies.keys():
            if len(sentence.text.split(' '))<30: #arbitrary value for max length of a sentence. Bigger sentences will have a bigger score
                if sentence not in sentence_scores.keys():
                    sentence_scores[sentence]=word_frequencies[word .text.lower()]
                else: 
                    sentence_scores[sentence]+=1
#print (sentence_scores)

#print("----------")
#finding top sentences with the largest score
from heapq import nlargest
summarized_sentences=nlargest(5,sentence_scores,key=sentence_scores.get)
#print (summarized_sentences)

#convert back to text
final_sentences  = [w.text for w in summarized_sentences]
summary = ' '.join(final_sentences)
print (summary)