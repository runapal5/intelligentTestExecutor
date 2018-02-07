from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
import nltk
from gensim import corpora

import csv


print "Step 1::::********************************************Parsing CSV file and converting into array*********************"
results = []
with open("Req_BM1.csv") as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    for row in reader: # each row is a list
        results.append(row[1])

# print results

doc_complete = results


print "Step 2::::********************************************Cleaning of Text ***********************************************"

# Function to remove punctuations
def remove_punctuations(text):
    text = text.decode('unicode_escape').encode('utf-8')
    words = nltk.word_tokenize(text)
    punt_removed = [w for w in words if w.lower() not in string.punctuation]
    return " ".join(punt_removed)

#print remove_punctuations(text)
#cleaned_text = remove_punctuations(text)
#print "Step 1: Cleaned Text(Remove Punctuation) =====================>", cleaned_text


# Function to remove stop words 
from nltk.corpus import stopwords
def remove_stopwords(text,lang='english'):
    words = nltk.word_tokenize(text)
    lang_stopwords = stopwords.words(lang)
    stopwords_removed = [w for w in words if w.lower() not in lang_stopwords]
    return " ".join(stopwords_removed)



# Function to apply lemmatization to a list of words
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
#words = nltk.word_tokenize(remove_stopwords(remove_punctuations(doc)))

def words_lemmatizer(words):
#    words = nltk.word_tokenize(text)
    lemma_words = []
    wl = WordNetLemmatizer()
    for word in words:
        pos = find_pos(word)
        lemma_words.append(wl.lemmatize(word, pos))
    return " ".join(lemma_words)


# Function to find part of speech tag for a word
def find_pos(word):
# Part of Speech constants
# ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
# You can learn more about these at http://wordnet.princeton.edu/wordnet/man/wndb.5WN.html#sect3
# You can learn more about all the penn tree tags at https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    pos = nltk.pos_tag(nltk.word_tokenize(word))[0][1]
# Adjective tags - 'JJ', 'JJR', 'JJS'
    if pos.lower()[0] == 'j':
       return 'a'
# Adverb tags - 'RB', 'RBR', 'RBS'
    elif pos.lower()[0] == 'r':
         return 'r'
# Verb tags - 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'
    elif pos.lower()[0] == 'v':
         return 'v'
# Noun tags - 'NN', 'NNS', 'NNP', 'NNPS'
    else:
         return 'n'

#print "Lemmatized: ", words_lemmatizer(words)

#cleaned_text = words_lemmatizer(words)
#print "Step 3: Cleaned Text(Lemmatization) =====================>", cleaned_text


#cleaned_text = words_lemmatizer(nltk.word_tokenize(remove_stopwords(remove_punctuations(doc))))
print "Step 3::::*****************************************Feature Extraction as Entitites ***************************"



# *******************************************************************************************************************************************
print "Method 1 : Function to Extract Entities as Feature via Topic Modelling using LDA" 

doc_clean = [words_lemmatizer(nltk.word_tokenize(remove_stopwords(remove_punctuations(doc)))).split() for doc in doc_complete]

#doc_clean = [remove_stopwords(remove_punctuations(doc)).split() for doc in doc_complete]

## Creating the term dictionary of our corpus, where every unique term is assigned an index.  
dictionary = corpora.Dictionary(doc_clean)

## Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above. 
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

## Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

## Running and Training LDA model on the document term matrix
ldamodel = Lda(doc_term_matrix, num_topics=5,id2word = dictionary, passes=100)

# Results 
print "LDA Topic Modelling:::",ldamodel.print_topics(num_topics=5, num_words=5)
#print "LDA Topic Modelling:::",ldamodel.print_topics()

print "******************************************************************************************************************************"

# ***************************************************************************************************

print "Method 2 : Function to extract Entities as Feature via N-Grams i.e N-Grams as Feature"
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

def get_ngrams(text, n ):
    n_grams = ngrams(text, n)
    return [ ' '.join(grams) for grams in n_grams]

ngrams_features = [get_ngrams(nltk.word_tokenize(remove_stopwords(remove_punctuations(doc))),2) for doc in doc_complete]
print "N-GRAMS for Feature Extraction:::",ngrams_features



# **************************************************************************************



# **********************************************************************************************************************************************

# Method 3 : Function to Extract Entities as Feature via Statistical Feature

#from sklearn.feature_extraction.text import TfidfVectorizer
#obj = TfidfVectorizer()

#X = obj.fit_transform(results)
#print "Statistical Feature :::::::", X

# **********************************************************************************************************************************************
