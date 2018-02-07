import nltk
import string


text="CR 4901 - HHT Reflight Bags > User interface needs to be simplified for less experienced users"
print "Raw Text:=======================================>",text
# ********************************* TEXT PRE-PROCESSING ****************************************************

from nltk.tokenize import sent_tokenize
print(sent_tokenize(text))


print "************************************************************Phase 1: Noisy Entities Removal :- Remove Punctuations , Remove Stop Words , Remove URLs , Remove Mentions"
# Function to remove punctuations
def remove_punctuations(text):
    words = nltk.word_tokenize(text)
    punt_removed = [w for w in words if w.lower() not in string.punctuation]
    return " ".join(punt_removed)

#print remove_punctuations(text)
cleaned_text = remove_punctuations(text)
print "Step 1: Cleaned Text(Remove Punctuation) =====================>", cleaned_text

# Function to remove stop words 
from nltk.corpus import stopwords
def remove_stopwords(text,lang='english'):
    words = nltk.word_tokenize(text) 
    lang_stopwords = stopwords.words(lang)
    stopwords_removed = [w for w in words if w.lower() not in lang_stopwords]
    return " ".join(stopwords_removed)

#print remove_stopwords(text)    
cleaned_text = remove_stopwords(cleaned_text)
print "Step 2: Cleaned Text(Remove Stop Words) =====================>", cleaned_text


print "***************************************************************Phase 2:Word Normalization :- Tokenization , Lemmanization, Stemming"

# Function to apply lemmatization to a list of words
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
words = nltk.word_tokenize(cleaned_text)

def words_lemmatizer(text, encoding="utf8"):
#    words = nltk.word_tokenize(text)
    lemma_words = []
    wl = WordNetLemmatizer()
    for word in words:
        pos = find_pos(word)
        lemma_words.append(wl.lemmatize(word, pos).encode(encoding))
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

cleaned_text = words_lemmatizer(words)
print "Step 3: Cleaned Text(Lemmatization) =====================>", cleaned_text



# Function to apply stemming to a list of words
from nltk import PorterStemmer, LancasterStemmer, SnowballStemmer
def words_stemmer(words,type="PorterStemmer", lang="english" , encoding="utf8"):
    supported_stemmers = ["PorterStemmer" , "LancasterStemmer" , "SnowballStemmer"]
    if type is False or type not in supported_stemmers:
       return words
    else:
       stem_words = []
       if type == "PorterStemmer":
          stemmer = PorterStemmer()
          for word in words:
            stem_words.append(stemmer.stem(word).encode(encoding))

       if type == "LancasterStemmer":
          stemmer = LancasterStemmer()
          for word in words:
            stem_words.append(stemmer.stem(word).encode(encoding))


       if type == "SnowballStemmer":
          stemmer = SnowballStemmer(lang)
          for word in words:
            stem_words.append(stemmer.stem(word).encode(encoding))
       return " ".join(stem_words)


#print "Original:" , text
#print "Porter:" , words_stemmer(nltk.word_tokenize(text),"PorterStemmer")
#print "Lancastar:" , words_stemmer(nltk.word_tokenize(text),"LancasterStemmer")
#print "Snowball:" , words_stemmer(nltk.word_tokenize(text),"SnowballStemmer")
cleaned_text = words_stemmer(nltk.word_tokenize(cleaned_text),"PorterStemmer")
print "Step 4: Cleaned Text(Stemming) =====================>", cleaned_text



# ********************************* TEXT PRE-PROCESSING ****************************************************

# ********************************* CONVERTING TEXT TO FEATURES *********************************************










# ********************************* CONVERTING TEXT TO FEATURES *********************************************
