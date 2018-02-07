import collections
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint


import csv


#print "Step 1::::********************************************Parsing CSV file and converting into array*********************"
results = []
with open("Req_BM2.csv") as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    for row in reader: # each row is a list
        results.append(row[1])

# print results

sentences = results



def word_tokenizer(text):
		#tokenizes and stems the text
		tokens = word_tokenize(text)
		stemmer = PorterStemmer()
		tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
		return tokens


def cluster_sentences(sentences, nb_of_clusters=5):
		tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,
			                           stop_words=stopwords.words('english'),
						   max_df=0.9,
						   min_df=0.1,
						   lowercase=True)
		#builds a tf-idf matrix for the sentences
		tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
		kmeans = KMeans(n_clusters=nb_of_clusters)
		kmeans.fit(tfidf_matrix)
		clusters = collections.defaultdict(list)
		for i, label in enumerate(kmeans.labels_):
				clusters[label].append(i)
		return dict(clusters)


if __name__ == "__main__":
		nclusters= 43
		clusters = cluster_sentences(sentences, nclusters)
		for cluster in range(nclusters):
				print "cluster ",cluster,":"
				for i,sentence in enumerate(clusters[cluster]):
						print "\tsentence ",i,": ",sentences[sentence]

