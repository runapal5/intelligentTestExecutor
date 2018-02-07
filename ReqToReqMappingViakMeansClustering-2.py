import collections
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from utils import build_feature_matrix
import csv
from normalization import normalize_corpus


#print "Step 1::::********************************************Parsing CSV file and converting into array*********************"
results = []
with open("Req_BM2.csv") as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    for row in reader: # each row is a list
        results.append(row[1])

# print results

sentences = results

# normalize corpus
norm_req_synopses = normalize_corpus(sentences,
                                       lemmatize=True,
                                       only_text_chars=False)


# extract tf-idf features
vectorizer, feature_matrix = build_feature_matrix(norm_req_synopses,
                                                  feature_type='tfidf',
                                                  min_df=0.1, max_df=0.9,
                                                  ngram_range=(1, 2))



# view number of features
#print feature_matrix.shape     

# get feature names
feature_names = vectorizer.get_feature_names()

# print sample features
#print feature_names[:20]      

topn_features = 10
cluster_details = {}


def word_tokenizer(text):
		#tokenizes and stems the text
		tokens = word_tokenize(text)
		stemmer = PorterStemmer()
		tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
		return tokens


def cluster_sentences(sentences, nb_of_clusters=5):
                #cluster_details = {} 
		tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,
			                           stop_words=stopwords.words('english'),
						   max_df=0.9,
						   min_df=0.1,
						   lowercase=True)
                
		#builds a tf-idf matrix for the sentences
		tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
		kmeans = KMeans(n_clusters=nb_of_clusters)
		#kmeans.fit(tfidf_matrix)
                kmeans.fit(feature_matrix)
                ordered_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
                #print ordered_centroids               
		clusters = collections.defaultdict(list)
                for cluster_num  in range(nb_of_clusters):
                       cluster_details[cluster_num] = {}
                       cluster_details[cluster_num]['cluster_num'] = cluster_num
                       key_features = [feature_names[index] 
                                          for index 
                                          in ordered_centroids[cluster_num, :topn_features]]
                       cluster_details[cluster_num]['key_features'] = key_features
                       print "For Cluster" ,cluster_num, "  the  Key Features are:" , key_features
                 

		for i, label in enumerate(kmeans.labels_):
				clusters[label].append(i)                                 
		return dict(clusters)


if __name__ == "__main__":
		nclusters= 2
		clusters = cluster_sentences(sentences, nclusters)
		for cluster in range(nclusters):
				print "cluster ",cluster,":"
				for i,sentence in enumerate(clusters[cluster]):
						print "\tsentence ",i,": ",sentences[sentence]



