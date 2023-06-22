import pickle
import glob, re
import subprocess
import os
import json
import pprint as pp
import re
import nltk
from nltk.tokenize import word_tokenize
import string
from string import punctuation
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import csv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import altair as alt
import pandas as pd
import os
import numpy as np
import pandas as pd


from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = set(stopwords.words('french'))
to_add = ["douleur",'douleurs','douloureux','douloureuse', 'chez','la','le','là',"tous",'dont','ni','peu','peut','très','elles','il','ils','elle','cela','ça','où',"cette","si","moins","cet","enfin","quand","mesme","comme","encore","tant","quelque", "tout", "sous", 'luy',"estre", "sans","plus","faire","fait","a","toute","moy","vostre", "leurs",'.', ',', ';', ':', '?', '!', '(', ')', '[', ']', '{', '}', '-', '_', '+', '=', '"', "'", '/', '\\', '@', '#', '$', '%', '^', '&', '*', '`', '~', '<', '>', '|', '，', '。', '；', '：', '？', '！', '（', '）', '【', '】', '『', '』', '—', '–', '－', '_', '+', '=' ,'“', '”', '‘', '’', '·', '…', '‖', '”', '’', '〝', '〞', '〈', '〉', '《', '》', '「', '」', '『', '』', '【', '】', '〔', '〕', '｛', '｝', '〖', '〗', '〘', '〙', '〚', '〛', '⸨', '⸩']
for w in to_add:
	stop_words.add(w)

'''
CLEANR = re.compile('<.*?>')
with open('termes.json') as outfile:
    TERMES = json.load(outfile)
'''

def cleanhtml(raw_html):
  cleantext = raw_html.replace(")",' ')
  cleantext = cleantext.replace("(",' ')
  cleantext = cleantext.replace(",",' ')
  cleantext = cleantext.replace(".",' ')
  cleantext = cleantext.replace("\\n",' ')
  cleantext = cleantext.replace("\\",' ')
  cleantext = cleantext.replace("\xa0",' ')
  cleantext = cleantext.replace("xa0",' ')
  cleantext = cleantext.replace("/",' ')

  return cleantext

def removeStopWords(tokens):
	filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
	return filtered_tokens



all_sentences = []



csv_file = open('douleur_douloureux.csv', 'r')
reader = csv.reader(csv_file)


i = 0
for row in reader:
	if '-' in row[5] : continue
	print(row[5])
	#print(row[3] + ' ' + row[2]+ ' '+ row[5])
	phrase = str(row[11]) + ' '+ str(row[12]) + ' ' + str(row[13])
	phrase = cleanhtml(phrase)
	tokens = nltk.word_tokenize(phrase, language='french')
	""" TRAITEMENT DES TERMES POLYLEXICAUX
	for terme in TERMES:
			if terme in passage and len(terme.split(" ")) > 1:
				condensed_terme = "_".join(terme.split(' '))
				new_passage = passage.replace(terme,condensed_terme).lower()
	"""
	filtered_tokens = " ".join(removeStopWords(tokens))
	all_sentences.append(filtered_tokens)

	#if i == 1000: break
	i+=1


print(i)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(all_sentences)
number_of_clusters = 5

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
model = KMeans(n_clusters=number_of_clusters, 
               init='k-means++', 
               max_iter=100, # Maximum number of iterations of the k-means algorithm for a single run.
               n_init=1)  # Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.

model.fit(X)
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

for i in range(number_of_clusters):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])



with open('annotations.pkl', 'rb') as f:
    # Load the pickled list from the file
    loaded_annotations = pickle.load(f)

annotations_text = []
new_load = []

for ann in loaded_annotations:
	for an in ann:
		annotations_text.append(an[0])
		new_load.append(an)


X = vectorizer.transform(annotations_text)

for i in range(100):
	cluster = model.predict(X)[i]
	print(str(new_load[i][1]) +' '+ str(cluster))