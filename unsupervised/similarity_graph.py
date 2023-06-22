from bs4 import BeautifulSoup
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
stop_words = set(stopwords.words('french'))

to_add = ['où',"cette","si","moins","cet","enfin","quand","mesme","comme","encore","tant","quelque", "tout", "sous", 'luy',"estre", "sans","plus","faire","fait","a","toute","moy","vostre", "leurs",'.', ',', ';', ':', '?', '!', '(', ')', '[', ']', '{', '}', '-', '_', '+', '=', '"', "'", '/', '\\', '@', '#', '$', '%', '^', '&', '*', '`', '~', '<', '>', '|', '，', '。', '；', '：', '？', '！', '（', '）', '【', '】', '『', '』', '—', '–', '－', '_', '+', '=' ,'“', '”', '‘', '’', '·', '…', '‖', '”', '’', '〝', '〞', '〈', '〉', '《', '》', '「', '」', '『', '』', '【', '】', '〔', '〕', '｛', '｝', '〖', '〗', '〘', '〙', '〚', '〛', '⸨', '⸩']

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
	if int(row[5]) < 1913: continue

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

	tokens = removeStopWords(tokens)
	all_sentences.append(tokens)


	#if i == 1000: break
	i+=1


print(i)
print(all_sentences)



model = Word2Vec(all_sentences, 
                 min_count=3,   # Ignore words that appear less than this # Dimensionality of word embeddings
                 workers=2,     # Number of processors (parallelisation)
                 window=5      # Context window for words during training
                 )       # Number of epochs training over corpus



# Get the top N most similar words to "death"
similar_words = model.wv.most_similar("douleur",topn=15)
# Extract the similar word from the list
similar_words.append(("douleur",1))
similar_words = [word[0] for word in similar_words]

# Get the embeddings of the similar words
X = model.wv[similar_words]

# Perform PCA on the word embeddings
pca = PCA(n_components=3)
result = pca.fit_transform(X)

data = pd.DataFrame(result, columns=['x', 'y', 'z'], index=similar_words)

data["word"] = data.index


chart = alt.Chart(data,width = 800, height = 600).mark_circle().encode(
    x='x',
    y='y',
    color=alt.Color('z', scale=alt.Scale(scheme='greenblue')),
    tooltip=['word'],
    size=alt.Size('z', scale=alt.Scale(range=[10, 500]), legend=None)
).interactive()

chart.save("pca_word_embeddings.html")



"""
# Create a 3D scatter plot of the PCA result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(result[:, 0], result[:, 1], result[:, 2])

for i, word in enumerate(similar_words):
    ax.text(result[i, 0], result[i, 1], result[i, 2], word)

plt.show()

"""

'''
print(dir(model.wv))
print('expérience_intérieure')
pp.pprint(model.wv.most_similar("expérience_intérieure",topn=50))
'''