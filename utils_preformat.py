import pandas as pd
import math
import pickle
import random
 

def preformat():

	dfs1 = pd.read_excel("douleur_AAnnot_Amal.xlsx",  engine='openpyxl')
	dfs2 = pd.read_excel("douleur_AAnnot_Anaëlle.xlsx",  engine='openpyxl')
	dfs3 = pd.read_excel("douleur_AAnnot_Laure.xlsx",  engine='openpyxl')
	dfs4 = pd.read_excel("douleur_AAnnot_Martial.xlsx",  engine='openpyxl')
	dfs5 = pd.read_excel("douleur_AAnnot_Zohra.xlsx",  engine='openpyxl')


	dfss = [dfs1,dfs2,dfs3,dfs4,dfs5]
	invalid_date_format = ["1789-1798","1832-1833"]


	train_test_data = []
	i = 0
	for dfs in dfss:
		for row in dfs.iterrows():


			if row[1][8] in invalid_date_format:
				row[1][8] = 1780

			if int(row[1][8]) < 1790:
				meta_label = "PERIODE_1"
			if int(row[1][8]) > 1790 and int(row[1][8]) < 1913:
				meta_label = "PERIODE_2"
			if int(row[1][8]) > 1913: 
				meta_label = "PERIODE_3"

			try:
				if int(row[1][19]) == 5:
					continue
				train_test_data.append((meta_label +' '+ str(row[1][14] + ' '+ row[1][15]+' '+ row[1][16]), int(row[1][19])))
			except:
				train_test_data.append((meta_label +' '+ str(row[1][14] + ' '+ row[1][15]+' '+ row[1][16]), 3))


	with open('fullcombined_dataset/annotations-metadate.txt', 'wb') as f:
		pickle.dump(train_test_data,f)  


	print(len(train_test_data))

	return(train_test_data)


preformat()