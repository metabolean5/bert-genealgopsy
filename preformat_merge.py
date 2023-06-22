import pandas as pd
import math
import pickle
import json


def preformat():

	dfs1 = pd.read_excel("douleur_AAnnot_Amal.xlsx",  engine='openpyxl')
	dfs2 = pd.read_excel("douleur_AAnnot_Anaëlle.xlsx",  engine='openpyxl')
	dfs3 = pd.read_excel("douleur_AAnnot_Laure.xlsx",  engine='openpyxl')
	dfs4 = pd.read_excel("douleur_AAnnot_Martial.xlsx",  engine='openpyxl')
	dfs5 = pd.read_excel("douleur_AAnnot_Zohra.xlsx",  engine='openpyxl')


	dfss = [dfs1,dfs2,dfs3,dfs4,dfs5]
	cat_dic ={}


	train_test_data = []
	i = 0
	for dfs in dfss:
		for row in dfs.iterrows():

						
			if str(row[1][19]) == '5.0' or str(row[1][19]) == '5':
				cat = 3

			if str(row[1][19]) in ["1.0",'1','2.0','2']:
				cat = 1

			if str(row[1][19]) in ["3.0",'3','4.0','4']:
				cat = 2
			try:
				train_test_data.append([str(row[1][14] + ' '+ row[1][15]+' '+ row[1][16]), cat])
				rowcat = int(row[1][19])
			except:
				rowcat = row[1][19] = 1
				cat = 1
				train_test_data.append([str(row[1][14] + ' '+ row[1][15]+' '+ row[1][16]), cat])

			cat_dic.setdefault(rowcat,[])
			cat_dic[rowcat].append([str(row[1][14] + ' '+ row[1][15]+' '+ row[1][16]), cat])

	train_set = train_test_data[0:803]
	test_set = train_test_data[803:1005]



	return train_test_data, cat_dic


towrite,cat_dic = preformat()


with open("cat_dic.json", "w") as f:
    json.dump(cat_dic, f, indent=4)

print(towrite)
print(len(towrite))

with open('annotations_merge.pkl', 'wb') as f:
    # Pickle the list and write it to the file
    pickle.dump(towrite, f)

#add 251 to 1
#add 15 to 2