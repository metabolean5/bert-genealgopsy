import pandas as pd
import math
import pickle


def preformat():

	dfs1 = pd.read_excel("douleur_AAnnot_Amal.xlsx",  engine='openpyxl')
	dfs2 = pd.read_excel("douleur_AAnnot_Anaëlle.xlsx",  engine='openpyxl')
	dfs3 = pd.read_excel("douleur_AAnnot_Laure.xlsx",  engine='openpyxl')
	dfs4 = pd.read_excel("douleur_AAnnot_Martial.xlsx",  engine='openpyxl')
	dfs5 = pd.read_excel("douleur_AAnnot_Zohra.xlsx",  engine='openpyxl')


	dfss = [dfs1,dfs2,dfs3,dfs4,dfs5]


	train_test_data = []
	i = 0
	for dfs in dfss:
		for row in dfs.iterrows():
			print(row[1][19])

			if str(row[1][19]) == '5.0' or str(row[1][19]) == '5':
					continue
			try:
				train_test_data.append([str(row[1][14] + ' '+ row[1][15]+' '+ row[1][16]), int(row[1][19])])
			except:
				train_test_data.append([str(row[1][14] + ' '+ row[1][15]+' '+ row[1][16]), 3])


	train_set = train_test_data[0:803]
	test_set = train_test_data[803:1005]



	return(train_test_data)


towrite = preformat()
print(len(towrite))


with open('annotations_no5.pkl', 'wb') as f:
    # Pickle the list and write it to the file
    pickle.dump(towrite, f)
