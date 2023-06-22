
import pandas as pd
import numpy as np
import csv
import random



def write_rows_to_csv(rows, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

dic_rows_date_count = {}
with open('douleur_douloureux_frantext_full.csv', 'r') as file:
    reader = csv.reader(file)
    rows = list(reader)
    random.shuffle(rows)
    new_rows = []
    for row in rows:
        print(row)
        if str(row[12]).strip() not in ['douleur','douleurs']: continue
        if '-' in row[5]:continue
        if int(row[5]) < 1790:
            meta_label = "PERIODE_1"
        if int(row[5]) > 1790 and int(row[5]) < 1913:
            meta_label = "PERIODE_2"
        if int(row[5]) > 1913: 
            meta_label = "PERIODE_3"

        dic_rows_date_count.setdefault(meta_label,1)
        if dic_rows_date_count[meta_label] >667:continue
        dic_rows_date_count[meta_label]+=1

        new_rows.append(row)

print(new_rows)
print(len(new_rows))
print(dic_rows_date_count)

filename = '2000_frantext_date_distribution.csv'
write_rows_to_csv(new_rows, filename)