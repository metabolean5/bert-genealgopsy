# Importing standard libraries for every machine/deep learning pipeline
import pandas as pd
import numpy as np
import csv
import sys
import torch

# Importing specific libraries for data prerpcessing, model archtecture choice, training and evaluation
from sklearn.model_selection import train_test_split
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#from transformers import CamembertTokenizer, CamembertForSequenceClassification
from transformers.tokenization_bert import BertTokenizer
from transformers.modeling_bert import BertForSequenceClassification

from transformers import AdamW
from utils_preformat import preformat
# import torch.optim as optim
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import seaborn as snsat
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

BERT_MODEL = "bert-base-multilingual-uncased"

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def all_metrics(preds, labels):
    print(print("<logits>"))
    print(preds)
    print(print("<labels>"))
    print(labels)
    y_pred = np.argmax(preds, axis=1).flatten()
    y_true = labels.flatten()
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})

    # create a classification report
    report = classification_report(df['y_true'], df['y_pred'],output_dict=True)
    print("Classification report:")
    print(report)
    print("Confusion matrix:")
    print(confusion_matrix(df['y_true'], df['y_pred']))
    label_scores.setdefault('wgt',{})
    label_scores['wgt'].setdefault('wgt_avg_recall',[])
    label_scores['wgt']["wgt_avg_recall"].append(report['weighted avg']['recall'])
    label_scores['wgt'].setdefault('wgt_avg_precison',[])
    label_scores['wgt']["wgt_avg_precison"].append(report['weighted avg']['precision'])
    label_scores['wgt'].setdefault('wgt_avg_f1_score',[])
    label_scores['wgt']["wgt_avg_f1_score"].append(report['weighted avg']['f1-score'])

    precision, recall, f1_score, support = precision_recall_fscore_support(df['y_true'],  df['y_pred'])
    for i,label in enumerate([1,2,3]):
        print(f"Label: {label}")
        label_scores.setdefault(label,{})
        print(f"Precision: {precision[i]}")
        label_scores[label].setdefault('precision',[])
        label_scores[label]['precision'].append(precision[i])
        print(f"Recall: {recall[i]}")
        label_scores[label].setdefault('recall',[])
        label_scores[label]['recall'].append(recall[i])
        print(f"F1-score: {f1_score[i]}")
        label_scores[label].setdefault('f1_score',[])
        label_scores[label]['f1_score'].append(f1_score[i])
        print(f"Support: {support[i]}\n")


def bert_predict(text,):
    #print(len(validation_inputs))
    #print(validation_inputs)
    # Defining constants
    MAX_LEN = 128
    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize CamemBERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL,do_lower_case=True)

    input_ids  = [tokenizer.encode(sent,add_special_tokens=True,max_length=MAX_LEN) for sent in text]
    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Create attention masks
    attention_masks = []
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]  
        attention_masks.append(seq_mask)

    # Convert all of our data into torch tensors, the required datatype for our model
    validation_inputs = torch.tensor(input_ids)
    #validation_labels = torch.tensor(labels)
    validation_masks = torch.tensor(attention_masks)

    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
    # with an iterator the entire dataset does not need to be loaded into memory

    validation_data = TensorDataset(validation_inputs, validation_masks)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=16)
    model = BertForSequenceClassification.from_pretrained("model_merge_shorten-1", num_labels = 3)
    model.to(device)

    model.eval()
    # Evaluate data for one epoch
    all_logits = []
    all_labels_ids =  []
    i=0
    for batch in validation_dataloader:
        # Add batch to device CPU or GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
            #loss, logits = outputs[:5]

            # Move logits and labels to CPU if GPU is used

        logits = logits.detach().cpu().numpy()
        #label_ids = b_labels.to('cpu').numpy()
        pred_flat = np.argmax(logits, axis=1).flatten()
        print("\n<preds/gold>")
        print("prediction : "+ str(pred_flat))
        #print(label_ids[0] -1)
        i+=1


    return pred_flat

'''
with open('annotations_merge.pkl', 'rb') as f:
    train_data = pickle.load(f)   

text = [i[0] for i in train_data]
labels = [i[1] for i in train_data] 

labels[:] = [number - 1 for number in labels]
'''

dataset = sys.argv[1]
new_rows=[]
with open(dataset, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print('\ncurrent row : '+ str(row[0]))
        if str(row[12]).strip() not in ['douleur','douleurs']: continue
        text = [str(row[11])+' '+str(row[12])+' '+str(row[13])]
        prediction =bert_predict(text)
        new_row = row
        new_row.append(str(prediction))
        print(new_row)
        new_rows.append(new_row)
        print("prediction : " + str(prediction[0]))
        data_prediction = dataset.replace('.csv', '_prediction.csv')
        with open(data_prediction, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(new_row)
        file.close()


'''
text = [el[0]]
#labels =[el[1]]
#user tokenizer to convert sentences into tokenizer
bert_predict(text)
print("gold : " +str(el[1] -1))
print("text : " +el[0])
    
'''

