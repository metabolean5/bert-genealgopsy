import pandas as pd
import torch
from tqdm import tqdm, trange
import numpy as np


# Importing specific libraries for data prerpcessing, model archtecture choice, training and evaluation
from sklearn.model_selection import train_test_split
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from transformers import AdamW
from utils_preformat import preformat
# import torch.optim as optim
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import seaborn as snsat

model = CamembertForSequenceClassification.from_pretrained("model")
tokenizer = CamembertTokenizer.from_pretrained('camembert-base',do_lower_case=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


val_data = preformat()[1]

texts = [i[0] for i in val_data]
labels = [i[1] for i in val_data] 


# Encode the comments
tokenized_comments_ids = [tokenizer.encode(text,add_special_tokens=True,max_length=128) for text in texts]
# Pad the resulted encoded comments
tokenized_comments_ids = pad_sequences(tokenized_comments_ids, maxlen=128, dtype="long", truncating="post", padding="post")

# Create attention masks 
attention_masks = []
for seq in tokenized_comments_ids:
	seq_mask = [float(i>0) for i in seq]
	attention_masks.append(seq_mask)

prediction_inputs = torch.tensor(tokenized_comments_ids)
prediction_masks = torch.tensor(attention_masks)


# Apply the finetuned model (Camembert)
flat_pred = []
with torch.no_grad():
    # Forward pass, calculate logit predictions
    outputs =  model(prediction_inputs.to(device),token_type_ids=None, attention_mask=prediction_masks.to(device))
    logits = outputs[0]
    logits = logits.detach().cpu().numpy() 
    flat_pred.extend(np.argmax(logits, axis=1).flatten())



valid = 0

cat_0 = 0
cat_1= 0
cat_2 = 0
cat_3 = 0
cat_4 = 0

cat_0_v = 0
cat_1_v = 0
cat_2_v = 0
cat_3_v = 0
cat_4_v = 0

actual = {"0":0, "1":0, "2":0, "3":0, "4":0  }
predictions = {"0":0, "1":0, "2":0, "3":0, "4":0  }


for i in range(len(flat_pred)):
    print('Comment: ', texts[i])
    print('prediction', flat_pred[i])
    print('actual_label', labels[i]-1)
    actual[str(labels[i]-1)]+=1

    if flat_pred[i] == labels[i] -1:
    	predictions[str(flat_pred[i])]+=1

    if flat_pred[i] == labels[i] -1:
    	valid+=1


print(predictions)
print(actual)
for key in predictions:
	print(key)
	print(predictions[key]/actual[key])



print(valid/len(labels))



#sudo scp -r /genealgoPSY mpastor@129.199.195.19:~/dev