# Importing standard libraries for every machine/deep learning pipeline
import pandas as pd
import torch
from tqdm import tqdm, trange
import numpy as np


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


def bert_train_test(random_state):
    # Use train_test_split to split our data into train and validation sets for training
    train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks = train_test_split(input_ids, labels, attention_masks,
                                                                random_state=random_state, test_size=0.2)
    print(len(validation_inputs))
    print(validation_inputs)

    # Convert all of our data into torch tensors, the required datatype for our model
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
    # with an iterator the entire dataset does not need to be loaded into memory

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels = 3)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=10e-8)

    print(model.classifier.parameters)
    # Store our loss and accuracy for plotting if we want to visualize training evolution per epochs after the training process
    train_loss_set = []
    # trange is a tqdm wrapper around the normal python range
    epoch=0
    for _ in trange(epochs, desc="Epoch"):  
        # Tracking variables for training
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
      
        # Train the model
        model.train()
        for step, batch in enumerate(train_dataloader):
            # Add batch to device CPU or GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            print(b_labels)
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(b_input_ids,token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            # Get loss value
            loss = outputs[0]
            # Add it to train loss list
            train_loss_set.append(loss.item())    
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
        
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss/nb_tr_steps))
        
        # Tracking variables for validation
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        # Validation of the model
        model.eval()
        # Evaluate data for one epoch
        all_logits = []
        all_labels_ids =  []
        for batch in validation_dataloader:
            # Add batch to device CPU or GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs =  model(b_input_ids,token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss, logits = outputs[:5]

            # Move logits and labels to CPU if GPU is used
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            for logit in logits:
                all_logits.append(logit)
            for ids in label_ids:
                all_labels_ids.append(ids)

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        epoch += 1

        print('\n\n<1>')
        all_metrics(np.array(all_logits),np.array(all_labels_ids))

    model.save_pretrained("model_merge_shorten")


with open('annotations_augmented_merge.pkl', 'rb') as f:
    train_data = pickle.load(f)  

print(train_data)



text = [i[0] for i in train_data] 
'''
for i in train_data:
    tokens = i[0].split(' ')
    sentence = tokens[30:len(tokens)-30]
    text.append(' '.join(sentence))
'''

labels = [i[1] for i in train_data] 

print(labels)
labels[:] = [number for number in labels]

# Defining constants
epochs = 5
MAX_LEN = 128
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize CamemBERT tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL,do_lower_case=True)

#user tokenizer to convert sentences into tokenizer
input_ids  = [tokenizer.encode(sent,add_special_tokens=True,max_length=MAX_LEN) for sent in text]
# Pad our input tokens
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Create attention masks
attention_masks = []
# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]  
    attention_masks.append(seq_mask)

label_scores = {}
for random_state in range(55,60):
    msg="cross validation fold on random_state : "+ str(random_state)
    print(msg)
    bert_train_test(random_state)

print(label_scores)
print("\nFINAL RESULTS")

for cat in label_scores:
    print('\n'+str(cat))
    for score in label_scores[cat]:
        print(score + ' : '+ str(np.mean(label_scores[cat][score])))



