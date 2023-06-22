import transformers
from transformers import CamembertForSequenceClassification, CamembertTokenizer
from torch.utils.data import TensorDataset, RandomSampler, DataLoader



from utils_preformat import preformat

train_data = preformat()[0]
val_data = preformat()[1]


texts = [i[0] for i in train_data]
labels = [i[1] for i in train_data]


# Load the Camembert model and tokenizer
model = CamembertForSequenceClassification.from_pretrained("camembert-base")
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

# Prepare your dataset
# Assuming your dataset is a list of tuples (text, label) where label is an integer in [0, 4]

encoded_texts = tokenizer(texts, return_tensors="pt", padding=True ,truncation=True)

# Create the DataLoader
batch_size = 32
dataset = TensorDataset(encoded_texts["input_ids"], encoded_texts["attention_mask"], labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Fine-tune the model
model.num_labels = 5
model.train()
optimizer = transformers.AdamW(model.parameters(), lr=2e-5)
loss_fn = transformers.CrossEntropyLoss()

for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


'''
# Save the fine-tuned model
model.save_pretrained("/path/to/save/model")
'''