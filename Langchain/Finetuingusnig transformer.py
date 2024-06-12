# !pip install transformers==4.30.2
# !pip install datasets

from datasets import load_dataset
import numpy as np
raw_dataset=load_dataset('glue','sst2')
from transformers import AutoTokenizer
model='distilbert-base-uncased'
tokenizer=AutoTokenizer.from_pretrained(model)
tokenized_sent=tokenizer(raw_dataset['train'][0:3]['sentence'])
from pprint import pprint
print(tokenized_sent)

def tokenized_fn(batch):
  return tokenizer(batch['sentence'],truncation=True)

tokenized_datasets=raw_dataset.map(tokenized_fn,batched=True)
from transformers import TrainingArguments

# from transformers import TrainingArguments

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir='./results',           # Output directory
#     num_train_epochs=3,               # Number of training epochs
#     per_device_train_batch_size=8,    # Batch size per device during training
#     per_device_eval_batch_size=8,     # Batch size per device during evaluation
#     logging_dir='./logs',             # Directory for storing logs
#     logging_steps=100,                # Log every 100 steps
#     save_steps=500,                   # Save checkpoint every 500 steps
#     evaluation_strategy='epoch',      # Evaluate at the end of each epoch
#     save_total_limit=2,               # Limit the total number of checkpoints
#     load_best_model_at_end=True,      # Load the best model from checkpoint at the end of training
#     metric_for_best_model='accuracy', # Metric to use for saving the best model
#     greater_is_better=True            # Whether the 'metric_for_best_model' should be maximized
# )

from transformers import Trainer, TrainingArguments
# !pip install accelerate==0.21.0
# !pip install transformers[torch]
# !pip install accelerate -U
# !pip show transformers accelerate

training_args=TrainingArguments(
    'my_trainer',                          #  Output directory where checkpoints and logs will be saved
    evaluation_strategy='epoch',           # Evaluate model at the end of each epoch
    save_strategy='epoch',                  # Save model checkpoint at the end of each epoch
    num_train_epochs=1,                     # Save model checkpoint at the end of each epoch
)


from transformers import AutoModelForSequenceClassification
checkpoint = "bert-base-uncased"
model=AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2
)

# !pip install torchinfo
from torchinfo import summary
summary(model)
params_before=[]
for name,p in model.named_parameters():
  params_before.append(p.detach().cpu().numpy())
from transformers import Trainer
from datasets import load_metric
metric=load_metric('glue','sst2')
metric.compute(predictions=[1,0,1],references=[1,0,0])
def compute_metrics(logits_and_labels):
  logits,labels=logits_and_labels
  predictions=np.argmax(logits,axis=1)
  return metric.compute(predictions=predictions,references=labels)
trainer=Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()
trainer.save_model('my_model')
from transformers import pipeline
my_model=pipeline('text-classification',model='my_model',device=0)
my_model('this movie is beautiful')