text = """ code geass is one of those series that everybody recomends and once you watch it you know why.
Mechas, war, romance, comedy, rivality, unexpected twsits, explotions, mindgames and more mechas! what else do you want!!! xD it totally deserver its place as one of the best animes ever made.
the only aspect i see as a flaw is the art with a 9/10 the people seem a little bit deformed or too skinny but once you get use to it is great!.The user wants freedom.
and it has by far the best story i have ever seen you just get hooked to it from the second episode and cant stop until finishing it.
almost perfect serie, recomended to anyone who is a human beeing (nevermind, my dog liked it) """

# !pip install transformers
# !pip install datasets
# !pip install sentencepiece

import datasets
import huggingface_hub
import matplotlib.pyplot as plt
import transformers

# Text Classification
from transformers import pipeline
classifier = pipeline('text-classification')
classifier(text)


#Named Entity Recoginition
import pandas as pd
text1 = "Narendra Modi is prime minister of India"
ner = pipeline('ner', aggregation_strategy='simple')
out = ner(text1)
print(pd.DataFrame(out))

#QuestionAnswer
reader = pipeline('question-answering')
question = 'what does user want ?'
outputs = reader(question=question,context = text)
pd.DataFrame([outputs])

#Summerization
summarizer = pipeline('summarization')
outputs = summarizer( text, clean_up_tokenization_spaces=True, max_length=90)
print(outputs[0]['summary_text'])

#translation
translate = pipeline('translation_en_to_de', model='Helsinki-NLP/opus-mt-en-de')
output = translate(text,clean_up_tokenization_spaces=True,min_length=120)
print(output[0]['translation_text'])

#test Classification
# !pip install --upgrade pip
# !pip install transformers
# !pip install datasets
# !pip install sentencepiece

# Emotion detector
from datasets import list_datasets
all_datasets = list_datasets()
print(f"There are total of {len(all_datasets)} in the hub")
print(f"Some example datasets are {all_datasets[:5]}")
from datasets import load_dataset
emotion_dataset = load_dataset('emotion')
train_dataset = emotion_dataset['train']
# !wget https://raw.githubusercontent.com/nachikethmurthy/Source-Code-Dataset-for-Machine-Learning-using-Python/main/Data/sentiment_train
# load the dataset
sentiment_train = load_dataset('csv',data_files = "sentiment_train",sep='\t')
import pandas as pd
emotion_dataset.set_format(type='pandas')
df = emotion_dataset['train'][:]

def label_to_str(row):
  return emotion_dataset['train'].features['label'].int2str(row)
df['labels_name'] = df['label'].apply(label_to_str)

emotion_dataset.reset_format()   #Hugging Face datasets library, such as the DatasetDict
from transformers import AutoTokenizer
model = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model)
text = "My favorite food is Masala Dosa"
encoded_text = tokenizer(text)
print(encoded_text)


#tokensization 
def tokenization(batch):
  return tokenizer(batch['text'], padding=True, truncation=True)
tokenization(emotion_dataset['train'][:2])
emotions_encoded = emotion_dataset.map(tokenization, batched=True, batch_size=None)
emotions_encoded
# !pip install accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistral-community/Mixtral-8x22B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)

text = "Hello my name is"
inputs = tokenizer(text, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


#Table Question Answer

from transformers import pipeline
import pandas as pd
# prepare table + question
data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
table = pd.DataFrame.from_dict(data)
question = "how many movies does Leonardo Di Caprio have?"

# pipeline model
# Note: you must to install torch-scatter first.
tqa = pipeline(task="table-question-answering", model="google/tapas-large-finetuned-wtq")

# result

print(tqa(table=table, query=question)['cells'][0])
#53

#Document Question Answer
from transformers import pipeline
from PIL import Image

pipe = pipeline("document-question-answering", model="naver-clova-ix/donut-base-finetuned-docvqa")

question = "What is the purchase amount?"
image = Image.open("/content/Screenshot 2024-04-13 021038.jpg")

pipe(image=image, question=question)


# Install the bert_score library
# !pip install bert_score
# Import the necessary functions
from bert_score import score
# Prepare your predictions and references
predictions = ["hello world", "general kenobi"]
references = ["hello world", "general kenobi"]

# Compute BERTScore specifying the language (English)
precision, recall, f1 = score(predictions, references, lang="en")
# Print the results
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
# !pip install evaluate


import nltk
from nltk.translate.bleu_score import sentence_bleu

# Define candidate and reference sentences
candidate = "hello world"
reference = ["hello world", "general kenobi"]

# Tokenize candidate and reference sentences
candidate_tokens = candidate.split()
reference_tokens = [ref.split() for ref in reference]

# Calculate BLEU score
bleu_score = sentence_bleu(reference_tokens, candidate_tokens)

# Print BLEU score
print("BLEU Score:", bleu_score)
