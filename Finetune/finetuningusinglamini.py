import lamini
lamini.api_key = "lamini API key"

llm = lamini.Lamini("meta-llama/Meta-Llama-3-8B-Instruct")
print(llm.generate("How are you?")) 
from llama import BasicModelRunner
non_finetuned = BasicModelRunner("meta-llama/Llama-2-7b-chat-hf")
non_finetuned_output = non_finetuned("Tell me how to train my dog to sit")
non_finetuned_output

import jsonlines
import itertools
import pandas as pd
from pprint import pprint

import datasets
from datasets import load_dataset
pretrained_dataset = load_dataset("c4", "en", split="train", streaming=True)

# Common Crawl datase: https://huggingface.co/datasets/c4
n = 5
print("dataset_pretrained")
top_n = itertools.islice(pretrained_dataset, n)
for i in top_n:
  print(i)
filename = "lamini_docs.jsonl"
instruction_dataset_df = pd.read_json(filename, lines=True)
instruction_dataset_df
examples = instruction_dataset_df.to_dict()
text = examples["question"][0] + examples["answer"][0]
text
if "question" in examples and "answer" in examples:
  text = examples["question"][0] + examples["answer"][0]
elif "instruction" in examples and "response" in examples:
  text = examples["instruction"][0] + examples["response"][0]
elif "input" in examples and "output" in examples:
  text = examples["input"][0] + examples["output"][0]
else:
  text = examples["text"][0]
prompt_template = """### Question:
{question}

### Answer:
{answer}"""
question = examples["question"][0]
answer = examples["answer"][0]

text_with_prompt_template = prompt_template.format(question=question, answer=answer)
text_with_prompt_template