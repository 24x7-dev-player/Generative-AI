# # installing the lib
# pip install llama-index
# pip install llama-hub
# pip install openai


# setting up the openai api key in os envoirment
import os
import openai
os.environ["OPENAI_API_KEY"] = "sk-xxxxxxx"
openai.api_key = "nananana"

from llama_index.core import VectorStoreIndex,SimpleDirectoryReader
documents = SimpleDirectoryReader(r"D:\llmaIndex-2\data_folder").load_data()
index = VectorStoreIndex.from_documents(documents,show_progress=True)
query_engine = index.as_query_engine()
response = query_engine.query("What are the names of the candidates?")
response.response