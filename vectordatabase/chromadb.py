# %pip install -Uq chromadb numpy datasets
from datasets import load_dataset
dataset = load_dataset("sciq", split="train")

# Filter the dataset to only include questions with a support
dataset = dataset.filter(lambda x: x["support"] != "")
print("Number of questions with support: ", len(dataset))

import chromadb

client = chromadb.Client()
# Create a new Chroma collection to store the supporting evidence. We don't need to specify an embedding fuction, and the default will be used.
collection = client.create_collection("sciq_supports")

# Embed and store the first 100 supports for this demo
collection.add(
    ids=[str(i) for i in range(0, 100)],  # IDs are just strings
    documents=dataset["support"][:100],
    metadatas=[{"type": "support"} for _ in range(0, 100)
    ],
)
results = collection.query(
    query_texts=dataset["question"][:10],
    n_results=1)

# Print the question and the corresponding support
for i, q in enumerate(dataset['question'][:10]):
    print(f"Question: {q}")
    print(f"Retrieved support: {results['documents'][i][0]}")
    print()
    
# pip install langchain-chroma
# import
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader


# pip install langchain-community langchain-core
loader = DirectoryLoader("/Users/myhome/Downloads/vectordata/new_articles", glob="**/*.txt", loader_cls= TextLoader)
document=loader.load()
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size =1000,chunk_overlap=200)
text = text_splitter.split_documents(document)

from langchain import embeddings
persist_directory= 'db'
embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(text, embedding, persist_directory=persist_directory)


vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

retriever = vectordb.as_retriever()
docs=retriever.get_relevant_documents("how much money does microsoft raise ")

retriever=vectordb.as_retriever(search_kwargs={"k":1})

from langchain.chains import RetrievalQA 
qa_chain=RetrievalQA.from_chain_type(llm=OpenAI(),chain_type="stuff", retriever=retriever, return_source_documents=True)
query="how much money does microsoft raise"
llm_response=qa_chain({"query":query})


def process_llm_response(llm_response) :
    print(llm_response['result'])
    print('\n\nSources: ')
    for source in llm_response[ "source_documents"]:
        print(source.metadata ['source'])
        
query="how much money does microsoft raise"
llm_response=qa_chain({"query":query})
llm_response
process_llm_response(llm_response)