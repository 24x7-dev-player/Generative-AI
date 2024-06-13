# !pip install langchain
# !pip install pinecone-client
# !pip install pypdf

from langchain. document_loaders import PyPDFDirectoryLoader
from langchain. text_splitter import RecursiveCharacterTextSplitter
from langchain. embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI

from langchain. vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from langchain. document_loaders import PyPDFLoader

loader = PyPDFLoader('/Users/myhome/Downloads/vectordata/Rich-Dad-Poor-Dad.pdf')
data=loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size =100,chunk_overlap=20)
text = text_splitter.split_documents(data)
text

embeddings = OpenAIEmbeddings()
result=embeddings.embed_query("hello")
len(result)

##pinecone                        
from langchain_community.vectorstores import FAISS
db1=FAISS.from_documents(text,embeddings)

query="give 3 learning form this book"
result =db1.similarity_search(query)
result[2]

