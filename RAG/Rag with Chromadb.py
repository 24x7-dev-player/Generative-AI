# import libraries
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') # add your OpenAI API Key


DOC_PATH = "your.pdf"
CHROMA_PATH = "your_db_name"




# ----- Data Indexing Process -----

# load your pdf doc
loader = PyPDFLoader(DOC_PATH)
pages = loader.load()

# split the doc into smaller chunks i.e. chunk_size=500
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(pages)

# get OpenAI Embedding model
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# embed the chunks as vectors and load them into the database
db_chroma = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)





# ----- Retrieval and Generation Process -----






# this is an example of a user question (query)
query = 'what are the top risks mentioned in the document?'

# retrieve context - top 5 most relevant (closests) chunks to the query vector
# (by default Langchain is using cosine distance metric)
docs_chroma = db_chroma.similarity_search_with_score(query, k=5)

# generate an answer based on given user query and retrieved context information
context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])

# you can use a prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {question}.
Provide a detailed answer.
Don’t justify your answers.
Don’t give information not mentioned in the CONTEXT INFORMATION.
Do not say "according to the context" or "mentioned in the context" or similar.
"""




# load retrieved context and user query in the prompt template
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=query)

# call LLM model to generate the answer based on the given context and query
model = ChatOpenAI()
response_text = model.predict(prompt)

from langchain_community.document_loaders import PyPDFLoader

pdf = PyPDFLoader ("/Users/myhome/Downloads/llmops/Catalogue.pdf")

pdfpages = pdf.load_and_split()

from langchain.text_splitter import CharacterTextSplitter
from langchain_community. vectorstores import FAISS

from langchain_openai.embeddings import OpenAIEmbeddings

import os

from langchain_openai import OpenAI


os.environ["OPENAI_API_KEY"]=api_key

#indexxing
mybooks = pdf.load ()

text_splitter = CharacterTextSplitter (chunk_size=1500, chunk_overlap=0)

split_text = text_splitter.split_documents (mybooks)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_text, embeddings)


# retrival. and generatiion
vectorstore_retriever = vectorstore.as_retriever()

from langchain.agents.agent_toolkits import create_retriever_tool

tool = create_retriever_tool(
    vectorstore_retriever,

    "catalogue_pdf",
    "Retrieve detailed information on title and language here which title is related to which language ."
)

tools = [tool]

from langchain.agents.agent_toolkits import create_conversational_retrieval_agent

from langchain_openai.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature = 0, model_name="gpt-3.5-turbo")

myagent = create_conversational_retrieval_agent(llm, tools, verbose=True)



context = "onl you have to give the answer of that book title  language fetch from the pdf ."
question = "What is the language of the book title Banaras ke ghat "

prompt = f"""You need to answer the question in the sentence as same as in the pdf content.
Given below is the context and question of the user.
context = {context}
question = {question}

"""

result = myagent.invoke({"input": prompt})