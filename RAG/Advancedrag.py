import os
import re
import nltk
import bs4
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader,ArxivLoader
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI




# # ! pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain cohere
# # pip install 'protobuf<=3.20.1' --force-reinstall
# # !python3 -m pip install pip --upgrade
# # !pip install pyopenssl --upgrade
# !pip install pymupdf
# !pip install langchain-cohere

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = ""

os.environ['COHERE_API_KEY'] = ""
os.environ['OPENAI_API_KEY'] = ""

### Paid
# llm= ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# embeddings=OpenAIEmbeddings()

### Free
llm = Ollama(model="llama3")
embeddings=CohereEmbeddings()

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Step 1: Load the documents
docs = ArxivLoader(query='2312.10997', load_max_docs=1, load_all_available_meta=True).load()
load_docs = docs[0].page_content

# Step 2: Filter complex metadata
filtered_docs = filter_complex_metadata(docs)

# Step 3: Split the documents
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=50)
splits = text_splitter.split_documents(filtered_docs)

# Step 4: Embed the documents
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Step 5: Create the retriever
retriever = vectorstore.as_retriever()

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# Chain
rag_chain = (
    {"context": retriever | format_docs , "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
rag_chain.invoke("What is Modular RAG ?")


# Prompt
template = """You are a Q&A assistant, You will refer the context provided and answer the question. 
If you dont know the answer , reply that you dont know the answer:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
prompt


# Multi Query: Different Perspectives
template = """You are an AI language model assistant. 
Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector 
database. 
By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: 
{question}"""

prompt_perspectives = ChatPromptTemplate.from_template(template)


generate_queries = (
    prompt_perspectives 
    | llm 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

from langchain.load import dumps, loads

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

# Retrieve
question = "What is Modular RAG ?"
retrieval_chain = generate_queries | retriever.map() | get_unique_union
docs = retrieval_chain.invoke({"question":question})
print(docs[0].page_content)

# RAG
from operator import itemgetter

template = """Answer the following question based on this context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    {"context": retrieval_chain, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"question":question})


from langchain.prompts import ChatPromptTemplate

# Decomposition
template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(template)
# Chain
generate_queries_decomposition = ( prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))

# Run
question = "I dont understand RAG ,Can you help me understand what are the components and one more thing I would like to know about whether is it same as Advanced RAG ?"
#### I gave an ambigious query which talks about 3 questions 1. RAG understanding 2. Components of RAG 3. Difference between RAG & Advanced RAG
questions = generate_queries_decomposition.invoke({"question":question})

questions


# Prompt
template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

decomposition_prompt = ChatPromptTemplate.from_template(template)

from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser

def format_qa_pair(question, answer):
    """Format Q and A pair"""
    
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()



q_a_pairs = ""
for q in questions:
    
    rag_chain = (
    {"context": itemgetter("question") | retriever, 
     "question": itemgetter("question"),
     "q_a_pairs": itemgetter("q_a_pairs")} 
    | decomposition_prompt
    | llm
    | StrOutputParser())

    answer = rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})
    q_a_pair = format_qa_pair(q,answer)
    q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair
    
print(q_a_pairs)    


from langchain.prompts import ChatPromptTemplate

# HyDE document genration
template = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""
prompt_hyde = ChatPromptTemplate.from_template(template)

from langchain_core.output_parsers import StrOutputParser

generate_docs_for_retrieval = (
    prompt_hyde | llm | StrOutputParser() 
)

# Run
generate_docs_for_retrieval.invoke({"question":question})
# Retrieve
retrieval_chain = generate_docs_for_retrieval | retriever 
retireved_docs = retrieval_chain.invoke({"question":question})
retireved_docs

# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"context":retireved_docs,"question":question})


from langchain.prompts import ChatPromptTemplate

# RAG-Fusion
template = """You are an assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):
"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)

from langchain_core.output_parsers import StrOutputParser


generate_queries = (
    prompt_rag_fusion 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

from langchain.load import dumps, loads

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

question = "What is pattern in Modular RAG ?"
retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
docs = retrieval_chain_rag_fusion.invoke({"question": question})

print(docs)


from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

# RAG
template = """
Answer the following question based on this context, 
If you dont find any answer then just revert with 'Answer not found'.
context: {context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

#llm = ChatOpenAI(temperature=0)

final_rag_chain = (
    {"context": retrieval_chain_rag_fusion, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"question":question})


from langchain_community.llms import Cohere
from langchain.retrievers import  ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

# Chain
normal_rag_chain = (
    {"context": retriever | format_docs , "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# Question
normal_rag_chain.invoke("What is pattern in Modular RAG ?")


# Re-rank
top_k=5
compressor = CohereRerank(top_n=top_k)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)
question="What is pattern in Modular RAG ?"
compressed_docs = compression_retriever.get_relevant_documents(question)

#### After using reranker
reranked_rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)

print("Answer after reranking comes out to be: ")
print(reranked_rag_chain.invoke({"context":compressed_docs,"question":question}))


# The retrieved source documents
print("\nRetrieved Documents:")
for i in range(top_k):
    print(f"\nDocument {i+1}:")
    print(compressed_docs[0].page_content)  # or doc.text depending on the document structure
    
    
    ###### RAG & Applying Cohere Reranker for document extraction
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA


from langchain import hub

# # Loads the latest version
# prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")



# Path to the PDF file

# Step 1: Load the documents
docs = ArxivLoader(query='2312.10997', load_max_docs=1, load_all_available_meta=True).load()
load_docs = docs[0].page_content

# Step 2: Filter complex metadata
data = filter_complex_metadata(docs)



# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
all_splits = text_splitter.split_documents(data)

# Store splits
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Create a vector store with Chroma
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)

# RetrievalQA
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI


# Set up the prompt template if needed
prompt_template = """
Answer the following question based on the provided context.

{context}

Question: {question}
Answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Create the retriever with a specified top_k value
retriever = vectorstore.as_retriever(search_kwargs={"k": 25})  # Set top_k to 25

# Create the QA chain with the retriever and prompt
qa_chain = RetrievalQA.from_chain_type(
    llm, retriever=retriever, chain_type_kwargs={"prompt": prompt}, return_source_documents=True
)

# Run a query and see the results along with the context documents
query = "Explain different components of Modular RAG?"
result = qa_chain(query)

# The answer to the question
print("Answer:", result['result'])

# The retrieved source documents
print("\nRetrieved Documents:")
for i, doc in enumerate(result['source_documents']):
    print(f"\nDocument {i+1}:")
    print(doc.page_content)  # or doc.text depending on the document structure


# Re-Rank them with cohere
import cohere
# Get your cohere API key on: www.cohere.com
co = cohere.Client(f"{os.environ['COHERE_API_KEY']}")
docs = [doc.page_content for doc in result['source_documents']]

# Re-Rank them with cohere
top_n=5
rerank_hits = co.rerank(query=query, documents=docs, top_n=top_n, model='rerank-multilingual-v3.0')
print(rerank_hits)
#[doc[rerank_hits.results[i].index] for i in range(5)]

for i, doc in enumerate(docs):
    if i>top_n-1:
        break
    else:
        print(f"\nDocument {i}:")
        print(f"Relevance score on the basis of reranking is : {rerank_hits.results[i].relevance_score}") 
        print(docs[rerank_hits.results[i].index])  


# 1. Character Text Splitting
print("#### Character Text Splitting ####")

text = """In 2024, India will find itself at the center of global attention, hosting both the highly anticipated general elections and the Cricket World Cup. On the political front, the general elections will see over 900 million eligible voters making their voices heard in a democratic exercise unparalleled in scale. 
Political parties are already mobilizing their bases, with campaigns focused on critical issues like economic growth, social justice, and national security. Meanwhile, cricket fever will grip the nation as teams from around the world compete for glory in the ICC Cricket World Cup. 
Stadia will roar with the cheers of passionate fans, and cricket pitches will become the stage for thrilling displays of skill and sportsmanship. 
As politicians rally for votes and cricketers battle for the championship, these parallel events will underscore the dual fervor that defines India's national identity: a deep commitment to democracy and an unbridled love for cricket
"""
# Manual Splitting
chunks = []
chunk_size = 35 # Characters
for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)
documents = [Document(page_content=chunk, metadata={"source": "local"}) for chunk in chunks]
print(documents)

# Automatic Text Splitting
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size = 35, chunk_overlap=0, separator='', strip_whitespace=False)
documents = text_splitter.create_documents([text])
print(documents)


# 2. Recursive Character Text Splitting
print("#### Recursive Character Text Splitting ####")

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 65, chunk_overlap=0) # ["\n\n", "\n", " ", ""] 65,450
print(text_splitter.create_documents([text])) 

# 3. Document Specific Splitting
print("#### Document Specific Splitting ####")

# Document Specific Splitting - Markdown
from langchain.text_splitter import MarkdownTextSplitter
splitter = MarkdownTextSplitter(chunk_size = 40, chunk_overlap=0)
markdown_text = text
print(splitter.create_documents([markdown_text]))

# Document Specific Splitting - Python
from langchain.text_splitter import PythonCodeTextSplitter
python_text = """
class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

p1 = Person("John", 36)

for i in range(10):
    print (i)
"""
python_splitter = PythonCodeTextSplitter(chunk_size=100, chunk_overlap=0)
print(python_splitter.create_documents([python_text]))

# Document Specific Splitting - Javascript
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
javascript_text = """
// Function is called, the return value will end up in x
let x = myFunction(4, 3);

function myFunction(a, b) {
// Function returns the product of a and b
  return a * b;
}
"""
js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS, chunk_size=65, chunk_overlap=0
)
print(js_splitter.create_documents([javascript_text]))


# 4. Semantic Chunking
print("#### Semantic Chunking ####")

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# Percentile - all differences between sentences are calculated, and then any difference greater than the X percentile is split
text_splitter = SemanticChunker(OpenAIEmbeddings())
text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="percentile" # "standard_deviation", "interquartile"
)
documents = text_splitter.create_documents([text])
print(documents)

