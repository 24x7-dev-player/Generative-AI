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

	
#PPrint
query_engine = index.as_query_engine()
ans  = query_engine.query("give me the list of the candidates")
from llama_index.core.response.pprint_utils import pprint_response
pprint_response(ans,show_source=True)

#Retriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
retrever = VectorIndexRetriever(
    index = index,
    similarity_top_k = 3
)
query_engine = RetrieverQueryEngine(retriever=retrever)
ans  = query_engine.query("how many years of the expirence Arushi Gupta has ?")
pprint_response(ans,show_source=True)


#similarity Post Process
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
s_processor = SimilarityPostprocessor(similarity_cutoff=0.75)
retrever = VectorIndexRetriever(
    index = index,
    similarity_top_k = 3
)
query_engine = RetrieverQueryEngine(retriever=retrever,node_postprocessors=[s_processor])
ans  = query_engine.query("how many years of the expirence Arushi Gupta has ?")

#Persising Index
from llama_index.core import VectorStoreIndex
from llama_index.core import  SimpleDirectoryReader
documents = SimpleDirectoryReader(r"D:\llamaIndex\data_folder").load_data()
index = VectorStoreIndex(documents,show_progress = True)
	
index.storage_context.persist(persist_dir=r"D:\llmaIndex-2\storage\cache\resume\sleep")
# for readinf the index from the disk we use StorageContext , load_index_from_storage
from llama_index.core import StorageContext,load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir=r"D:\llmaIndex-2\storage\cache\resume\sleep")
index = load_index_from_storage(storage_context)

#how to count tokens used when creating and querying llamaIndex
import tiktoken
from llama_index.core import ServiceContext
from llama_index.core.callbacks import CallbackManager,TokenCountingHandler
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("text-embedding-ada-002").encode,
    verbose=True
)
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("text-embedding-ada-002").encode,
    verbose=True
)
callback_manager = CallbackManager([token_counter])
service_context = ServiceContext.from_defaults(callback_manager=callback_manager)
index = VectorStoreIndex(document,show_progress=True,service_context=service_context)
query_engine = index.as_query_engine()




#USE LLM in LLama Index
from llama_index.llms.openai import OpenAI
llm = OpenAI(temperature=0,model="gpt-3.5-turbo",max_tokens=250)
res = llm.complete("What is an AI?")

#chat Model
from llama_index.core.llms import ChatMessage
message = [
    ChatMessage(role="system",content="Talk like a 5 year olf funny and cute girls who always answer in joke"),
    ChatMessage(role="user",content="tell me about your math's teacher ? ")
]
res = llm.chat(message)


#Prmpt In LLM
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader
document = SimpleDirectoryReader(r"D:\llmaIndex-2\data_folder").load_data()
index = VectorStoreIndex(document)
res = index.as_query_engine().query("who have more expirence ?")

# lets create PromptTemplate using LLamaIndex
from llama_index.core.prompts import PromptTemplate
string = (
    "You are a Human Resource Assistant of a company.\n"
    "Your task is to find the fields asked by the Hr from the given context"
    "{context_str}\n"
    "------------------"
    "use the context information and answer the below query\n"
    "answer the question : {query_str}\n" 
    "if you are not getting the answer from the context just return N/A"
)
text_qa_template = PromptTemplate(string)