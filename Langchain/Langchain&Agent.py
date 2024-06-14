from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Access the API key from the environment variables
api_key = os.getenv("OPEN_API_KEY")

from langchain.llms import OpenAI
llm = OpenAI()

response = llm.invoke("top 5 ipl team")
print(response)

for chunk in llm.stream("give three  ipl palyer"):
    print(chunk, end="", flush=True)



"""# Chat Messages
## Like text, but specified with a message type (System, Human, AI)

### System - Helpful background context that tell the AI what to do
### Human - Messages that are intented to represent the user
### AI - Messages that show what the AI responded with
"""

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

chat(
    [
        SystemMessage(content="You are a nice AI bot that helps a user figure out what to eat in one short sentence"),
        HumanMessage(content="I like tomatoes, what should I eat?")
    ]
)

chat(
    [
        SystemMessage(content="You are a nice AI bot that helps a user figure out where to travel in one short sentence"),
        HumanMessage(content="I like the beaches where should I go?"),
        AIMessage(content="You should go to Nice, France"),
        HumanMessage(content="What else should I do when I'm there?")
    ]
)

chat(
    [
        HumanMessage(content="What day comes after Thursday?")
    ]
)

from langchain.schema import Document

Document(page_content="This is my document. It is full of text that I've gathered from other places",
         metadata={
             'my_document_id' : 234234,
             'my_document_source' : "The LangChain Papers",
             'my_document_create_time' : 1680013019
         })

Document(page_content="This is my document. It is full of text that I've gathered from other places")

from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI()

from langchain.schema.messages import HumanMessage, SystemMessage
messages = [
    SystemMessage(content="You are virat kohli."),
    HumanMessage(content="Which shoe manufacturer are you associated with?"),
]
response = chat.invoke(messages)
print(response.content)

from langchain.embeddings import OpenAIEmbeddings

embeddings=OpenAIEmbeddings()

text = "Hi! It's time for the beach"


text_embedding = embeddings.embed_query(text)

print (f"Here's a sample: {text_embedding[:2]}...")
print (f"Your embedding is length {len(text_embedding)}")





from langchain.llms import OpenAI


# I like to use three double quotation marks for my prompts because it's easier to read
prompt = """
Today is Monday, tomorrow is Wednesday.

What is wrong with that statement?
"""

print(llm(prompt))

from langchain.prompts import PromptTemplate

# Simple prompt with placeholders
prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} cricket  {content}."
)

# Filling placeholders to create a prompt
filled_prompt = prompt_template.format(adjective="great", content="player name")
print(filled_prompt)

from langchain.llms import OpenAI



# I like to use three double quotation marks for my prompts because it's easier to read
prompt = """
Today is Monday, tomorrow is Wednesday.

What is wrong with that statement?
"""

print(llm(prompt))

from langchain.prompts import ChatPromptTemplate

# Defining a chat prompt with various roles
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
    ]
)

# Formatting the chat prompt
formatted_messages = chat_template.format_messages(name="prince", user_input="What is your name?")
for message in formatted_messages:
    print(message)

from langchain.llms import OpenAI
from langchain import PromptTemplate


# Notice "location" below, that is a placeholder for another value later
template = """
I really want to travel to {location}. What should I do there?
give me name of three historic place ot visit in india only

Respond in one short sentence
"""

prompt = PromptTemplate(
    input_variables=["location"],
    template=template,
)

final_prompt = prompt.format(location='kanpur')

print (f"Final Prompt: {final_prompt}")
print ("-----------")
print (f"LLM Output: {llm(final_prompt)}")

pip install OpenAI

from langchain.output_parsers.json import SimpleJsonOutputParser


model = OpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
# Create a JSON prompt
json_prompt = PromptTemplate.from_template(
    "Return a JSON object with `birthdate` and `birthplace` key that answers the following question: {question}"
)

# Initialize the JSON parser
json_parser = SimpleJsonOutputParser()

# Create a chain with the prompt, model, and parser
json_chain = json_prompt | model | json_parser

# Stream through the results
result_list = list(json_chain.stream({"question": "bithday and birthplace of saurabh yadav"}))

# The result is a list of JSON-like dictionaries
print(result_list)

from langchain.output_parsers.json import SimpleJsonOutputParser

json_prompt = PromptTemplate.from_template(
    "Return a JSON object with an `answer` key that answers the following question: {question}"
)
json_parser = SimpleJsonOutputParser()
json_chain = json_prompt | model | json_parser

list(json_chain.stream({"question": "Who invented the microscope?"}))



from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Initialize the parser
output_parser = CommaSeparatedListOutputParser()

# Create format instructions
format_instructions = output_parser.get_format_instructions()

# Create a prompt to request a list
prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions}
)

# Define a query to prompt the model
query = "indian premier cirkcet team"

# Generate the output
output = model(prompt.format(subject=query))

# Parse the output using the parser
parsed_result = output_parser.parse(output)

# The result is a list of items
print(parsed_result)

from langchain.document_loaders import TextLoader

loader = TextLoader("/Users/myhome/Downloads/LANG/sample1.txt")
document = loader.load()
document

from langchain.text_splitter import RecursiveCharacterTextSplitter

with open('/Users/myhome/Downloads/LANG/sample1.txt') as f:
    pg_work = f.read()

print (f"You have {len([pg_work])} document")

from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("/Users/myhome/Downloads/LANG/Finance_Bill.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 150,
    chunk_overlap  = 20,
)

texts = text_splitter.create_documents([pg_work])

print (f"You have {len(texts)} documents")

print ("Preview:")
print (texts[0].page_content, "\n")
print (texts[1].page_content)

from langchain.text_splitter import RecursiveCharacterTextSplitter

state_of_the_union = "this is generative ai class in which i today"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    add_start_index=True,
)

texts = text_splitter.create_documents([state_of_the_union])
print(texts[0])  # Access the only element in the texts list

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

loader = TextLoader("/Users/myhome/Downloads/LANG/sample1.txt")
documents = loader.load()

# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

# Split your docs into texts
texts = text_splitter.split_documents(documents)

# Get embedding engine ready
embeddings = OpenAIEmbeddings()

# Embedd your texts
db = FAISS.from_documents(texts, embeddings)
db

# Init your retriever. Asking for just 1 document back
retriever = db.as_retriever()
retriever

docs = retriever.get_relevant_documents("what types of things did the author want to build?")print("\n\n".join([x.page_content[:200] for x in docs[:2]]))

print("\n\n".join([x.page_content[:200] for x in docs[:2]]))









from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([state_of_the_union])
print(texts[0])

from langchain.embeddings import OpenAIEmbeddings

# Initialize the model
embeddings_model = OpenAIEmbeddings()

# Embed a list of texts
embeddings = embeddings_model.embed_documents(
    ["Hi there!", "Oh, hello!", "What's your name?", "My friends call me World", "Hello World!"]
)
print("Number of documents embedded:", len(embeddings))
print("Dimension of each embedding:", len(embeddings[0]))

from langchain.embeddings import OpenAIEmbeddings

# Initialize the model
embeddings_model = OpenAIEmbeddings()

# Embed a single query
embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
print("First five dimensions of the embedded query:", embedded_query[:5])



from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

import fitz  # PyMuPDF
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Open the PDF file and extract text
with fitz.open("Finance_Bill.pdf") as doc:
    full_text = ""
    for page in doc:
        full_text += page.get_text()

# Continue with the rest of your code
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_text(full_text)

embeddings = OpenAIEmbeddings()
db = Chroma.from_texts(texts, embeddings)
retriever = db.as_retriever()

retrieved_docs = retriever.invoke("what is the income-tax c")
print(retrieved_docs[0].page_content)













"""# part 2 👍"""







"""Why LangChain?
Components - LangChain makes it easy to swap out abstractions and components necessary to work with language models.

Customized Chains - LangChain provides out of the box support for using and customizing 'chains' - a series of actions strung together.

Speed 🚢 - This team ships insanely fast. You'll be up to date with the latest LLM features.

Community 👥 - Wonderful discord and community support, meet ups, hackathons, etc.

## Main Use Cases
Summarization - Express the most important facts about a body of text or chat interaction

Question and Answering Over Documents - Use information held within documents to answer questions or query
Extraction - Pull structured data from a body of text or an user query
Evaluation - Understand the quality of output from your application
Querying Tabular Data - Pull data from databases or other tabular source
Code Understanding - Reason about and digest code
Interacting with APIs - Query APIs and interact with the outside world
Chatbots - A framework to have a back and forth interaction with a user combined with memory in a chat interface
Agents - Use LLMs to make decisions about what to do next. Enable these decisions with tools.

# Module III : Agents
"""

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
import json

llm = OpenAI(temperature=0, openai_api_key=api_key)

# Load environment variables from the .env file
load_dotenv()
serpapi_api_key = os.getenv("SERP_API_KEY")

toolkit = load_tools(["serpapi"], llm=llm, serpapi_api_key=serpapi_api_key )

agent = initialize_agent(toolkit, llm, agent="zero-shot-react-description", verbose=True, return_intermediate_steps=True)

response = agent({"input":"what was the first hindi movie of the"
                    "of which the amitabh bachan was a part of "})

response = agent({"input":"what was the first hindi movie of the"
                    "of which the amitabh bachan was a part of "})

response = agent({"input":"What is the capital of india?"})
response['output']



from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

model = OpenAI(model_name="gpt-3.5-turbo", temperature=0.0)

llm.invoke("how many letters in the word prince?")

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.1)

tools = load_tools(["ddg-search", "llm-math", "wikipedia"], llm=llm)

tools[0].name, tools[0].description



"""## Initialize the agent"""

agent = initialize_agent(tools,
                         llm,
                         agent="zero-shot-react-description",
                         verbose=True)

print(agent.agent.llm_chain.prompt.template)



"""# chains"""

from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions."),
    ("human", "{question}")
])
runnable = prompt | model | StrOutputParser()

for chunk in runnable.stream({"question": "What are the seven wonders of the world"}):
    print(chunk, end="", flush=True)

"""# 1. Simple Sequential Chains"""



from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain

# Commented out IPython magic to ensure Python compatibility.

template = """Your job is to come up with a classic dish from the area that the users suggests.
# % USER LOCATION
{user_location}

YOUR RESPONSE:
"""
prompt_template = PromptTemplate(input_variables=["user_location"], template=template)

location_chain = LLMChain(llm=llm, prompt=prompt_template)

# Commented out IPython magic to ensure Python compatibility.
template = """Given a meal, give a short and simple recipe on how to make that dish at home.
# % MEAL
{user_meal}

YOUR RESPONSE:
"""
prompt_template = PromptTemplate(input_variables=["user_meal"], template=template)

# Holds my 'meal' chain
meal_chain = LLMChain(llm=llm, prompt=prompt_template)

overall_chain = SimpleSequentialChain(chains=[location_chain, meal_chain], verbose=True)

review = overall_chain.run("delhi")



"""# 2. Summarization Chain"""

from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = TextLoader('/Users/myhome/Downloads/LANG/sample1.txt')
documents = loader.load()

# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)

# Split your docs into texts
texts = text_splitter.split_documents(documents)

# There is a lot of complexity hidden in this one line. I encourage you to check out the video above for more detail
chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
chain.run(texts)

from langchain import PromptTemplate, OpenAI, LLMChain

# the language model
llm = OpenAI(temperature=0)

# the prompt template
prompt_template = "Act like a super heroand write a super funny two-sentence short story about {thing}?"

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

llm_chain("chatgpt and google fight")

input_list = [
    {"thing": "people eat pizza"},
    {"thing": "a dancer "},
    {"thing": "a data scientist who donot know python "}
]

llm_chain.apply(input_list)

llm_chain.generate(input_list)

# Single input example
llm_chain.predict(thing="best python learingi course")

llm_chain.run("playing cricket")



"""# router chain"""

from langchain.chains.router import MultiPromptChain
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{input}"""


math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{input}"""

prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "prompt_template": physics_template,
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": math_template,
    },
]

destination_chains = {}

for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

default_chain = ConversationChain(llm=llm, output_key="text")

default_chain.run("What is math?")



"""# Simple sequential chain"""

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# This is an LLMChain to write a rap.
llm = OpenAI(temperature=.7)

template = """

You are a indain ipl player .

Given a topic, it is your final match.

Topic: {topic}
"""
prompt_template = PromptTemplate(input_variables=["topic"], template=template)

rap_chain = LLMChain(llm=llm, prompt=prompt_template)

# This is an LLMChain to write a diss track

llm = OpenAI(temperature=.7)

template = """

You are an extremely hitter palyer .

Given the rap from you did very  good in fineal match and topic.

match:
{match}
"""

prompt_template = PromptTemplate(input_variables=["match"], template=template)

diss_chain = LLMChain(llm=llm, prompt=prompt_template)

# This is the overall chain where we run these two chains in sequence.
from langchain.chains import SimpleSequentialChain

overall_chain = SimpleSequentialChain(chains=[rap_chain, diss_chain], verbose=True)

review = overall_chain.run("Drinking Crown Royal and mobbin in my red Challenger")



"""# sequntial chain"""

llm = OpenAI(temperature=.7)

template = """

You are a Punjabi Jatt rapper, like AP Dhillon or Sidhu Moosewala.

Given two topics, it is your job to create a rhyme of two verses and one chorus
for each topic.

Topic: {topic1} and {topic2}

Rap:

"""

prompt_template = PromptTemplate(input_variables=["topic1", "topic2"], template=template)

rap_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="rap")


template = """

You are a rap critic from the Rolling Stone magazine and Metacritic.

Given a, it is your job to write a review for that rap.

Your review style should be scathing, critical, and no holds barred.

Rap:

{rap}

Review from the Rolling Stone magazine and Metacritic critic of the above rap:

"""

prompt_template = PromptTemplate(input_variables=["rap"], template=template)

review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")

# This is the overall chain where we run these two chains in sequence.
from langchain.chains import SequentialChain

overall_chain = SequentialChain(
    chains=[rap_chain, review_chain],
    input_variables=["topic1", "topic2"],
    # Here we return multiple variables
    output_variables=["rap", "review"],
    verbose=True)

overall_chain({"topic1":"Tractors and sugar canes", "topic2": "Dasuya, Punjab"})









"""# MEMORY"""



from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

llm = OpenAI(temperature=0)
template = "Your conversation template here..."
prompt = PromptTemplate.from_template(template)
memory = ConversationBufferMemory(memory_key="chat_history")
conversation = LLMChain(llm=llm, prompt=prompt, memory=memory)

response = conversation({"question": "What's the weather like?"})
print(response)

from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI


chat = ChatOpenAI(temperature=0, openai_api_key=api_key)
history = ChatMessageHistory()

history.add_ai_message("what is  ipl ")

history.add_user_message("ipl is indian premier league")

history.messages

ai_response = chat(history.messages)
ai_response

history.add_ai_message(ai_response.content)
history.messages

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("Hello!")
memory.chat_memory.add_ai_message("How can I assist you?")
memory.chat_memory.messages



from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=1)
memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.save_context({"input": "not much you"}, {"output": "not much"})
memory.load_memory_variables({})

{'history': 'Human: not much you\nAI: not much'}



from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("hi!")
memory.chat_memory.add_ai_message("what's up?")
memory.load_memory_variables({})

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("tell my about ipl?")
memory.chat_memory.add_ai_message("indian premier league?")

memory.load_memory_variables({})

memory = ConversationBufferMemory(memory_key="chat_history")

memory.chat_memory.add_user_message("tell my about ipl")

memory.chat_memory.add_ai_message("indian premier league")

memory.load_memory_variables({})

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("wagwan, bruv?")
memory.chat_memory.add_ai_message("Alright, guv'nor? Just been 'round the old manor, innit?")

memory.load_memory_variables({})

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

#instantiate the language model
llm = OpenAI(temperature=0.1)

# Look how "chat_history" is an input variable to the prompt template
template = """

You are Spider-Punk, Hobart Brown from Earth-138.

Your manner of speaking is rebellious and infused with punk rock lingo,
often quippy and defiant against authority.

Speak with confidence, wit, and a touch of brashness, always ready
to  challenge the status quo with passion.

Your personality swings between that classic cockney sensibility
and immeasurable Black-British street swagger

Previous conversation:
{chat_history}

New human question: {question}
Response:
"""

prompt = PromptTemplate.from_template(template)

# Notice that we need to align the `memory_key`

memory = ConversationBufferMemory(memory_key="chat_history")

conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

conversation({"question":"wagwan, bruv?"})

from langchain.chat_models import ChatOpenAI

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chains import LLMChain

from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI()

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            """
            You are Spider-Punk, Hobart Brown from Earth-138.

            Your manner of speaking is rebellious and infused with punk rock lingo,
            often quippy and defiant against authority.

            Speak with confidence, wit, and a touch of brashness, always ready
            to  challenge the status quo with passion.

            Your personality swings between that classic cockney sensibility
            and immeasurable Black-British street swagger
            """
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

# Notice that we `return_messages=True` to fit into the MessagesPlaceholder
# Notice that `"chat_history"` aligns with the MessagesPlaceholder name.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

conversation.predict(question="wagwan, bruv?")



from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=2)

conversation_with_summary = ConversationChain(
    llm=OpenAI(temperature=0),
    memory=ConversationBufferWindowMemory(k=3),
    verbose=True
)

conversation_with_summary.predict(input="Wagwan, Bruv?")























