pip install pymysql

"""### setting up the openai api key in the os envoirment"""

import os
os.environ["OPENAI_API_KEY"] = "sk-xxxxxx"
import openai
openai.api_key = "sk-xxxx"

"""### Importing all the neccessary lib"""

import os
from langchain.agents import *
from langchain.llms import OpenAI
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentExecutor

"""### connect to your database"""

db_user = "root"
db_password = "12345"
db_host = "localhost"
db_name = "ahi_database"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

"""### set up the LLm,toolkit and agen executer"""

# initilizing the llm model
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name = "gpt-3.5-turbo")

toolkit = SQLDatabaseToolkit(db=db,llm=llm)

agen_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

"""## Lets ask the question"""

agen_executor.run("How many tables do we have ?")

agen_executor.run("How many rows do we have in cattle table ?")

agen_executor.run("How mnay animals in cattle table where cvolor animal color is black ")

agen_executor.run("give me chart of visualization of the cattle table")

agen_executor.run("what is the average price of Hereford Breed?")

