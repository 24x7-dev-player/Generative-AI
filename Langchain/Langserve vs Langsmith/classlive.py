from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    decsription="A simple API Server"

)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)

model=ChatOpenAI()
prompt=ChatPromptTemplate.from_template("what is  topic is given explain that topic in poinst wise {topic}")
prompt1=ChatPromptTemplate.from_template("make story  {topic}")

add_routes(
    app,
    prompt|model,
    path="/topic"

)

add_routes(
    app,
    prompt1|model,
    path="/story"

)

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)
