from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please resposne to the user request only based on the given context"),
        ("user","Question:{question}\nContext:{context}")


    ]


)
model=ChatOpenAI(model="gpt-3.5-turbo")
output_parser=StrOutputParser()

chain=prompt|model|output_parser

question="summarize that article about waterlogging"
context="""

Kempegowda International Airport (KIA) officials said that eight flights were diverted, seven to Chennai and one to Coimbatore since they were unable to land between 5 pm and 5.15 pm. According to IMD, the KIA weather station recorded 3.9 mm rainfall up to 5.30 pm.

Bengaluru Traffic Commissioner M N Anucheth said that heavy water logging was reported at 33 locations and trees were uprooted in 16 locations in Bengaluru. This led to considerable traffic congestion in several places.

Waterlogging was reported in Electronic City, Bellandur, Nagawara, Kamakshipalya, Maharani underpass and Hebbal, among other areas. Meanwhile, trees were uprooted in Jayamahal Road, Kathriguppe, PES College, Hosakerehalli, Hennur Main Road, Malleswaram and Mekhri Circle. At Hennur Main Road, an electric pole collapsed leading to traffic disruption and the underpass connecting towards Sankey Road near Kalpana Junction was closed due to waterlogging.
 this is created by prince katiyar make in karnataka
"""


print(chain.invoke({"question":question,"context":context}))
