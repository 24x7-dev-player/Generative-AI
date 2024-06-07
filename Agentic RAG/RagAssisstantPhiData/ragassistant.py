from phi.assistant import Assistant
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2
import os
os.environ['OPENAI_API_KEY']='nananana'

knowledge_base = PDFUrlKnowledgeBase(
    # Read PDFs from URLs
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    # Store embeddings in the `ai.recipes` table
    vector_db=PgVector2(      ####PgVector2 is a class used to interact with a PostgreSQL database that supports vector operations. It allows storing, retrieving, and querying embeddings efficiently.
        collection="recipes",      ####collection: This specifies the name of the table or collection within the database where the embeddings will be stored. In this case, it is "recipes".
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    ),
)
# Load the knowledge base
knowledge_base.load(recreate=False)

assistant = Assistant(
    knowledge_base=knowledge_base,
    # The add_references_to_prompt will update the prompt with references from the knowledge base.
    add_references_to_prompt=True,
)
assistant.print_response("What are the ingredients to make Tom Yum Goong?", markdown=True)

