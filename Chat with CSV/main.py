from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st

# Ensure page config is set first
st.set_page_config(page_title="Ask your CSV")

def main():
    load_dotenv()

    # Load the OpenAI API key from the environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None or api_key == "":
        st.error("OPENAI_API_KEY is not set or is empty")
        return
    else:
        st.success("OPENAI_API_KEY is set")

    st.header("Ask your CSV 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        agent = create_csv_agent(
            OpenAI(api_key=api_key, temperature=0), csv_file, verbose=True
        )

        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write(agent.run(user_question))

if __name__ == "__main__":
    main()
