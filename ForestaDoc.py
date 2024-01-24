import streamlit as st
from streamlit_chat import message
import openai
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os
import PyPDF2
import requests
import PyPDF2
from bs4 import BeautifulSoup
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import markdown
import re

INDEX = "uploads2"

# Initialise streamlit session state variables
st.session_state.setdefault('messages', [
    {"role": "system", "content": "You are an AI model trained to answer questions based on the provided documentation"}
])
st.session_state.setdefault('ai_message', [])
st.session_state.setdefault('user_message', [])

# Setting page title and header
st.set_page_config(page_title="ChatWith")
st.markdown(f"<h1 style='text-align: center;'>ForestaGPT</h1>", unsafe_allow_html=True)

# Create a text input box
user_input = st.text_input("Enter your API key here:", "")

# Function for interacting with ChatGPT API
def generate_response(prompt, page_chunks):
    vectorstore = FAISS.from_documents(page_chunks, OpenAIEmbeddings(openai_api_key=openai.api_key))
    get_relevant_sources = vectorstore.similarity_search(prompt, k=3)

    template = f"\n\nYou are an AI model trained to answer questions only based on the provided documentation.\n\n{get_relevant_sources[0].page_content}\n\n{get_relevant_sources[1].page_content}"

    with st.expander("Source 1", expanded=False):
        st.write(get_relevant_sources[0].page_content)
    with st.expander("Source 2", expanded=False):
        st.write(get_relevant_sources[1].page_content)

    system_source_help = {"role": "system", "content": template}

    st.session_state['messages'].append({"role": "user", "content": prompt})
    
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai.api_key)
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=get_relevant_sources, question=prompt)
        print(cb)


    return response

if "pdf_index" not in st.session_state:

    upload_directory = 'uploads2'

    # Create the directory if it doesn't exist
    if not os.path.exists(upload_directory):
        os.makedirs(upload_directory)


    pdf_file_path = 'document.pdf'
    loader = PyPDFLoader(pdf_file_path)
    pages = loader.load_and_split()

        # Split the pages into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=12000, chunk_overlap=1000)
    page_chunks = text_splitter.split_documents(pages)

        # Embed into FAISS
    vectorstore = FAISS.from_documents(page_chunks, OpenAIEmbeddings(openai_api_key=openai.api_key))

        # Initialize a counter to create unique keys
    counter = 0    
    # Define Streamlit Containers
    response_container = st.container()
    container = st.container()
        
    # Set Streamlit Containers
    with container:
        with st.form(key=f'my_form_{counter}', clear_on_submit=True):
            user_input = st.text_area("You:", placeholder="Ask me a question!", key=f'input_{counter}', height=100) 
            submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                if 'ai_message' in st.session_state and len(st.session_state['ai_message']) == 0:
                    with response_container:
                        message(user_input, is_user=True)
                output = generate_response(user_input, page_chunks)
                st.session_state['user_message'].append(user_input)
                st.session_state['ai_message'].append(output)
    counter += 1
    if st.session_state['ai_message']:
        with response_container:
            if len(st.session_state['ai_message']) == 1:
                message(st.session_state["ai_message"][0])
            else:
                for i in range(len(st.session_state['ai_message'])):
                    message(st.session_state["user_message"][i], is_user=True)
                    message(st.session_state["ai_message"][i])
                    
