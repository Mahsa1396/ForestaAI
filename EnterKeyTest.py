# Libraries
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
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import markdown
from tenacity import retry, wait_random_exponential, stop_after_attempt
from langchain.text_splitter import CharacterTextSplitter


# Setup OpenAI
#openai.organization = config("OPENAI_ORG_ID")

GPT_MODEL = "gpt-4-1106-preview"


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


user_input = st.text_input("Enter your API key here:", "")

if user_input:
    try:
        openai.api_key = user_input
        st.write("API key set successfully")
    except Exception as e:
        st.error(f"Error setting API key: {e}")

#os.environ["OPENAI_API_KEY"] = openai.api_key

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {"model": model, "messages": messages}
    if tools is not None:
        json_data.update({"tools": tools})
    if tool_choice is not None:
        json_data.update({"tool_choice": tool_choice})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


def chat_with_openai(pdf_content, messages, openai_api_key):
    openai.api_key = openai_api_key

    response = chat_completion_request(messages,
        model="gpt-4-1106-preview"
        )

    return response




def read_pdf(file_path):
    # read text from pdf
    pdfreader = PdfReader('Foresta Docs.pdf')
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

  # We need to split the text using Character Text Split such that it should not increse token size
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 50000,
        chunk_overlap  = 10000,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    return texts

if "pdf_index" not in st.session_state:

    upload_directory = 'uploads2'

    # Create the directory if it doesn't exist
    if not os.path.exists(upload_directory):
        os.makedirs(upload_directory)
    
    
    #pdf_file = st.file_uploader("Upload a pdf file", type="pdf")
    #pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    pdf_content = read_pdf('Foresta Docs.pdf')
    messages=[{"role": "system", "content": "You are a helpful assistant knowledgeable about the following text which is the documentation about the Foresta app that is created by Decision Spot: " + pdf_content[0]}]
  
    
    
    # Initialize a counter to create unique keys
    counter = 0   
    # Define Streamlit Containers
    response_container = st.container()
    container = st.container()

# Set Streamlit Containers
    with container:
        with st.form(key=f'my_form_{counter}', clear_on_submit=True):
            user_input = st.text_area("You:", placeholder="Ask me a question!", key='input', height=100) 
            submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                if 'ai_message' in st.session_state and len(st.session_state['ai_message']) == 0:
                    with response_container:
                        message(user_input, is_user=True)
                
                messages.append({"role": "user", "content": user_input})
                response = chat_with_openai(pdf_content, messages, openai.api_key)
                output = response.json()["choices"][0]["message"]
                messages.append(output)
                #output = generate_response(messages, page_chunks)
                st.session_state['user_message'].append(user_input)
                st.session_state['ai_message'].append(output['content'])
    counter += 1
    if st.session_state['ai_message']:
        with response_container:
            if len(st.session_state['ai_message']) == 1:
                message(st.session_state["ai_message"][0])
            else:
                for i in range(len(st.session_state['ai_message'])):
                    message(st.session_state["user_message"][i], is_user=True)
                    message(st.session_state["ai_message"][i])
