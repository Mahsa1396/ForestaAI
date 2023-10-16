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
import requests
import PyPDF2
from bs4 import BeautifulSoup
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# Setup OpenAI
#openai.organization = config("OPENAI_ORG_ID")



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


openai.api_key = user_input 

# Saving the cleaned text and append it to the pdf_merger
def savePDF(cleaned_text, pdf_merger, arg):

    # Create a PDF file
    if arg == 'liquid':
        pdf_file = "liquid.pdf"
    else:
        pdf_file = "flex.pdf"
    c = canvas.Canvas(pdf_file, pagesize=letter)

    # Set font and font size
    c.setFont("Helvetica", 10)

    # Set the position (x, y) to start adding text
    x, y = 50, 700
    
    # Your long text content, split into paragraphs
    text = cleaned_text

    # Split the text into paragraphs
    paragraphs = text.split("\n")

    # Function to add a new page and reset the position
    def new_page():
        c.showPage()
        c.setFont("Helvetica", 10)
        return 50, 700  # Reset the starting position

    # Function to calculate the width of a text string
    def text_width(text):
        return c.stringWidth(text)

    
    # Loop through paragraphs and add them to the PDF
    for paragraph in paragraphs:
        if y < 50:  # Check if there is not enough space on the current page
            x, y = new_page()  # Add a new page and reset the position

        available_width = 500 - x  # Adjust for your page layout
        while text_width(paragraph) > available_width:
            # The paragraph is too wide for the page, split it
            split_index = int(len(paragraph) * available_width // text_width(paragraph))
            line, paragraph = paragraph[:split_index], paragraph[split_index:]
            c.drawString(x, y, line)
            y -= 15  # Adjust the vertical position for the next line
            x, y = new_page() if y < 50 else (x, y)  # Check if a new page is needed

        c.drawString(x, y, paragraph)
        y -= 15  # Adjust the vertical position for the next line

    # Save the PDF file
    c.save()
    

    st.write(f"PDF file '{pdf_file}' created successfully.")
    if pdf_file == "liquid.pdf":
        pdf_merger.append("liquid.pdf")
    else:
        pdf_merger.append("flex.pdf")
    st.write("pdf_merger got updated successfully!")

    return pdf_merger
    
    

# Scraping liquid carton packaging market
def liquid(pdf_merger):
    # URL of the website to scrape
    url = "https://www.futuremarketinsights.com/reports/liquid-carton-packaging-market"

    # Send an HTTP GET request to the URL
    response = requests.get(url)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Find and remove all <div> elements with class "r-Banner-one-child mr-5 pr-4"
        for unwanted_div in soup.find_all("div", class_="r-Banner-one-child mr-5 pr-4"):
          unwanted_div.extract()

        # Find and remove all <div> elements with class "QuestionBox text-black mr-2"
        for unwanted_div in soup.find_all("div", class_="QuestionBox text-black mr-2"):
          unwanted_div.extract()

        for unwanted_div in soup.find_all("div", class_="r-Banner-two reqMethodBox d-flex align-items-center my-4"):
          unwanted_div.extract()
        # Find all the <div> elements with class "reportContent"
        report_content_divs = soup.find_all("div", class_="reportContent")

        # Initialize a variable to store the cleaned text
        cleaned_text = ""

        # Loop through the found elements, extract the text, and clean it
        for div in report_content_divs:
            text = div.get_text()
            cleaned_text += text.strip() + "\n"  # Remove leading/trailing whitespace and add a newline

        # Remove extra blank lines
        cleaned_text = "\n".join(line for line in cleaned_text.splitlines() if line.strip())
         
        st.write("Done Scraping!")
        # Save the cleaned text to a PDF file
        pdf_merger = savePDF(cleaned_text, pdf_merger, "liquid")
    else:
      st.write(f"Failed to retrieve content from {url}. Status code: {response.status_code}")

    
    
    return pdf_merger


def flexible(pdf_merger):
    # URL of the website to scrape
    url = "https://www.arizton.com/market-reports/europe-flexible-packaging-market"

    # Send an HTTP GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        report_content_divs = soup.find_all("div", id="marketOverview")

        # Initialize a variable to store the cleaned text
        cleaned_text = ""

        # Loop through the found elements, extract the text, and clean it
        for div in report_content_divs:
            text = div.get_text()
            cleaned_text += text.strip() + "\n"  # Remove leading/trailing whitespace and add a newline

        # Remove extra blank lines
        cleaned_text = "\n".join(line for line in cleaned_text.splitlines() if line.strip())
        
        # Print or save the cleaned text as needed
        pdf_merger = savePDF(cleaned_text, pdf_merger, "flex")
    else:
      st.write(f"Failed to retrieve content from {url}. Status code: {response.status_code}")
    
    return pdf_merger

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

    # Get Previous messages and append context
    #to_send = st.session_state['messages'].copy()
    #to_send.insert(-1, system_source_help)

    #st.session_state['messages'].append({"role": "user", "content": prompt})
    #completion = openai.ChatCompletion.create(
    #    model="gpt-3.5-turbo",
    #    temperature=0,
    #    messages=to_send,
    #)
    #response = completion.choices[0].message.content
    #st.session_state['messages'].append({"role": "assistant", "content": response})
    
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai.api_key)
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=get_relevant_sources, question=prompt)
        print(cb)
    #st.write(response)
    
    return response

if "pdf_index" not in st.session_state:

    upload_directory = 'uploads2'

    # Create the directory if it doesn't exist
    if not os.path.exists(upload_directory):
        os.makedirs(upload_directory)

    #pdf_file = st.file_uploader("Upload a pdf file", type="pdf")
    pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    # Prompt the user to select multiple options
    selected_options = st.multiselect("Select one or more options:", ["Liquid carton packaging market", "Flexible packaging global market"])

    # Display the selected options
    st.write(f"You selected: {selected_options}")

    
    if (pdf_files is not None and len(pdf_files) > 0) or (len(selected_options) > 0):
        pdf_merger = PyPDF2.PdfMerger()
        for pdf_file in pdf_files:

            pdf_merger.append(pdf_file)

        # Check the selected options and call the corresponding functions
        if "Liquid carton packaging market" in selected_options:
            pdf_merger = liquid(pdf_merger)
        if "Flexible packaging global market" in selected_options:
            pdf_merger = flexible(pdf_merger)
        
        merged_pdf_filename = 'merged.pdf'

        # Define the path to save the merged PDF
        pdf_file_path = os.path.join(upload_directory, merged_pdf_filename)

        # Save the merged PDF to the specified directory
        with open(pdf_file_path, 'wb') as f:
            pdf_merger.write(f)


        st.success(f"PDF files '{merged_pdf_filename}' saved successfully.")
        
        loader = PyPDFLoader(pdf_file_path)
        pages = loader.load_and_split()

        # Split the pages into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=12000, chunk_overlap=1000)
        page_chunks = text_splitter.split_documents(pages)

        # Embed into FAISS
        vectorstore = FAISS.from_documents(page_chunks, OpenAIEmbeddings(openai_api_key=openai.api_key))

            
    # Define Streamlit Containers
        response_container = st.container()
        container = st.container()

    # Set Streamlit Containers
        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_area("You:", placeholder="Ask me a question!", key='input', height=100) 
                submit_button = st.form_submit_button(label='Send')

                if submit_button and user_input:
                    if 'ai_message' in st.session_state and len(st.session_state['ai_message']) == 0:
                        with response_container:
                            message(user_input, is_user=True)
                    output = generate_response(user_input, page_chunks)
                    st.session_state['user_message'].append(user_input)
                    st.session_state['ai_message'].append(output)

        if st.session_state['ai_message']:
            with response_container:
                if len(st.session_state['ai_message']) == 1:
                    message(st.session_state["ai_message"][0])
                else:
                    for i in range(len(st.session_state['ai_message'])):
                        message(st.session_state["user_message"][i], is_user=True)
                        message(st.session_state["ai_message"][i])