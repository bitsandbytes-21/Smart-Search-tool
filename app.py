import streamlit as st 
from streamlit_chat import message
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

DB_FAISS_PATH = 'vectorstore/db_faiss'
CSV_FILE_PATH = 'courses.csv'  

def load_llm():
    llm = ChatGoogleGenerativeAI(
        model = "gemini-1.5-pro",
        temperature = 0.5
    )
    return llm


st.title("Search Vidhya 🦜")
st.markdown("<h3 style='text-align: center; color: white;'></h3>", unsafe_allow_html=True)

# Load the CSV file
loader = CSVLoader(file_path=CSV_FILE_PATH, encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()

# Prepare the embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001",
                                       model_kwargs={'device': 'cpu'})

db = FAISS.from_documents(data, embeddings)
db.save_local(DB_FAISS_PATH)

llm = load_llm()
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]


if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey! Ask me anything?👋"]

response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Search here (:", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        output = conversational_chat(user_input)
        
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)


if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
