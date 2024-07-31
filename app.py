
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from html_template import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator= "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len)
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorstore(text_chunks):
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error initializing vectorstore: {e}")
        return None

def get_conversation_chain(vector_store):
    try:
        llm = ChatOpenAI()
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
        st.write("Conversation chain initialized.")
        return conversation_chain
    except Exception as e:
        st.error(f"Error initializing conversation chain: {e}")
        return None

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html= True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html= True)
    return 



def main():
    load_dotenv()
    st.set_page_config(page_title= "YourGPT", page_icon= ":robot_face:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


    st.header("YourGPT :robot_face:")
    user_question = st.text_input("Chat with YourGPT about your docs:")

    if user_question:
        handle_userinput(user_question)
    
    st.write(user_template.replace("{{MSG}}", "Hey there!"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hey how may I assist you ?"), unsafe_allow_html=True)


    with st.sidebar:
        st.subheader("Drop your documents here:")
        pdf_docs = st.file_uploader(label= "",accept_multiple_files=True)
        
        if st.button("Done"):
            with st.spinner("Reading your files"):
                

        # Get pdfs
                raw_text = get_pdf_text(pdf_docs)

        # Get chunks
                text_chunks = get_text_chunks(raw_text)

        # Store in vector db
                vector_store = get_vectorstore(text_chunks)

        # Create a conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)




if __name__ == '__main__':
    main()