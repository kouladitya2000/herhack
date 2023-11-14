# main.py
import streamlit as st
from helper import get_pdf_text, get_text_chunks, get_vector_store, create_vector_db, get_conversational_chain, user_input,get_qa_chain

# Set page config outside of the function
st.set_page_config("General Q&A")

# Page 1: Chat with PDFs
def main_pdf():
    st.header("Chat with Confluence 📄") 

    user_question = st.text_input("Input here")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.subheader("Upload your Documents") 
        pdf_docs = st.file_uploader("", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("Done")

# Page 2:  Q&A
def main_qa():
    st.header("General Q&A")


    create_vector_db()

    question = st.text_input("Input here ")
    if question:
        chain = get_qa_chain()
        response = chain(question)
        st.header("Answer")
        st.write(response["result"])

# Run the selected page
selected_page = st.sidebar.radio("Select a Page", ["Chat with Confluence", "General Q&A"])

if selected_page == "Chat with Confluence":
    main_pdf()
elif selected_page == "General Q&A":
    main_qa()
