import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_groq import ChatGroq

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
    if not text.strip():
        st.warning("No text could be extracted from the uploaded PDFs.")
    return text

def get_text_chunks(text):
    if not text.strip():
        return []
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    if not text_chunks:
        st.error("Cannot create vector store: No text chunks provided.")
        return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def get_conversation_chain(vectorstore, model_choice):
    if vectorstore is None:
        return None

    model_options = {
        "Llama 3.1 8B (Fast)": "llama-3.1-8b-instant",
        "Llama 3.1 70B (Powerful)": "llama-3.1-70b-versatile",
        "Mixtral 8x7B (Balanced)": "mixtral-8x7b-32768",
        "Gemma 2 9B (New)": "gemma2-9b-it"
    }
    
    model_name = model_options.get(model_choice)
    if not model_name:
        st.error(f"Invalid model choice: {model_choice}")
        return None
    
    try:
        llm = ChatGroq(
            temperature=0.7,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name
        )
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
            memory=memory,
        )
    except Exception as e:
        st.error(f"Failed to create conversation chain with Groq: {str(e)}")
        return None

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please process documents first.")
        return
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

def process_user_input():
    user_question = st.session_state.user_question
    if user_question:
        handle_userinput(user_question)
       
def main():
    load_dotenv()
    
    st.set_page_config(page_title="Pdfs Explorer", page_icon="📚")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""

    with st.sidebar:
        st.sidebar.title("⚙️ Setup & Configuration")
        st.sidebar.markdown("**Step 1: Choose Your Model**")
        

        model_choice = st.selectbox(
            "Select a Groq Model",
            ("Llama 3.1 8B (Fast)", "Llama 3.1 70B (Powerful)", "Mixtral 8x7B (Balanced)", "Gemma 2 9B (New)"),
            index=0,
            label_visibility="collapsed"
        )
        
        st.sidebar.divider()

        st.sidebar.markdown("**Step 2: Upload Your Documents**")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click 'Process'", 
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    if vectorstore:
                        st.session_state.conversation = get_conversation_chain(vectorstore, model_choice)
                        st.session_state.chat_history = []
                        st.toast('Documents processed successfully!', icon='🎉')
            else:
                st.warning("Please upload at least one PDF document.")

    st.title("📚 Pdfs Explorer: Chat with Your PDFs")
    st.markdown("Welcome! Upload your PDF documents on the left, click 'Process', and start asking questions.")
    
    # Display chat history
    if "chat_history" in st.session_state:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    # Handle user input using the on_change callback
    if st.session_state.conversation is not None:
        st.text_input(
            "Ask a question about your documents:", 
            key="user_question",
            on_change=process_user_input
        )
    else:
        st.info("👋 Get started by uploading your documents and clicking 'Process' in the sidebar!")
        
if __name__ == '__main__':
    main()
