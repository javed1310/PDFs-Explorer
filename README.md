# 📚 PDFs Explorer: Chat with Your Documents
This project is a powerful and intuitive web application that allows you to upload multiple PDF documents and engage in a conversation about their contents. It leverages a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-aware answers based on the information within your files.

The application is built with Python, using Streamlit for the interactive user interface and LangChain to orchestrate the workflow with the Groq API for fast LLM inference.

(Feel free to replace this with a screenshot of your application!)

# ✨ Features
Multi-PDF Upload: Upload and process several PDF documents at once.

Custom Chat Interface: A clean and user-friendly chat interface with distinct styles for user and bot messages.

Conversational Memory: The chatbot remembers previous turns in the conversation, allowing for follow-up questions.

Fast & Powerful LLM: Utilizes the Groq API for high-speed language model inference.

Efficient Text Processing: Splits documents into manageable chunks and creates a searchable vector database using FAISS.

Easy to Set Up: A requirements.txt file is included for a straightforward setup process.

# 🚀 How It Works
The application follows a Retrieval-Augmented Generation (RAG) workflow:

PDF Processing: When you upload your PDF files, the application extracts all the text content.

Text Chunking: The extracted text is split into smaller, overlapping chunks to ensure semantic context is preserved.

Vector Embeddings: Each text chunk is converted into a numerical vector representation using Hugging Face sentence transformers.

Vector Store: These vectors are stored in a FAISS vector database, which allows for incredibly fast and efficient similarity searches.

Conversational Chain: When you ask a question, the application searches the vector store for the most relevant text chunks. These chunks, along with your question and the chat history, are sent to the Groq LLM.

Answer Generation: The LLM generates a coherent, human-like answer based on the provided context, which is then displayed in the chat interface.

# 🛠️ Installation & Setup
Follow these steps to set up and run the project on your local machine.

Prerequisites
Python 3.8 or higher

pip package manager

1. Clone the Repository
git clone [https://github.com/javed1310/PDFs-Explorer.git](https://github.com/javed1310/PDFs-Explorer.git)
cd your-repository-name

2. Create a Virtual Environment (Recommended)
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
Install all the required Python libraries using the requirements.txt file.

pip install -r requirements.txt

4. Set Up Environment Variables
Create a file named .env in the root directory of the project and add your Groq API key:

GROQ_API_KEY="your_groq_api_key_here"

# ▶️ How to Run the Application
Once the setup is complete, you can start the Streamlit web application.

Run the app.py file using Streamlit:

streamlit run app.py

Your web browser should automatically open a new tab with the application running. Upload your PDFs in the sidebar, click "Process", and start asking questions!

# 📂 File Structure
app.py: The main Python script that runs the Streamlit web application and contains the RAG logic.

htmlTemplates.py: A helper file containing CSS and HTML templates for styling the chat interface.

requirements.txt: A list of all Python dependencies required to run the project.

.env: A file to securely store your API keys.

# 🔧 Technologies Used
Frameworks: Streamlit, LangChain

LLM Provider: Groq

Embeddings: Hugging Face Sentence Transformers

Vector Database: FAISS

PDF Processing: PyPDF2
