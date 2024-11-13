import os
import requests
from dotenv import load_dotenv
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

load_dotenv()

api_key = os.getenv("GROQCLOUD_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS(embedding_function=embeddings)

# Function to upload documents, convert text to embeddings, and store in FAISS
def upload_documents(texts):
    documents = [Document(page_content=txt) for txt in texts]
    vector_store.add_documents(documents)

# Function to retrieve relevant documents based on user query
def retrieve_relevant_docs(query):
    docs = vector_store.similarity_search(query)
    return docs

# Function to answer user query using retrieved documents and GroqCloud API
def answer_query(query):
    retrieved_docs = retrieve_relevant_docs(query)
    context = " ".join([doc.page_content for doc in retrieved_docs])

    # Preparing the request payload
    payload = {
        "model": "gpt-3.5-groq",  # Update with the actual model name if different
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {query}"}
        ]
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    #hitting groq api endpoint
    response = requests.post("https://api.groqcloud.com/v1/chat/completions", json=payload, headers=headers)
    
    #success / error catch
    if response.status_code == 200:
        response_data = response.json()
        answer = response_data['choices'][0]['message']['content']
        return answer
    else:
        return "Error: Unable to get a response from GroqCloud API."

# Streamlit UI part
st.title("Personalized Knowledge Base Assistant")
uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True)

if uploaded_files:
    texts = [file.read().decode("utf-8") for file in uploaded_files]
    upload_documents(texts)
    st.write("Documents uploaded successfully!")

user_query = st.text_input("Ask a question")
if user_query:
    answer = answer_query(user_query)
    st.write("Answer:", answer)


