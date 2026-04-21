# Bedrock-RAG-Assistant
# 📄 Chat with PDF using AWS Bedrock (RAG Application)

## 🚀 Overview
This project is a **Retrieval-Augmented Generation (RAG)** application that allows users to **chat with PDF documents** using **AWS Bedrock (LLMs + Embeddings)**.

It processes a PDF file, creates embeddings, stores them in a FAISS vector database, and retrieves relevant context to generate accurate answers using a Large Language Model.

---

## 🧠 Key Features
- 📂 Upload & process PDF documents
- 🔍 Semantic search using vector embeddings
- 🤖 LLM-powered question answering (Meta LLaMA 3 via AWS Bedrock)
- ⚡ Fast retrieval with FAISS vector store
- 🌐 Interactive UI using Streamlit
- 🔄 Update vector database dynamically

---

## 🏗️ Tech Stack
- **Frontend:** Streamlit  
- **Backend:** Python  
- **LLM:** AWS Bedrock (LLaMA 3 - 8B Instruct)  
- **Embeddings:** Amazon Titan Embeddings  
- **Vector DB:** FAISS  
- **Framework:** LangChain  

---

## 📁 Project Structure
├── app.py # Main Streamlit application
├── Attention.pdf # Input PDF file
├── faiss_index/ # Stored vector database
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## ⚙️ How It Works

1. **Data Ingestion**
   - Loads PDF using `PyPDFLoader`
   - Splits text into chunks using `RecursiveCharacterTextSplitter`

2. **Embedding Creation**
   - Converts text chunks into embeddings using Amazon Titan

3. **Vector Storage**
   - Stores embeddings in FAISS for fast similarity search

4. **Query Processing**
   - User query is embedded
   - Relevant chunks are retrieved from FAISS

5. **Answer Generation**
   - Retrieved context + query passed to LLaMA 3 model
   - Final answer generated using RAG pipeline

---

## 🛠️ Setup Instructions

### 1️⃣ Clone Repository

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

### 2️⃣ Install Dependencies
pip install -r requirements.txt

### 3️⃣ Configure AWS Credentials
#Make sure you have AWS credentials configured:
aws configure

#Or set environment variables:
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

### ▶️ Run the Application
streamlit run app.py
