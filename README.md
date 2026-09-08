# PDF Question Answering System (Gemini RAG)

A full-stack RAG web application built with Streamlit, LangChain, and Google Gemini. Allows you to upload any PDF dynamically, generate embeddings on the fly, and ask questions with cited page numbers.

## Project Structure

```
rag_pdf_project/
├── app.py                 # Streamlit UI & Chat workflow
├── requirements.txt       # Python project dependencies
├── .env.example           # Environment template for credentials
├── core/
│   ├── __init__.py
│   ├── config.py          # Centralized configuration & environment loader
│   └── rag_engine.py      # PDF loader, Chroma vector store & chain logic
└── uploads/               # Scratch directory (if persistent files are needed)
```

## Quickstart Guide

### 1. Clone or Navigate to Project
```bash
cd rag_pdf_project
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Set up `.env`
Create a `.env` file from `.env.example`:
```bash
cp .env.example .env
```
Add your key inside `.env`:
```
GOOGLE_API_KEY=AIzaSy...
```
*(You can also paste the API key directly into the UI sidebar.)*

### 5. Run the Application
```bash
streamlit run app.py
```
