# Chat with Your Document

This project is a simple Streamlit web application that allows users to upload a PDF, get a summary of the document, and ask questions based on its content using a language model.

## Features

- Upload PDF files
- Automatically generate a summary of the document
- Ask questions related to the document
- Uses vector search to retrieve relevant content
- Powered by Groq's Gemma language model and Hugging Face embeddings

## Technologies Used

- Streamlit – for the user interface
- LangChain – for LLM orchestration
- Chroma – for vector storage and retrieval
- Hugging Face – for text embeddings
- Groq – for running the LLM (Gemma2-9b-it)
- PyPDFLoader – for reading PDF files

