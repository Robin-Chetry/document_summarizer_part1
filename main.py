import streamlit as st
import os
from dotenv import load_dotenv

# LangChain imports
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document

# Load environment variables from .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Prompt template for retrieval-augmented generation
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.

<context>
{context}
</context>

Question: {input}
""")

def main():
    # Streamlit app config and title
    st.set_page_config(page_title="Chat with Your Document", layout="centered")
    st.title("Chat with Your Document")

    # File uploader for PDF
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'], help="Please select a PDF document")

    # Reset session state
    if st.button("Reset App"):
        st.session_state.clear()
        st.experimental_rerun()

    # Submit file
    if st.button("Submit", type="primary", use_container_width=True):
        if uploaded_file is not None:
            st.success("PDF submitted successfully. Processing...")

            # Save uploaded file locally
            with open("temp_uploaded.pdf", "wb") as f:
                f.write(uploaded_file.read())

            # Load PDF and split it into chunks
            loader = PyPDFLoader("temp_uploaded.pdf")
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            final_documents = text_splitter.split_documents(docs)

            # Generate embeddings and store in vector DB
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectordb = Chroma.from_documents(documents=final_documents, embedding=embeddings)

            # Load Groq LLM
            model = ChatGroq(model="Gemma2-9b-it", groq_api_key=groq_api_key)

            # Save objects to session state
            st.session_state.vectordb = vectordb
            st.session_state.model = model
            st.session_state.final_documents = final_documents

            # Summarize the document using map-reduce strategy
            docs_for_summary = [Document(page_content=doc.page_content) for doc in final_documents]
            summary_chain = load_summarize_chain(model, chain_type="map_reduce")

            with st.spinner("Summarizing the document..."):
                summary = summary_chain.run(docs_for_summary)

            # Save summary to session
            st.session_state.summary = summary

            st.success("Summary generated successfully.")
            with st.expander("Click to view summary"):
                st.write(summary)
        else:
            st.warning("Please upload a PDF file before submitting.")

    # Q&A section: if vector DB and model are ready
    if "vectordb" in st.session_state and "model" in st.session_state:
        user_prompt = st.text_input("Ask a question about the PDF")

        if user_prompt:
            retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 3})
            model = st.session_state.model

            # Create document QA pipeline
            document_chain = create_stuff_documents_chain(model, prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            try:
                with st.spinner("Generating answer..."):
                    response = retrieval_chain.invoke({"input": user_prompt})
                st.success("Answer:")
                st.write(response["answer"])
            except Exception as e:
                st.error("An error occurred while generating the answer.")
                st.exception(e)

# Run main app
if __name__ == "__main__":
    main()
