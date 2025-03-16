import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)
prompt = ChatPromptTemplate.from_template(
    """
Answer the query based on the following pdf context:
<context>
{context}
</context>

"""
)
csdc = create_stuff_documents_chain(model, prompt)
st.title("📄 No repeat uploads. Just continuous conversations with your documents.")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF (Max 1MB)", type="pdf")

if uploaded_file:
    if uploaded_file.size > 1_000_000:
        st.error("File size exceeds 1MB. Please upload a smaller file.")
    else:
        with st.spinner("Processing PDF..."):
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                pdf_path = tmp_file.name

            try:
                # Load PDF from the temporary file
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=100
                )
                text_chunks = text_splitter.split_documents(documents)

                # Store in FAISS
                db = FAISS.from_documents(text_chunks, embeddings)
                retriever = db.as_retriever()
                crc = create_retrieval_chain(retriever, csdc)
                st.success("PDF successfully processed! Now ask your questions.")

                # User query input
                query = st.text_input("🔍 Ask a question from the document")
                if query:
                    res = crc.invoke({"input": query})
                    st.write(res["answer"])

            finally:
                # Clean up the temporary file
                os.unlink(pdf_path)
