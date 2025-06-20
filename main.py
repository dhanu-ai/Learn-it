import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Load environment variables
dotenv.load_dotenv()

gemini_api_key = st.secrets["GOOGLE_API_KEY"]
if gemini_api_key is None:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=gemini_api_key)

# PDF Text Extraction
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Chunking the text
def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_text(raw_text)

# Vector Embedding
def get_vector(chunks):
    if not chunks:
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=gemini_api_key)
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

# Memory-aware Question Handling
def user_question(question, db, chain, raw_text):
    if db is None:
        return "Please upload and process a PDF first."

    docs = db.similarity_search(question, k=5)

    # Memory integration: combine chat history
    history = ""
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            history += f"User: {content}\n"
        else:
            history += f"Assistant: {content}\n"

    response = chain.invoke(
        {"input_documents": docs, "question": question, "context": raw_text, "history": history},
        return_only_outputs=True
    )
    return response.get("output_text")

# Load chain with fixed prompt
def conversation_chain():
    template = """
You are a highly intelligent AI educator.
Your task is to explain or elaborate any concept strictly based on the uploaded PDF/document content, but in high detail — as if preparing study material or a technical guide.
✅ Use bullet points, structured sections, and diagrams (if context permits or image generation is integrated).
✅ Expand simple definitions into deep-dive explanations with intuition, formulas (if available), real-world analogies, and use cases.
✅ If the user asks about a concept found in the document (e.g., "k-NN"), explain it deeply — including theoretical background, working steps, advantages/disadvantages, variations, and where applicable, visual illustrations.
✅ Do not fabricate answers. If something is not in the PDF context, say:
“This topic was not covered in the uploaded document.” 
    Context: \n{context}\n
    Question: \n{question}\n
    Answer:
    """
    model_instance = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=gemini_api_key)
    prompt = PromptTemplate(template=template, input_variables=["context", "question", "history"])
    return load_qa_chain(model_instance, chain_type="stuff", prompt=prompt), model_instance

# Main Streamlit App
def main():
    st.set_page_config(page_title="Learn It", page_icon="👨‍🏫", layout="wide")
    st.header("Learn It - Your Personal Learning Assistant 👨‍🏫")

    # Session state setup
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = None

    # Sidebar for PDF upload
    with st.sidebar:
        st.subheader("Upload the notes or PDF files for analysis")
        pdf_docs = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")

        if st.button("Process PDF"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    vector_store = get_vector(chunks)
                    chain, _ = conversation_chain()

                    if vector_store and chain and raw_text:
                        st.session_state.vector_store = vector_store
                        st.session_state.chain = chain
                        st.session_state.raw_text = raw_text
                        st.success("PDF processed successfully.")

    # Display past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input section
    if st.session_state.vector_store and st.session_state.chain and st.session_state.raw_text:
        user_query = st.chat_input("Ask your question:")
        if user_query:
            # User message
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.messages.append({"role": "user", "content": user_query})

            # Assistant response
            response = user_question(
                user_query,
                st.session_state.vector_store,
                st.session_state.chain,
                st.session_state.raw_text
            )
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
