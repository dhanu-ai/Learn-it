import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import os
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Read API key
gemini_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

if not gemini_api_key:
    st.error("GOOGLE_API_KEY missing. Add it to .env or Streamlit secrets.")
    st.stop()

genai.configure(api_key=gemini_api_key)

# -----------------------------------------
# Extract text from PDFs
# -----------------------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


# -----------------------------------------
# Chunk text
# -----------------------------------------
def get_text_chunks(raw_text):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_text(raw_text)


# -----------------------------------------
# Create VectorStore
# -----------------------------------------
def get_vector_store(chunks):
    if not chunks:
        return None
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=gemini_api_key
    )
    return FAISS.from_texts(chunks, embeddings)


# -----------------------------------------
# Conversation Chain
# -----------------------------------------
def build_chain():
    template = """
You are an AI educator. Explain concepts strictly using ONLY the PDF content.

Rules:
- Use bullet points, headings, diagrams (ASCII if helpful).
- Expand concepts with intuition, theory, variations and real-world use-cases.
- If answer is not in the PDF context, reply:
"This topic was not covered in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=gemini_api_key
    )

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    chain = load_qa_chain(
        llm=model,
        chain_type="stuff",
        prompt=prompt
    )

    return chain


# -----------------------------------------
# Answer user questions
# -----------------------------------------
def answer_user(question):
    db = st.session_state.vector_store
    chain = st.session_state.chain
    raw_text = st.session_state.raw_text

    if not db:
        return "Upload and process PDFs first."

    docs = db.similarity_search(question, k=5)

    response = chain.invoke({
        "input_documents": docs,
        "context": raw_text,
        "question": question
    })

    return response.get("output_text", "No answer found.")


# -----------------------------------------
# STREAMLIT APP
# -----------------------------------------
def main():
    st.set_page_config(page_title="Learn It", page_icon="📘", layout="wide")
    st.header("📘 Learn It — PDF Learning Assistant")

    # Session state defaults
    for key in ["messages", "vector_store", "chain", "raw_text"]:
        if key not in st.session_state:
            st.session_state[key] = None if key != "messages" else []

    with st.sidebar:
        st.subheader("Upload PDF files")
        pdfs = st.file_uploader("Select PDFs", type="pdf", accept_multiple_files=True)

        if st.button("Process PDFs"):
            if not pdfs:
                st.error("Upload at least one PDF.")
            else:
                with st.spinner("Processing..."):
                    raw = get_pdf_text(pdfs)
                    chunks = get_text_chunks(raw)
                    vector_store = get_vector_store(chunks)
                    chain = build_chain()

                    st.session_state.raw_text = raw
                    st.session_state.vector_store = vector_store
                    st.session_state.chain = chain

                    st.success("PDFs processed successfully!")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    if st.session_state.vector_store:
        user_input = st.chat_input("Ask a question about the PDF...")

        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            # Generate answer
            answer = answer_user(user_input)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)


if __name__ == "__main__":
    main()
