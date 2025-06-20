# 📚 Learn It - AI-Powered PDF Learning Assistant

**Learn It** is a Streamlit-based chatbot that transforms your PDF documents into interactive learning sessions. Powered by **Google's Gemini (via LangChain)** and **FAISS vector search**, this app helps you explore any concept from your uploaded documents — with detailed, structured, and high-quality explanations.

---

## 🚀 Features

- 📄 **Multi-PDF Upload**: Upload one or more PDF files for instant document ingestion.
- ✂️ **Chunk-Based Processing**: Splits text into overlapping chunks to preserve context.
- 🔍 **Semantic Search with FAISS**: Finds the most relevant sections of your documents for each question.
- 🧠 **Gemini-Powered QA**: Uses Gemini 1.5 Flash via LangChain to generate study-level answers.
- 📜 **Memory-Aware Chat**: Maintains conversational history to answer follow-ups coherently.
- 🛑 **Grounded Responses**: If the answer isn't in the document, the bot clearly says so — no hallucinations.

---

## 📦 Tech Stack

| Technology       | Purpose                              |
|------------------|--------------------------------------|
| Streamlit        | Web app UI                           |
| LangChain        | Vector search + Prompt chaining      |
| Google Gemini    | LLM backend for question answering   |
| FAISS            | Local vectorstore for text chunks    |
| PyPDF2           | Extract text from PDFs               |
| dotenv           | Environment variable management      |

---

## 🖼️ Example Use-Cases

- Study from handwritten class notes in PDF
- Analyze technical research papers or documentation
- Create study guides and summaries on-the-fly
- Dig deep into any concept — only from your uploaded material

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/learn-it-pdf-qa.git
cd learn-it-pdf-qa
