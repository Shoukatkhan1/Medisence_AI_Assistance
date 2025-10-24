<div align="center">

# 💊 MediSense AI — Intelligent Medical Assistant  

### 🩺 A Smart RAG-based AI Chatbot for Kidney-Related Medical Guidance

Built with **LangChain**, **Groq**, **Pinecone**, and **Streamlit**, MediSense AI transforms complex medical literature into interactive, conversational intelligence — empowering healthcare professionals and learners with fast, evidence-based answers.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG_Framework-green)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 🧩 Project Overview

**MediSense AI** is a medical domain chatbot powered by **Retrieval-Augmented Generation (RAG)**.  
It allows users to upload and query medical literature (e.g., kidney disease PDFs) and get context-based responses — not generic AI replies.

🩻 **Use Case**: Medical professionals, students, and researchers can quickly retrieve accurate insights from their PDFs or knowledge bases.

---

## 🚀 Features

✅ PDF-based medical document ingestion  
✅ Advanced text chunking & semantic search  
✅ Hugging Face embeddings integration  
✅ Pinecone vector store for similarity retrieval  
✅ Groq API for LLM inference (ultra-fast responses)  
✅ Streamlit-based interactive UI  
✅ Secure key management via `.env`  
✅ Custom system prompt for **MBBS-level medical expertise**

---

## 🏗️ System Architecture

```mermaid
graph TD
A[PDF Documents] -->|Extract & Split| B[LangChain Text Splitter]
B -->|Embed| C[HuggingFace Embeddings]
C -->|Store| D[Pinecone Vector DB]
E[User Query] -->|RAG Pipeline| F[Groq LLM via LangChain]
D -->|Retrieve Context| F
F -->|Answer| G[Streamlit Chat UI]
