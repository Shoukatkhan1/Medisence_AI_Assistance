import streamlit as st
from dotenv import load_dotenv
import os
from operator import itemgetter
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from prompt import system_prompt
import time

# ===========================================================
# Load environment variables
# ===========================================================
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ===========================================================
# Load embeddings and connect to Pinecone
# ===========================================================
embeddings = download_hugging_face_embeddings()
index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# ===========================================================
# Streamlit UI Setup
# ===========================================================
st.set_page_config(page_title="Medisense AI", layout="wide", page_icon="🩺")
st.markdown("<h1 class='main-title'>💊 MediSense AI Assistant(kidney related) </h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Your intelligent medical knowledge partner — powered by LangChain, Groq & Pinecone.</p>", unsafe_allow_html=True)

st.sidebar.header("ℹ️ About")
st.sidebar.write("""
**MediSense AI** helps healthcare professionals and learners  
interact with medical literature instantly.  

**Built With:**  
- 🧠 LangChain for RAG  
- ⚙️ Pinecone for vector search  
- 🚀 Groq for LLM inference  
- 💻 Streamlit for interactive UI
""")
st.sidebar.markdown("---")
st.sidebar.caption("Created by **Shoukat Khan**, AI Developer.")

st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose LLM Model:", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])
temperature = st.sidebar.slider("Response Temperature:", 0.0, 1.0, 0.3, 0.05)

# ===========================================================
# Initialize messages state (automatically empty on new session)
# ===========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ===========================================================
# Clear chat history button
# ===========================================================
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.success("Chat history cleared!")

# ===========================================================
# Build LangChain pipeline
# ===========================================================
llm = ChatGroq(
    model=model_choice,
    temperature=temperature,
    api_key=os.getenv("GROQ_API_KEY")
)

prompt_str = f"""
{system_prompt}

Context:
{{context}}

Chat history:
{{chat_history}}

Question:
{{question}}
"""
prompt = ChatPromptTemplate.from_template(prompt_str)

# Academy-style fetchers
question_fetcher = itemgetter("question")
history_fetcher = itemgetter("chat_history")
setup = {"question": question_fetcher, "chat_history": history_fetcher, "context": itemgetter("context")}

chain = setup | prompt | llm | StrOutputParser()

# ===========================================================
# User input & chat interaction with session memory
# ===========================================================
user_input = st.chat_input("Type your medical question here... (e.g., ask about kidney-related diseases)")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Retrieve context from Pinecone
    context_docs = retriever.get_relevant_documents(user_input)
    context_text = "\n\n".join([d.page_content for d in context_docs]) if context_docs else None

    # Prepare full chat history for session memory
    chat_hist_text = ""
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        chat_hist_text += f"{role}: {msg['content']}\n"

    # Typing animation
    placeholder = st.empty()
    for i in range(3):
        placeholder.markdown(f"<p style='color:#311b92;'>Assistant is typing{'.'*(i+1)}</p>", unsafe_allow_html=True)
        time.sleep(0.5)

    # Generate AI response
    try:
        if context_text:
            # Answer based on both context (medical knowledge) and session memory
            answer = chain.invoke({
                "question": user_input,
                "chat_history": chat_hist_text,
                "context": context_text
            })
        else:
            # If no context is found in database
            answer = "I do not know. Please ask something from the medical knowledge base."
    except Exception as e:
        answer = f"❌ Error: {str(e)}"

    placeholder.empty()
    st.session_state.messages.append({"role": "assistant", "content": answer})

# ===========================================================
# Display chat messages
# ===========================================================
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    if role == "user":
        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, #00c6ff, #0072ff);
            color: white;
            padding: 12px 20px;
            border-radius: 25px;
            margin: 8px 0;
            width: fit-content;
            max-width:70%;
        ">{content}</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, #fbc2eb, #a6c1ee);
            color: #311b92;
            padding: 12px 20px;
            border-radius: 25px;
            margin: 8px 0;
            width: fit-content;
            max-width:70%;
            margin-left:auto;
        ">{content}</div>
        """, unsafe_allow_html=True)

# ===========================================================
# Footer
# ===========================================================
st.markdown("""
<div style="
    text-align: center;
    font-size: 0.9rem;
    color: #555;
    margin-top: 30px;
    padding-top: 10px;
    border-top: 1px solid #ddd;
">
Developed by <strong>SHOUKAT KHAN</strong> • © 2025
</div>
""", unsafe_allow_html=True)
