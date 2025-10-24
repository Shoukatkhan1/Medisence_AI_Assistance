# 📚 Import all the needed libraries
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# 🌍 Load secret keys (from .env file)
load_dotenv()

# 🗝️ Get the Pinecone API key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# 1️⃣ Load and split PDFs
# This will read all PDF files from the "Data/" folder
# Then it will break the text into small parts (chunks)
extracted_data = load_pdf_file("Data/")
text_chunks = text_split(extracted_data)

# 2️⃣ Download embeddings
# Embeddings are special number lists that help the computer understand text meaning
embeddings = download_hugging_face_embeddings()

# 3️⃣ Create a Pinecone index (like a smart search box)
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot"

# Check if the index already exists, if not, make a new one
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,          # Name of your index
        dimension=384,            # Size of embedding vector
        metric="cosine",          # How to measure similarity between texts
        spec=ServerlessSpec(      # Run it on the cloud (AWS)
            cloud="aws", 
            region="us-east-1"
        ),
    )

# 4️⃣ Upload data to Pinecone
# This step sends all the small text chunks with their embeddings to Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

# ✅ Print message when done
print("✅ Data ingestion complete and uploaded to Pinecone.")
