import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="my_chroma_db", 
    collection_name="sample"            
)

# Notice the fix: "embeddings" instead of "embedings"
# data = vector_store.get(include=["embeddings", "documents", "metadatas"])
# print(data)

# print(vector_store.similarity_search(
#     query = "who among these are bowler",
#     k=2
# ))

# print(vector_store.similarity_search_with_score(
#     query = "who among these are bowler",
#     k=2
# ))

print(
    vector_store.similarity_search_with_score(
        query="Tell me about the captain",
        filter={"team": "Chennai Super Kings"},
    )
)