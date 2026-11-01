import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="my_chroma_db", 
    collection_name="sample"            
)

new_doc = Document(
    page_content="""
**Jasprit Bumrah** is widely regarded as one of the greatest and most feared fast bowlers in modern cricket history. Known for his unorthodox, hyper-extended bowling action and extreme accuracy, he has redefined pace bowling across all three formats of the game.

### 1. Unique Style and Mechanics
Unlike traditional fast bowlers who rely on a textbook, high-arm classic run-up, Bumrah has a short, stuttering run-up and a slinging arm action. This distinctive delivery makes it notoriously difficult for batters to pick up the ball early out of his hand. Combined with his pinpoint control, he is a master of bowling lethal yorkers, sharp bouncers, and clever slower balls—especially during the intense "death overs" of a limited-overs match.

### 2. Rise through the IPL
Bumrah’s talent was first spotted by the Mumbai Indians (MI) coaching staff, leading to his IPL debut in 2013. Over the years, he evolved into the backbone of the Mumbai Indians' bowling attack, helping the franchise capture multiple IPL titles. His ability to choke run-rates and pick up crucial wickets under immense pressure cemented his status as a legendary figure in T20 league history.

### 3. International Dominance
After making his international debut for India in 2016, Bumrah quickly transitioned from a white-ball specialist into a world-class all-format powerhouse.
- **In Test Cricket:** He terrorized batting lineups globally, delivering historic spells in SENA countries (South Africa, England, New Zealand, Australia)—including a memorable multi-wicket performance and captaining India. He became one of the fastest Indian pacers to reach major milestone wickets.
- **In White-Ball Cricket:** He has consistently ranked among the top-performing bowlers globally, delivering championship-winning spells in high-stakes ICC tournaments.

### 4. Awards and Legacy
Bumrah's brilliance has earned him numerous global accolades, including prestigious honors like the **ICC Men's Cricketer of the Year** and **ICC Test Cricketer of the Year**. Revered by legends of the game as a "national treasure", Jasprit Bumrah remains a nightmare for opposing batsmen and a match-winner for India.
""",
    metadata={"team": "Mumbai Indians"} # Good practice to include metadata too!
)

# 3. Now update the document successfully
vector_store.update_document(
    document_id="e36e0a4e-ddb0-4fba-a092-16167a4281f3", 
    document=new_doc
)

print("Document updated successfully!")