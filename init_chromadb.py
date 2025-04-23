import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb.utils import embedding_functions
import docx
import pandas as pd
import logging
import streamlit as st

api_key = st.secrets.get("open-key")
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Read API key from secrets.toml
import toml
secrets_path = "/workspaces/AI-Chatbot/.streamlit/secrets.toml"
try:
    secrets = toml.load(secrets_path)
except Exception as e:
    logger.error(f"Failed to load secrets.toml: {str(e)}")
    raise

# Initialize ChromaDB client
try:
    chroma_client = chromadb.PersistentClient(path="/workspaces/AI-Chatbot/chroma_db")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
    raise

# Setup OpenAI embedding function
try:
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key= api_key,
        model_name="text-embedding-ada-002"
    )
except Exception as e:
    logger.error(f"Failed to initialize OpenAI embedding function: {str(e)}")
    raise

# Create or get collections
university_collection = chroma_client.get_or_create_collection(
    name="university_info",
    embedding_function=embedding_function
)
living_expenses_collection = chroma_client.get_or_create_collection(
    name="living_expenses",
    embedding_function=embedding_function
)
employment_collection = chroma_client.get_or_create_collection(
    name="employment_projections",
    embedding_function=embedding_function
)

def load_word_document(file_path: str) -> str:
    """Load content from a Word document."""
    try:
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        logger.error(f"Error loading Word document: {str(e)}")
        return ""

# Load and populate data
if university_collection.count() == 0:
    logger.info("Loading university data...")
    university_text = load_word_document("/workspaces/AI-Chatbot/Dataset/University_Data.docx")
    if not university_text:
        logger.warning("No university data loaded; skipping university collection.")
    else:
        chunk_size = 1000
        chunks = [university_text[i:i + chunk_size] for i in range(0, len(university_text), chunk_size)]
        for idx, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                continue
            university_collection.add(
                documents=[chunk],
                metadatas=[{"chunk_id": idx, "type": "university", "length": len(chunk)}],
                ids=[f"university_{idx}"]
            )
        logger.info(f"Loaded {len(chunks)} chunks into university collection.")

if living_expenses_collection.count() == 0:
    logger.info("Loading living expenses data...")
    try:
        living_expenses_df = pd.read_csv("/workspaces/AI-Chatbot/Dataset/Avg_Living_Expenses.csv")
    except Exception as e:
        logger.error(f"Failed to load Avg_Living_Expenses.csv: {str(e)}")
        raise
    for idx, row in living_expenses_df.iterrows():
        content = (
            f"State: {row['State']}\n"
            f"Cost of Living Index: {row['Index']}\n"
            f"Grocery: {row['Grocery']}\n"
            f"Housing: {row['Housing']}\n"
            f"Utilities: {row['Utilities']}\n"
            f"Transportation: {row['Transportation']}\n"
            f"Health: {row['Health']}\n"
            f"Miscellaneous: {row['Misc']}"
        )
        living_expenses_collection.add(
            documents=[content],
            metadatas=[{
                "state": row["State"].strip(),
                "type": "living_expenses",
                "index": float(row["Index"]),
                "housing_index": float(row["Housing"])
            }],
            ids=[f"living_expenses_{idx}"]
        )
    logger.info(f"Loaded {len(living_expenses_df)} entries into living expenses collection.")

if employment_collection.count() == 0:
    logger.info("Loading employment projections data...")
    try:
        employment_df = pd.read_csv("/workspaces/AI-Chatbot/Dataset/Employment_Projections.csv")
    except Exception as e:
        logger.error(f"Failed to load Employment_Projections.csv: {str(e)}")
        raise
    for idx, row in employment_df.iterrows():
        content = (
            f"Occupation: {row['Occupation Title']}\n"
            f"Occupation Code: {row['Occupation Code']}\n"
            f"Employment 2023: {row['Employment 2023']}\n"
            f"Employment 2033: {row['Employment 2033']}\n"
            f"Employment Change (2023-2033): {row['Employment Change, 2023-2033']}\n"
            f"Growth Rate: {row['Employment Percent Change, 2023-2033']}%\n"
            f"Annual Openings: {row['Occupational Openings, 2023-2033 Annual Average']}\n"
            f"Median Wage: ${row['Median Annual Wage 2023']}\n"
            f"Required Education: {row['Typical Entry-Level Education']}\n"
            f"Work Experience: {row['Work Experience in a Related Occupation']}\n"
            f"Typical On-the-Job Training: {row['Typical on-the-job Training']}"
        )
        employment_collection.add(
            documents=[content],
            metadatas=[{
                "occupation": row["Occupation Title"],
                "occupation_code": row["Occupation Code"],
                "type": "employment",
                "median_wage": str(row["Median Annual Wage 2023"]),
                "growth_rate": float(row["Employment Percent Change, 2023-2033"]),
                "education": row["Typical Entry-Level Education"],
                "work_experience": row["Work Experience in a Related Occupation"],
                "training": row["Typical on-the-job Training"]
            }],
            ids=[f"employment_{idx}"]
        )
    logger.info(f"Loaded {len(employment_df)} entries into employment projections collection.")

logger.info("Successfully initialized ChromaDB collections")
print("Collections initialized successfully!")
