#!/usr/bin/env python3
"""
Reset ChromaDB - Completely deletes and reinitializes the ChromaDB database
"""

import os
import shutil
import logging
import chromadb
from chromadb.config import Settings
from config.settings import VECTORSTORE_PERSIST_DIRECTORY, VECTORSTORE_COLLECTION_NAME

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reset_chromadb():
    """Reset ChromaDB by deleting the database files and recreating it"""
    try:
        # Delete the entire vectorstore directory
        if os.path.exists(VECTORSTORE_PERSIST_DIRECTORY):
            logger.info(f"Removing directory: {VECTORSTORE_PERSIST_DIRECTORY}")
            shutil.rmtree(VECTORSTORE_PERSIST_DIRECTORY)
            logger.info(f"Successfully deleted: {VECTORSTORE_PERSIST_DIRECTORY}")
        
        # Recreate the directory
        os.makedirs(VECTORSTORE_PERSIST_DIRECTORY, exist_ok=True)
        
        # Also delete the main sqlite file if it exists separately
        chroma_db_file = os.path.join(os.path.dirname(VECTORSTORE_PERSIST_DIRECTORY), "vectorstore", "chroma.sqlite3")
        if os.path.exists(chroma_db_file):
            logger.info(f"Removing ChromaDB SQLite file: {chroma_db_file}")
            os.remove(chroma_db_file)
        
        # Initialize ChromaDB to create a new, empty database
        logger.info("Initializing new ChromaDB database")
        client = chromadb.PersistentClient(
            path=VECTORSTORE_PERSIST_DIRECTORY,
            settings=Settings(allow_reset=True)
        )
        
        # Create empty collection
        collection = client.create_collection(
            name=VECTORSTORE_COLLECTION_NAME,
            metadata={"description": "Document chunks with semantic embeddings"}
        )
        
        count = collection.count()
        logger.info(f"Created new collection '{VECTORSTORE_COLLECTION_NAME}' with {count} documents")
        logger.info("ChromaDB reset completed successfully")
    
    except Exception as e:
        logger.error(f"Error resetting ChromaDB: {e}")
        raise

if __name__ == "__main__":
    reset_chromadb()
