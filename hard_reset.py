#!/usr/bin/env python3
"""
Complete Hard Reset - Deletes and reinitializes all data in the DocuMind app
"""

import os
import shutil
import logging
import sys
import sqlite3

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths
try:
    from config.settings import (
        VECTORSTORE_PERSIST_DIRECTORY,
        VECTORSTORE_COLLECTION_NAME,
        DATA_DIR
    )
except ImportError:
    # Fallback if config can't be imported
    logger.warning("Could not import settings, using hardcoded paths")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(current_dir, "data")
    VECTORSTORE_PERSIST_DIRECTORY = os.path.join(DATA_DIR, "vectorstore")
    VECTORSTORE_COLLECTION_NAME = "documents"

def confirm_action(message):
    """Ask for confirmation before proceeding"""
    response = input(f"{message} (y/n): ").lower().strip()
    return response == 'y'

def hard_reset():
    """Completely reset all data in the DocuMind app"""
    print("=" * 80)
    print("COMPLETE HARD RESET OF DOCUMIND")
    print("=" * 80)
    print("This will delete ALL data:")
    print(" - Vector database (ChromaDB)")
    print(" - Feedback database")
    print(" - Logs")
    print(" - Any cached files")
    print(" - Session files")
    print("=" * 80)
    
    if not confirm_action("Are you sure you want to proceed?"):
        print("Hard reset cancelled.")
        return False
    
    try:
        # Clear ChromaDB vectorstore
        if os.path.exists(VECTORSTORE_PERSIST_DIRECTORY):
            logger.info(f"Removing directory: {VECTORSTORE_PERSIST_DIRECTORY}")
            shutil.rmtree(VECTORSTORE_PERSIST_DIRECTORY)
            logger.info(f"Successfully deleted: {VECTORSTORE_PERSIST_DIRECTORY}")
        
        # Recreate the directory
        os.makedirs(VECTORSTORE_PERSIST_DIRECTORY, exist_ok=True)
        
        # Remove the ChromaDB SQLite file if it exists
        chroma_db_file = os.path.join(DATA_DIR, "vectorstore", "chroma.sqlite3")
        if os.path.exists(chroma_db_file):
            logger.info(f"Removing ChromaDB SQLite file: {chroma_db_file}")
            os.remove(chroma_db_file)
        
        # Delete feedback database
        feedback_db_file = os.path.join(DATA_DIR, "feedback.db")
        if os.path.exists(feedback_db_file):
            logger.info(f"Removing feedback database: {feedback_db_file}")
            os.remove(feedback_db_file)
        
        # # Clear log file but don't delete it
        # log_file = os.path.join(DATA_DIR, "documind.log")
        # if os.path.exists(log_file):
        #     logger.info(f"Clearing log file: {log_file}")
        #     with open(log_file, 'w') as f:
        #         f.write("# Log file cleared by hard_reset.py\n")
        
        # Clear any Streamlit cache
        streamlit_cache = os.path.expanduser("~/.streamlit/cache")
        if os.path.exists(streamlit_cache):
            logger.info(f"Clearing Streamlit cache: {streamlit_cache}")
            shutil.rmtree(streamlit_cache)
            os.makedirs(streamlit_cache, exist_ok=True)
        
        print("=" * 80)
        print("HARD RESET COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("All data has been reset. Please restart your Streamlit app.")
        return True
        
    except Exception as e:
        logger.error(f"Error during hard reset: {e}")
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    hard_reset()
