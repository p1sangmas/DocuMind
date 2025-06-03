#!/usr/bin/env python3
"""
Fix document metadata
This script fixes metadata issues in the document collection
"""

import logging
import argparse
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for importing modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import our modules
from src.retriever import HybridRetriever

def fix_generic_doc_ids():
    """Fix documents with generic doc_ID metadata"""
    logger.info("Searching for documents with generic doc_ID metadata...")
    
    # Initialize retriever
    retriever = HybridRetriever()
    
    # Get all documents
    try:
        collection_results = retriever.collection.get()
        
        if not collection_results or not collection_results['ids']:
            logger.warning("No documents found in collection")
            return 0
            
        # Check for documents with generic IDs
        fixed_count = 0
        for i, (doc_id, metadata, content) in enumerate(zip(
            collection_results['ids'], 
            collection_results['metadatas'], 
            collection_results['documents']
        )):
            source_doc = metadata.get('source_document', '')
            
            # Check if this is a generic doc_ID
            if source_doc.startswith('doc_') or source_doc == 'Unknown':
                logger.info(f"Found document with generic ID: {source_doc}")
                
                # Try to extract a better name
                better_name = None
                
                # Check for title in the content (first line might be title)
                first_line = content.split('\n')[0] if content else ''
                if first_line and len(first_line) < 100:
                    better_name = first_line.strip()
                
                # Use original filename if available
                if 'original_filename' in metadata:
                    better_name = metadata['original_filename']
                    
                # Use filename if available
                elif 'filename' in metadata:
                    better_name = metadata['filename']
                
                # Update only if we found a better name
                if better_name:
                    logger.info(f"Updating document {doc_id} source from '{source_doc}' to '{better_name}'")
                    
                    # Update metadata
                    updated_metadata = {
                        'source_document': better_name,
                        'title': better_name
                    }
                    
                    if retriever.update_document_metadata(doc_id, updated_metadata):
                        fixed_count += 1
                    
        logger.info(f"Fixed {fixed_count} documents with generic IDs")
        return fixed_count
        
    except Exception as e:
        logger.error(f"Error fixing document metadata: {e}")
        return 0

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Fix document metadata')
    parser.add_argument('--fix-generic-ids', action='store_true', 
                       help='Fix documents with generic doc_ID metadata')
    parser.add_argument('--all', action='store_true',
                       help='Apply all fixes')
                       
    args = parser.parse_args()
    
    if not (args.fix_generic_ids or args.all):
        parser.print_help()
        return
        
    if args.fix_generic_ids or args.all:
        fixed = fix_generic_doc_ids()
        if fixed > 0:
            print(f"✅ Fixed {fixed} documents with generic IDs")
        else:
            print("ℹ️ No documents with generic IDs found")

if __name__ == "__main__":
    main()
