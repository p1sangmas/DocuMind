#!/usr/bin/env python3
"""
PDF Text Extraction Diagnostic Tool

This script checks if a PDF document requires OCR by testing all available extraction methods
and reporting the results. This helps diagnose problematic PDFs.

Usage:
    python check_pdf.py path/to/document.pdf
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to run the PDF diagnostic tool"""
    if len(sys.argv) < 2:
        print("Usage: python check_pdf.py path/to/document.pdf")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"Error: File '{pdf_path}' not found")
        sys.exit(1)
    
    # Import the document processor
    try:
        from src.document_processor import DocumentProcessor, OCR_AVAILABLE
    except ImportError:
        print("Error: Failed to import DocumentProcessor. Make sure you're running this from the project root directory.")
        sys.exit(1)
    
    # Create document processor
    processor = DocumentProcessor()
    
    print(f"\n📄 Analyzing PDF: {pdf_path}\n")
    print("Testing extraction methods:")
    
    # Try each extraction method individually
    methods = ['pymupdf', 'pdfplumber', 'pypdf2']
    results = {}
    
    for method_name in methods:
        if method_name in processor.processors:
            method = processor.processors[method_name]
            try:
                print(f"\n🔍 Testing {method_name}...")
                text = method(pdf_path)
                words = text.split()
                char_count = len(text)
                word_count = len(words)
                
                # Validation check
                is_valid = processor._validate_extraction(text)
                needs_ocr = processor._needs_ocr(text)
                
                # Store results
                results[method_name] = {
                    'status': 'Success' if is_valid else 'Failed validation',
                    'characters': char_count,
                    'words': word_count,
                    'needs_ocr': needs_ocr,
                    'sample': text[:100] + '...' if text else 'No text extracted'
                }
                
                print(f"  Characters: {char_count}")
                print(f"  Words: {word_count}")
                print(f"  Validation: {'Passed ✅' if is_valid else 'Failed ❌'}")
                print(f"  Needs OCR: {'Yes' if needs_ocr else 'No'}")
                print(f"  Sample text: {text[:100]}...")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                results[method_name] = {
                    'status': f'Error: {str(e)}',
                    'characters': 0,
                    'words': 0,
                    'needs_ocr': True,
                    'sample': ''
                }
    
    # OCR test if available
    if OCR_AVAILABLE and 'ocr' in processor.processors:
        print("\n🔍 Testing OCR...")
        try:
            text = processor.processors['ocr'](pdf_path)
            words = text.split()
            char_count = len(text)
            word_count = len(words)
            
            results['ocr'] = {
                'status': 'Success' if word_count > 5 else 'Low word count',
                'characters': char_count,
                'words': word_count,
                'needs_ocr': False,  # OCR already applied
                'sample': text[:100] + '...' if text else 'No text extracted'
            }
            
            print(f"  Characters: {char_count}")
            print(f"  Words: {word_count}")
            print(f"  Sample text: {text[:100]}...")
        except Exception as e:
            print(f"  OCR Error: {str(e)}")
            results['ocr'] = {
                'status': f'Error: {str(e)}',
                'characters': 0,
                'words': 0,
                'needs_ocr': False,
                'sample': ''
            }
    else:
        print("\n❌ OCR support not available")
        print("   Install OCR dependencies with: pip install pytesseract pdf2image pillow")
        print("   See OCR_SETUP.md for more details")
    
    # Summary
    print("\n📊 Summary:")
    best_method = max(results.items(), key=lambda x: x[1]['words']) if results else (None, None)
    if best_method[0]:
        print(f"  Best extraction method: {best_method[0]} ({best_method[1]['words']} words)")
        print(f"  OCR recommended: {any(v['needs_ocr'] for v in results.values())}")
    
        # Final recommendation
        if any(v['needs_ocr'] for v in results.values()):
            if 'ocr' in results and results['ocr']['words'] > 0:
                print("\n✅ Recommendation: Use OCR for this document")
            else:
                print("\n⚠️ Recommendation: This document needs OCR, but OCR is not available or failed")
        else:
            print(f"\n✅ Recommendation: Use standard extraction ({best_method[0]})")
    else:
        print("  No successful extraction methods found")

if __name__ == "__main__":
    main()
