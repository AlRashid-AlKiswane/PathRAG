"""
PDF Image Extraction Module

This module provides functionality to extract images from PDF documents using PyMuPDF (fitz).
It processes all pages of a PDF file, identifies embedded images, and saves them to a specified
output directory with proper error handling and logging.

Dependencies:
    - PyMuPDF (fitz): For PDF processing and image extraction
    - PIL (Pillow): For image format validation
    - os, sys: For file system operations
    - logging: Via custom setup_logging from src.infra

Classes:
    ExtractionImagesFromPDF: Main class for extracting images from PDF files

Example:
    extractor = ExtractionImagesFromPDF("document.pdf", "output_images/")
    extractor.extract_images()

Author: AlRashid AlKiswane
Created: 24-Aug-2025
"""

import logging
import os
import sys
import fitz

# Project path setup for relative imports
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if MAIN_DIR not in sys.path:
        sys.path.append(MAIN_DIR)
    logging.debug("Main directory path configured: %s", MAIN_DIR)
except (ImportError, OSError) as e:
    logging.critical("Failed to set up main directory path: %s", e, exc_info=True)
    sys.exit(1)

from src.infra import setup_logging
logger = setup_logging(name="EX-Images-PDF")

class ExtractionImagesFromPDF:
    """
    Extract images from PDF documents and save them to a specified directory.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str): Directory to save extracted images
    """
    
    def __init__(self, pdf_path: str, output_dir: str = "./extracted_images"):
        """
        Initialize the PDF image extractor.
        
        Args:
            pdf_path (str): Path to the PDF file to extract images from
            output_dir (str): Directory path where extracted images will be saved
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            PermissionError: If can't access PDF or create output directory
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            self.pdf_path = pdf_path
            self.output_dir = output_dir
            
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"Created output directory: {output_dir}")
            
            logger.info(f"Initialized extractor for: {pdf_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize: {str(e)}")
            raise
    
    def extract_images(self):
        """Extract all images from the PDF and save them."""
        try:
            images_paths = []
            with fitz.open(self.pdf_path) as pdf:
                logger.info(f"Processing PDF with {len(pdf)} pages")
                
                for page_index in range(len(pdf)):
                    try:
                        page = pdf.load_page(page_index)
                        image_list = page.get_images(full=True)
                        logger.debug(f"Found {len(image_list)} images on page {page_index + 1}")
                        
                        for img_index, img in enumerate(image_list, start=1):
                            try:
                                xref = img[0]
                                base_image = pdf.extract_image(xref)
                                img_bytes = base_image["image"]
                                img_ext = base_image["ext"]
                                img_name = f"page{page_index+1}_img{img_index}.{img_ext}"
                                
                                out_path = os.path.join(self.output_dir, img_name)
                                with open(out_path, "wb") as img_file:
                                    img_file.write(img_bytes)

                                logger.info(f"Saved {img_name}")
                                images_paths.append(out_path)
                            except Exception as e:
                                logger.error(f"Failed to extract image {img_index} from page {page_index + 1}: {str(e)}")
                                continue
                                
                    except Exception as e:
                        logger.error(f"Error processing page {page_index + 1}: {str(e)}")
                        continue
                        
            return images_paths  # ✅ return only after all pages are processed
        
        except Exception as e:
            logger.error(f"Failed to process PDF: {str(e)}")
            raise

if __name__ == "__main__":
    # PDF input path
    # pdf_path = "/home/alrashida/Tessafold/PathRAG/assets/docs/Clockify_Time_Report_Summary_01_06_2025-30_06_2025.pdf"
    pdf_path = "/home/alrashida/Tessafold/PathRAG/assets/docs/ALR/Clockify_Time_Report_Summary_01_07_2025_31_07_2025_20250831_131744_2236c841.pdf"
    pdf_path = "/home/alrashida/Tessafold/PathRAG/assets/docs/ALR/07_ConstructionMethodsStatements_20250831_131827_14dda024.pdf"
    # pdf_path = "/home/alrashida/Tessafold/PathRAG/assets/docs/Phys320_L8.pdf"
    # Output directory for extracted images
    output_dir = "./extracted_images"

    # Initialize the extractor
    extractor = ExtractionImagesFromPDF(pdf_path, output_dir)

    # Extract images
    print(extractor.extract_images())

    print(f"✅ Images extracted from PDF and saved to {output_dir}")
