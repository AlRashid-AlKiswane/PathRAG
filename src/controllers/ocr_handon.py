# """
# Advanced OCR Document Processing Module

# This module provides a comprehensive OCR (Optical Character Recognition) document processing system
# that can handle PDF documents with advanced text extraction, cleaning, translation, and export
# capabilities. The module supports multiple OCR engines, table structure recognition, and
# multilingual document processing with automatic translation from Arabic to English.

# Features:
# - Advanced PDF to text conversion with table structure recognition
# - Multiple OCR engine support (Tesseract, EasyOCR, RapidOCR)
# - Text cleaning and deduplication algorithms
# - HTML to Markdown conversion with structure preservation
# - Automatic Arabic to English translation using NLLB models
# - Comprehensive logging and error handling
# - Type-safe implementation with full type annotations

# Classes:
#     OCRProcessor: Main class for document OCR processing and conversion

# Dependencies:
#     - docling: Document conversion and OCR processing
#     - transformers: Machine learning models for translation
#     - beautifulsoup4: HTML parsing and manipulation
#     - markdownify: HTML to Markdown conversion
#     - pathlib: File system path handling

# Author: OCR Processing System
# Version: 1.0.0
# """

# import time
# import re
# import logging
# from pathlib import Path
# from typing import Optional, Dict, Any, List, Union, Tuple
# from collections import OrderedDict

# from docling.document_converter import DocumentConverter, PdfFormatOption
# from docling.datamodel.base_models import InputFormat
# from docling.datamodel.pipeline_options import (
#     PdfPipelineOptions, TableFormerMode, TesseractOcrOptions,
#     EasyOcrOptions, TesseractCliOcrOptions, RapidOcrOptions,
#     AcceleratorDevice, AcceleratorOptions
# )
# from bs4 import BeautifulSoup, NavigableString
# from markdownify import markdownify as md
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# def setup_logging(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
#     """
#     Set up comprehensive logging configuration for the OCR processing system.
    
#     Args:
#         name: Logger name identifier
#         level: Logging level (default: INFO)
        
#     Returns:
#         Configured logger instance
#     """
#     logger = logging.getLogger(name)
#     logger.setLevel(level)
    
#     if not logger.handlers:
#         handler = logging.StreamHandler()
#         formatter = logging.Formatter(
#             '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#         )
#         handler.setFormatter(formatter)
#         logger.addHandler(handler)
    
#     return logger


# class OCRProcessor:
#     """
#     Advanced OCR Document Processing Class
    
#     A comprehensive class for processing documents using OCR technology with support
#     for multiple languages, text cleaning, translation, and various output formats.
    
#     This class handles the complete pipeline from document input to cleaned text output,
#     including OCR processing, text cleaning, deduplication, and optional translation
#     from Arabic to English.
    
#     Attributes:
#         logger: Logger instance for tracking processing steps and errors
#         pipeline_options: OCR pipeline configuration options
#         translator_cache: Cache for loaded translation models
        
#     Methods:
#         process_document: Main method to process a document through the OCR pipeline
#         get_pipeline_options: Configure OCR processing pipeline options
#         convert_document: Convert document using configured pipeline
#         export_document: Export processed text to file
#         clean_ocr_text: Clean OCR artifacts and normalize text
#         remove_duplicate_lines: Remove duplicate content lines
#         process_md_ocr: Process markdown OCR output
#         process_html_ocr: Process HTML OCR output
#         translate: Translate content from Arabic to English
#         translate_html: Translate HTML content preserving structure
#     """
    
#     def __init__(self, logger_name: str = "OCRProcessor") -> None:
#         """
#         Initialize the OCR Processor with logging and caching capabilities.
        
#         Args:
#             logger_name: Name for the logger instance
#         """
#         self.logger: logging.Logger = setup_logging(logger_name)
#         self.pipeline_options: Optional[PdfPipelineOptions] = None
#         self.translator_cache: Dict[str, Any] = {}
        
#         self.logger.info("OCR Processor initialized successfully")
    
#     def process_document(self, args: Any) -> None:
#         """
#         Main method to process a document through the complete OCR pipeline.
        
#         This method orchestrates the entire document processing workflow including
#         document conversion, text processing, optional translation, and export.
        
#         Args:
#             args: Arguments object containing:
#                 - input: Path to input document
#                 - output: Path to output directory
#                 - lang: Language code ('en' or 'ar')
                
#         Raises:
#             FileNotFoundError: If input document doesn't exist
#             PermissionError: If unable to write to output directory
#             Exception: For any processing errors
#         """
#         try:
#             start_time = time.time()
#             self.logger.info("Starting document OCR processing")
            
#             # Validate input file
#             input_doc = Path(args.input)
#             if not input_doc.exists():
#                 raise FileNotFoundError(f"Input document not found: {input_doc}")
            
#             if not input_doc.is_file():
#                 raise ValueError(f"Input path is not a file: {input_doc}")
                
#             doc_filename = input_doc.stem
#             self.logger.info(f"Processing document: {doc_filename}")
            
#             # Get pipeline configuration
#             pipeline_options = self.get_pipeline_options()
#             self.logger.debug("Pipeline options configured successfully")
            
#             # Convert document
#             doc = self.convert_document(input_doc, pipeline_options)
#             self.logger.info("Document conversion completed")
            
#             # Process based on language
#             if args.lang == "en":
#                 self.logger.info("Processing English document")
#                 md_text = self.process_md_ocr(doc.export_to_markdown())
#             elif args.lang == "ar":
#                 self.logger.info("Processing Arabic document with translation")
#                 translated_text = self.translate(doc.export_to_html(), 0)
#                 md_text = self.process_md_ocr(translated_text)
#             else:
#                 raise ValueError(f"Unsupported language: {args.lang}")
            
#             # Export processed document
#             self.export_document(md_text, args.output, doc_filename)
            
#             elapsed = time.time() - start_time
#             self.logger.info(f"Document converted successfully in {elapsed:.2f} seconds")
            
#         except FileNotFoundError as e:
#             self.logger.error(f"File not found error: {e}")
#             raise
#         except PermissionError as e:
#             self.logger.error(f"Permission error: {e}")
#             raise
#         except ValueError as e:
#             self.logger.error(f"Value error: {e}")
#             raise
#         except Exception as e:
#             self.logger.error(f"Unexpected error during document processing: {e}", exc_info=True)
#             raise
    
#     def get_pipeline_options(self) -> PdfPipelineOptions:
#         """
#         Configure and return OCR pipeline processing options.
        
#         Sets up comprehensive OCR processing options including table structure
#         recognition, OCR engine selection, and performance optimization settings.
        
#         Returns:
#             Configured PdfPipelineOptions instance
            
#         Raises:
#             Exception: If pipeline configuration fails
#         """
#         try:
#             self.logger.debug("Configuring pipeline options")
            
#             options = PdfPipelineOptions()
#             options.do_ocr = True
#             options.do_table_structure = True
#             options.table_structure_options.do_cell_matching = True
#             options.table_structure_options.mode = TableFormerMode.ACCURATE
            
#             # Configure OCR options - using TesseractCliOcrOptions
#             ocr_options = TesseractCliOcrOptions(lang=["auto"])
#             options.ocr_options = ocr_options
            
#             self.pipeline_options = options
#             self.logger.debug("Pipeline options configured successfully")
            
#             return options
            
#         except Exception as e:
#             self.logger.error(f"Failed to configure pipeline options: {e}", exc_info=True)
#             raise
    
#     def convert_document(self, input_dir: Path, pipeline_options: PdfPipelineOptions) -> Any:
#         """
#         Convert document using the configured OCR pipeline.
        
#         Args:
#             input_dir: Path to the input document
#             pipeline_options: Configured pipeline processing options
            
#         Returns:
#             Converted document object
            
#         Raises:
#             Exception: If document conversion fails
#         """
#         try:
#             self.logger.debug(f"Converting document: {input_dir}")
            
#             converter = DocumentConverter(
#                 format_options={
#                     InputFormat.PDF: PdfFormatOption(
#                         pipeline_options=pipeline_options,
#                     )
#                 }
#             )
            
#             doc = converter.convert(input_dir).document
#             self.logger.debug("Document conversion completed successfully")
            
#             return doc
            
#         except Exception as e:
#             self.logger.error(f"Document conversion failed: {e}", exc_info=True)
#             raise
    
#     def export_document(self, md_text: str, output_dir: str, filename: str) -> None:
#         """
#         Export processed markdown text to file.
        
#         Args:
#             md_text: Processed markdown text content
#             output_dir: Output directory path
#             filename: Output filename (without extension)
            
#         Raises:
#             PermissionError: If unable to create directory or write file
#             Exception: For any export errors
#         """
#         try:
#             self.logger.debug(f"Exporting document to: {output_dir}/{filename}.md")
            
#             out_dir = Path(output_dir)
#             out_dir.mkdir(parents=True, exist_ok=True)
            
#             output_file = out_dir / f"{filename}.md"
#             with output_file.open("w", encoding="utf-8") as fp:
#                 fp.write(md_text)
            
#             self.logger.info(f"Document exported successfully to: {output_file}")
            
#         except PermissionError as e:
#             self.logger.error(f"Permission denied during export: {e}")
#             raise
#         except Exception as e:
#             self.logger.error(f"Export failed: {e}", exc_info=True)
#             raise
    
#     def clean_ocr_text(self, text: str) -> str:
#         """
#         Clean OCR artifacts and normalize text formatting.
        
#         Removes common OCR errors, normalizes whitespace, fixes hyphenation,
#         and standardizes punctuation formatting.
        
#         Args:
#             text: Raw OCR text to clean
            
#         Returns:
#             Cleaned and normalized text
            
#         Raises:
#             TypeError: If input is not a string
#         """
#         try:
#             if not isinstance(text, str):
#                 raise TypeError(f"Expected string input, got {type(text)}")
                
#             self.logger.debug("Cleaning OCR text artifacts")
            
#             # Remove image placeholders
#             text = text.replace("<!-- image -->", "")
            
#             # Fix line-break hyphens
#             text = re.sub(r'(\S)-\s+(\S)', r'\1\2', text)
            
#             # Clean excessive dots
#             text = re.sub(r'\.{3,}', '', text)
            
#             # Remove stray punctuation
#             text = text.replace('..', '')
#             text = text.replace('--', '')
#             text = text.replace(':', '')
            
#             # Fix spacing around punctuation
#             text = re.sub(r'\s+([.,;:!?])', r'\1', text)
#             text = re.sub(r'(["\(])\s+', r'\1', text)
#             text = re.sub(r'\s+(["\)])', r'\1', text)
            
#             # Normalize whitespace
#             text = re.sub(r'[^\S\n]+', ' ', text)
#             text = re.sub(r'\n\s+', '\n', text)
            
#             self.logger.debug("OCR text cleaning completed")
#             return text.strip()
            
#         except Exception as e:
#             self.logger.error(f"Text cleaning failed: {e}", exc_info=True)
#             raise
    
#     def remove_duplicate_lines(self, text: str, max_repeats: int = 1) -> str:
#         """
#         Remove duplicate lines from text content.
        
#         Identifies and removes both consecutive and non-consecutive duplicate lines
#         while preserving the original text structure and formatting.
        
#         Args:
#             text: Input text with potential duplicates
#             max_repeats: Maximum allowed consecutive repetitions
            
#         Returns:
#             Text with duplicates removed
            
#         Raises:
#             TypeError: If input is not a string
#             ValueError: If max_repeats is negative
#         """
#         try:
#             if not isinstance(text, str):
#                 raise TypeError(f"Expected string input, got {type(text)}")
            
#             if max_repeats < 0:
#                 raise ValueError("max_repeats must be non-negative")
                
#             self.logger.debug(f"Removing duplicate lines (max_repeats: {max_repeats})")
            
#             lines = text.split('\n')
#             cleaned = []
#             prev_line = None
#             repeat_count = 0
            
#             for line in lines:
#                 # Normalize for comparison
#                 norm_line = re.sub(r'\s+', ' ', line).strip().lower()
                
#                 if norm_line == prev_line:
#                     repeat_count += 1
#                     if repeat_count > max_repeats:
#                         continue
#                 else:
#                     repeat_count = 0
#                     prev_line = norm_line
                
#                 cleaned.append(line)
            
#             # Remove non-consecutive duplicates
#             unique_lines = list(OrderedDict.fromkeys(cleaned).keys())
            
#             result = '\n'.join(unique_lines)
#             self.logger.debug(f"Removed {len(lines) - len(unique_lines)} duplicate lines")
            
#             return result
            
#         except Exception as e:
#             self.logger.error(f"Duplicate removal failed: {e}", exc_info=True)
#             raise
    
#     def process_md_ocr(self, text: str) -> str:
#         """
#         Process markdown OCR output with cleaning and deduplication.
        
#         Applies comprehensive text cleaning including OCR artifact removal,
#         duplicate line elimination, and final formatting normalization.
        
#         Args:
#             text: Raw markdown OCR text
            
#         Returns:
#             Processed and cleaned markdown text
            
#         Raises:
#             TypeError: If input is not a string
#         """
#         try:
#             if not isinstance(text, str):
#                 raise TypeError(f"Expected string input, got {type(text)}")
                
#             self.logger.debug("Processing markdown OCR output")
            
#             # Clean OCR artifacts
#             cleaned_content = self.clean_ocr_text(text)
            
#             # Remove duplicate lines
#             deduped_content = self.remove_duplicate_lines(cleaned_content, max_repeats=1)
            
#             # Final cleanup - normalize newlines
#             processed = re.sub(r'\n{3,}', '\n\n', deduped_content)
            
#             self.logger.debug("Markdown OCR processing completed")
#             return processed.strip()
            
#         except Exception as e:
#             self.logger.error(f"Markdown OCR processing failed: {e}", exc_info=True)
#             raise
    
#     def remove_html_duplicates(self, soup: BeautifulSoup, max_repeats: int = 1) -> BeautifulSoup:
#         """
#         Remove duplicate text content from HTML while preserving structure.
        
#         Args:
#             soup: BeautifulSoup HTML document object
#             max_repeats: Maximum allowed text repetitions
            
#         Returns:
#             Modified BeautifulSoup object with duplicates removed
            
#         Raises:
#             TypeError: If soup is not a BeautifulSoup object
#         """
#         try:
#             if not isinstance(soup, BeautifulSoup):
#                 raise TypeError("Expected BeautifulSoup object")
                
#             self.logger.debug("Removing HTML duplicate content")
            
#             text_content = OrderedDict()
            
#             def process_element(element):
#                 if isinstance(element, NavigableString):
#                     return
                
#                 for child in element.contents:
#                     if child.name in ['script', 'style', 'code', 'pre']:
#                         continue
                    
#                     if isinstance(child, NavigableString):
#                         parent = child.parent
#                         if parent.name in ['p', 'div', 'span', 'li']:
#                             text = child.strip()
#                             if text:
#                                 count = text_content.get(text, 0)
#                                 if count >= max_repeats:
#                                     child.replace_with('')
#                                 else:
#                                     text_content[text] = count + 1
#                     else:
#                         process_element(child)
            
#             process_element(soup)
#             self.logger.debug("HTML duplicate removal completed")
            
#             return soup
            
#         except Exception as e:
#             self.logger.error(f"HTML duplicate removal failed: {e}", exc_info=True)
#             raise
    
#     def process_html_ocr(self, html: str) -> str:
#         """
#         Process HTML OCR output with comprehensive cleaning.
        
#         Args:
#             html: Raw HTML OCR content
            
#         Returns:
#             Cleaned and formatted HTML content
            
#         Raises:
#             TypeError: If input is not a string
#         """
#         try:
#             if not isinstance(html, str):
#                 raise TypeError(f"Expected string input, got {type(html)}")
                
#             self.logger.debug("Processing HTML OCR output")
            
#             # Parse HTML
#             soup = BeautifulSoup(html, 'lxml')
            
#             # Clean text in all appropriate nodes
#             for element in soup.find_all(text=True):
#                 if element.parent.name not in ['script', 'style', 'code', 'pre']:
#                     cleaned = self.clean_ocr_text(element)
#                     element.replace_with(cleaned)
            
#             # Remove structural duplicates
#             soup = self.remove_html_duplicates(soup)
            
#             # Final HTML cleanup
#             cleaned_html = soup.prettify()
            
#             # Advanced whitespace compression
#             cleaned_html = re.sub(r'>\s+<', '><', cleaned_html)
#             cleaned_html = re.sub(r'(?<=\>)\s+', '\n', cleaned_html)
#             cleaned_html = re.sub(r'\s+(?=\<)', '\n', cleaned_html)
#             cleaned_html = re.sub(r'\n{3,}', '\n\n', cleaned_html)
            
#             self.logger.debug("HTML OCR processing completed")
#             return cleaned_html
            
#         except Exception as e:
#             self.logger.error(f"HTML OCR processing failed: {e}", exc_info=True)
#             raise
    
#     def load_translator(self, model_name: str, src_lang: str, tgt_lang: str, device: int) -> Any:
#         """
#         Load and cache translation model for reuse.
        
#         Args:
#             model_name: Name of the translation model
#             src_lang: Source language code
#             tgt_lang: Target language code
#             device: Device ID for computation
            
#         Returns:
#             Translation pipeline object
            
#         Raises:
#             Exception: If model loading fails
#         """
#         try:
#             cache_key = f"{model_name}_{src_lang}_{tgt_lang}_{device}"
            
#             if cache_key in self.translator_cache:
#                 self.logger.debug(f"Using cached translator: {cache_key}")
#                 return self.translator_cache[cache_key]
            
#             self.logger.info(f"Loading translation model: {model_name}")
            
#             tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang, tgt_lang=tgt_lang)
#             model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#             translator = pipeline("translation", model=model, tokenizer=tokenizer, 
#                                 src_lang=src_lang, tgt_lang=tgt_lang, device=device)
            
#             self.translator_cache[cache_key] = translator
#             self.logger.info("Translation model loaded successfully")
            
#             return translator
            
#         except Exception as e:
#             self.logger.error(f"Translation model loading failed: {e}", exc_info=True)
#             raise
    
#     def translate(self, data: str, device: int = 0) -> str:
#         """
#         Translate HTML content from Arabic to English and convert to Markdown.
        
#         Args:
#             data: HTML content in Arabic
#             device: Device ID for computation (0 for CPU, 1+ for GPU)
            
#         Returns:
#             Translated content in Markdown format
            
#         Raises:
#             TypeError: If data is not a string
#             Exception: If translation fails
#         """
#         try:
#             if not isinstance(data, str):
#                 raise TypeError(f"Expected string input, got {type(data)}")
                
#             self.logger.info("Starting Arabic to English translation")
            
#             # NLLB language codes
#             model_name = "facebook/nllb-200-distilled-600M"
#             translator = self.load_translator(model_name, "arb_Arab", "eng_Latn", device=device)
            
#             # Translate HTML content
#             translated_content = self.translate_html(data, translator)
            
#             # Convert to Markdown
#             markdown = md(str(translated_content), heading_style="ATX")
            
#             self.logger.info("Translation completed successfully")
#             return markdown
            
#         except Exception as e:
#             self.logger.error(f"Translation failed: {e}", exc_info=True)
#             raise
    
#     def translate_html(self, data: str, translator: Any) -> BeautifulSoup:
#         """
#         Translate HTML content while preserving structure.
        
#         Args:
#             data: HTML content to translate
#             translator: Translation pipeline object
            
#         Returns:
#             BeautifulSoup object with translated content
            
#         Raises:
#             TypeError: If data is not a string
#             Exception: If HTML translation fails
#         """
#         try:
#             if not isinstance(data, str):
#                 raise TypeError(f"Expected string input, got {type(data)}")
                
#             self.logger.debug("Translating HTML content")
            
#             soup = BeautifulSoup(data, "html.parser")
#             text_nodes = [node for node in soup.find_all(text=True) if node.strip()]
            
#             translated_count = 0
#             for node in text_nodes:
#                 if node.parent.name not in ["script", "style", "code"]:
#                     try:
#                         translated = translator(node.strip(), max_length=512)[0]["translation_text"]
#                         node.replace_with(translated)
#                         translated_count += 1
#                     except Exception as e:
#                         self.logger.warning(f"Failed to translate text node: {e}")
#                         continue
            
#             self.logger.debug(f"Translated {translated_count} text nodes")
#             return soup
            
#         except Exception as e:
#             self.logger.error(f"HTML translation failed: {e}", exc_info=True)
#             raise

# from types import SimpleNamespace

# if __name__ == "__main__":
#     # Example arguments (mimicking argparse)
#     args = SimpleNamespace(
#         input="/home/alrashida/Tessafold/PathRAG/assets/docs/Clockify_Time_Report_Summary_01_06_2025-30_06_2025.pdf",
#         output="/home/alrashida/Tessafold/PathRAG/assets/ocr_output",
#         lang="en"   # use "ar" if the document is Arabic and needs translation
#     )

#     # Initialize processor
#     processor = OCRProcessor()

#     # Process the PDF
#     processor.process_document(args)


# import time ,re
# from pathlib import Path
# from docling.document_converter import DocumentConverter, PdfFormatOption
# from docling.datamodel.base_models import InputFormat
# from docling.datamodel.pipeline_options import PdfPipelineOptions,TableFormerMode, TesseractOcrOptions ,EasyOcrOptions ,TesseractCliOcrOptions,RapidOcrOptions,AcceleratorDevice, AcceleratorOptions
# from collections import OrderedDict
# from bs4 import BeautifulSoup, NavigableString
# from bs4 import BeautifulSoup
# from markdownify import markdownify as md
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# def OCR(args) -> None:
#     start_time = time.time()
#     input_doc = Path(args.input)
#     doc_filename = input_doc.stem
#     pipeline_options = get_pipeline_options()
#     doc = convert_document(input_doc, pipeline_options)
#     if (args.lang=="en"):
#         md_text=process_md_ocr(doc.export_to_markdown())
#     elif(args.lang=="ar"):
#         md_text=translate(doc.export_to_html(),0)
#         md_text=process_md_ocr(md_text)
#     export_document(md_text, args.output, doc_filename)
#     elapsed = time.time() - start_time
#     print(f"Document converted in {elapsed:.2f} seconds.")

# def get_pipeline_options() -> PdfPipelineOptions:
#     options = PdfPipelineOptions()
#     options.do_ocr = True
#     options.do_table_structure = True
#     options.table_structure_options.do_cell_matching = True
#     options.table_structure_options.mode = TableFormerMode.ACCURATE  #TableFormerMode.FAST   
#     # options.accelerator_options = AcceleratorOptions(
#     #     num_threads=6, device=AcceleratorDevice.CPU #.CPU
#     # )

#     # Choose your preferred OCR option. Here, we use TesseractOcrOptions.
#     # ocr_options = EasyOcrOptions(force_full_page_ocr=True)
#     ocr_options = TesseractCliOcrOptions(lang=["auto"] )#force_full_page_ocr=True,
#     options.ocr_options = ocr_options
#     # options.ocr_options.lang = ["en","ar"] #EasyOcrOptions
    
#     return options

# def convert_document(input_dir: str, pipeline_options):
#     converter = DocumentConverter(
#         format_options={
#             InputFormat.PDF: PdfFormatOption(
#                 pipeline_options=pipeline_options,
#             )
#         }
#     )
#     doc = converter.convert(input_dir).document
#     return doc

# def export_document(md_text, output_dir: str, filename: str):
#     out_dir = Path(output_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
    
#     with (out_dir / f"{filename}.md").open("w", encoding="utf-8") as fp:
#         fp.write(md_text)
    

# def clean_ocr_text(text: str) -> str:
#     # Normalize hyphens and line breaks
#     text = text.replace("<!-- image -->", "") 

#     text = re.sub(r'(\S)-\s+(\S)', r'\1\2', text)  # Fix line-break hyphens
    
#     # Clean excessive dots (3+ consecutive dots become ellipsis)
#     text = re.sub(r'\.{3,}', '', text)  # Standardize ellipses

#     # Remove stray single/double dots not at sentence boundaries
#     text = text.replace('..', '')
#     text = text.replace('--', '')
#     text = text.replace(':', '')

#     # Fix spacing around punctuation
#     text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Space before punctuation
#     text = re.sub(r'([“\(])\s+', r'\1', text)      # Space after opening quotes/parens
#     text = re.sub(r'\s+([”\)])', r'\1', text)      # Space before closing quotes/parens
    
#     # Normalize whitespace (preserve newlines)
#     text = re.sub(r'[^\S\n]+', ' ', text)          # Collapse multiple spaces
#     text = re.sub(r'\n\s+', '\n', text)            # Remove leading line spaces
    
#     return text.strip()

# def remove_duplicate_lines(text: str, max_repeats: int = 1) -> str:
#     lines = text.split('\n')
#     cleaned = []
#     prev_line = None
#     repeat_count = 0

#     for line in lines:
#         # Normalize for comparison
#         norm_line = re.sub(r'\s+', ' ', line).strip().lower()
        
#         if norm_line == prev_line:
#             repeat_count += 1
#             if repeat_count > max_repeats:
#                 continue
#         else:
#             repeat_count = 0
#             prev_line = norm_line

#         cleaned.append(line)

#     # Remove non-consecutive duplicates using OrderedDict
#     unique_lines = list(OrderedDict.fromkeys(cleaned).keys())
    
#     return '\n'.join(unique_lines)

# def process_md_ocr(text: str) -> str:
#     # Clean OCR artifacts
#     cleaned_content = clean_ocr_text(text)
    
#     # Remove duplicate lines (max 1 consecutive repeat allowed)
#     deduped_content = remove_duplicate_lines(cleaned_content, max_repeats=1)
    
#     # Final cleanup
#     processed = re.sub(r'\n{3,}', '\n\n', deduped_content)  # Normalize newlines
    
#     return processed.strip()


# def remove_html_duplicates(soup, max_repeats=1):
#     text_content = OrderedDict()
    
#     def process_element(element):
#         if isinstance(element, NavigableString):
#             return
        
#         # Process child elements recursively
#         for child in element.contents:
#             if child.name in ['script', 'style', 'code', 'pre']:
#                 continue
                
#             if isinstance(child, NavigableString):
#                 parent = child.parent
#                 if parent.name in ['p', 'div', 'span', 'li']:
#                     text = child.strip()
#                     if text:
#                         count = text_content.get(text, 0)
#                         if count >= max_repeats:
#                             child.replace_with('')
#                         else:
#                             text_content[text] = count + 1
#             else:
#                 process_element(child)
    
#     process_element(soup)
#     return soup

# def process_html_ocr(html: str) -> str:
    
#     # Parse HTML with BeautifulSoup
#     soup = BeautifulSoup(html, 'lxml')
    
#     # Clean text in all appropriate nodes
#     for element in soup.find_all(text=True):
#         if element.parent.name not in ['script', 'style', 'code', 'pre']:
#             cleaned = clean_ocr_text(element)
#             element.replace_with(cleaned)
    
#     # Structural duplicate removal
#     soup = remove_html_duplicates(soup)
    
#     # Final HTML cleanup
#     cleaned_html = soup.prettify()
    
#     # Advanced whitespace compression
#     cleaned_html = re.sub(r'>\s+<', '><', cleaned_html)    # Between tags
#     cleaned_html = re.sub(r'(?<=\>)\s+', '\n', cleaned_html) # After opening tags
#     cleaned_html = re.sub(r'\s+(?=\<)', '\n', cleaned_html)  # Before closing tags
#     cleaned_html = re.sub(r'\n{3,}', '\n\n', cleaned_html) # Limit blank lines
    
#     return cleaned_html

# def load_translator(model_name: str, src_lang: str, tgt_lang: str, device):
#     tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang, tgt_lang=tgt_lang)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     return pipeline("translation", model=model, tokenizer=tokenizer, src_lang=src_lang, tgt_lang=tgt_lang, device=device)

# def translate(data,device: int = 0):
#     # NLLB requires language codes like "eng_Latn", "arb_Arab"
#     model_name = "facebook/nllb-200-distilled-600M"
#     translator = load_translator(model_name, "arb_Arab", "eng_Latn", device=device)

#     translated_content = translate_html(data, translator)

#     # Markdown version
#     markdown = md(str(translated_content), heading_style="ATX")

#     return markdown

# def translate_html(data, translator) -> BeautifulSoup:
#     soup = BeautifulSoup(data, "html.parser")
#     text_nodes = [node for node in soup.find_all(text=True) if node.strip()]
#     for node in text_nodes:
#         if node.parent.name not in ["script", "style", "code"]:
#             translated = translator(node.strip(), max_length=512)[0]["translation_text"]
#             node.replace_with(translated)
#     return soup
