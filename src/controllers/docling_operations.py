# import logging
# import os
# import sys
# import warnings
# warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true")

# # Project path setup for relative imports
# try:
#     MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
#     if MAIN_DIR not in sys.path:
#         sys.path.append(MAIN_DIR)
#     logging.debug("Main directory path configured: %s", MAIN_DIR)
# except (ImportError, OSError) as e:
#     logging.critical("Failed to set up main directory path: %s", e, exc_info=True)
#     sys.exit(1)

# from src.helpers import get_settings, Settings
# from src.infra import setup_logging

# app_settings: Settings = get_settings()
# logger = setup_logging()


# # from docling.document_converter import DocumentConverter
# # from docling.chunking import HybridChunker
# # def main():
# #     try:
# #         source = "https://arxiv.org/pdf/2408.09869"
# #         converter = DocumentConverter()
# #         result = converter.convert(source)
# #         chunker = HybridChunker()
# #         chunk_iter = chunker.chunk(dl_doc=result.document)
# #         print(chunk_iter)
# #         # for i, chunk in enumerate(chunk_iter):
# #         #     print(f"=== {i} ===")
# #         #     print(f"chunk.text:\n{f'{chunk.text[:300]}…'!r}")

# #         #     enriched_text = chunker.contextualize(chunk=chunk)
# #         #     print(f"chunker.contextualize(chunk):\n{f'{enriched_text[:300]}…'!r}")
# #     except Exception as e:
# #         logger.error("Failed to convert document: %s", e, exc_info=True)

# # if __name__ == "__main__":
# #     main()

# from pathlib import Path
# from docling.datamodel.base_models import InputFormat
# from docling.datamodel.pipeline_options import (
#     PdfPipelineOptions,
#     TesseractCliOcrOptions
# )
# from docling.document_converter import DocumentConverter, PdfFormatOption

# def main():
#     input_doc_path = "/home/alrashid/Desktop/PathRAG-LightRAG/assets/docs/ML/2209.03032v1.pdf"

#     pipeline_options = PdfPipelineOptions()
#     pipeline_options.do_ocr = True
#     pipeline_options.do_table_structure = True
#     pipeline_options.table_structure_options.do_cell_matching = True

#     # Any of the OCR options can be used:EasyOcrOptions, TesseractOcrOptions, TesseractCliOcrOptions, OcrMacOptions(Mac only), RapidOcrOptions
#     # ocr_options = EasyOcrOptions(force_full_page_ocr=True)
#     # ocr_options = TesseractOcrOptions(force_full_page_ocr=True)
#     # ocr_options = OcrMacOptions(force_full_page_ocr=True)
#     # ocr_options = RapidOcrOptions(force_full_page_ocr=True)
#     ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
#     pipeline_options.ocr_options = ocr_options

#     converter = DocumentConverter(
#         format_options={
#             InputFormat.PDF: PdfFormatOption(
#                 pipeline_options=pipeline_options,
#             )
#         }
#     )

#     doc = converter.convert(input_doc_path).document
#     md = doc.export_to_markdown()
#     print(md)

# if __name__ == "__main__":
#     main()