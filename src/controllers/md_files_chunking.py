#!/usr/bin/env python3
"""
Markdown Document Chunker Module.

This module provides functionality to split markdown documents into semantically
meaningful chunks while preserving document hierarchy and structure. It supports
multiple splitting strategies, batch processing of files in directories, and
detailed logging and error handling.

Author: AI Assistant
Date: 2025
Version: 1.0.0
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from tqdm import tqdm

try:
    # Adjust main directory to locate 'src' package
    MAIN_DIR = Path(__file__).resolve().parents[1]
    sys.path.append(str(MAIN_DIR))
except Exception as err:
    logging.basicConfig(level=logging.ERROR)
    logging.error(f"Failed to set main directory path: {err}")
    sys.exit(1)

try:
    from src.infra import setup_logging
    from src.utils import clear_markdown
except ImportError as err:
    logging.basicConfig(level=logging.ERROR)
    logging.error(f"Failed to import project modules: {err}")
    sys.exit(1)

logger = setup_logging(name="MD-FILES-CHUNKR")


class MarkdownChunker:
    """
    MarkdownChunker splits markdown documents into smaller chunks preserving
    the document's header structure and further splitting large sections into
    manageable chunks using LangChain's text splitters.

    Attributes:
        chunk_size (int): The target size of each chunk.
        chunk_overlap (int): The allowed overlap size between chunks.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        """
        Initialize the MarkdownChunker with splitting parameters.

        Args:
            chunk_size (int): Max size for chunks. Defaults to 500.
            chunk_overlap (int): Overlap between chunks. Defaults to 50.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
                ("#####", "Header 5"),
            ]
        )
        logger.debug(
            f"Initialized MarkdownChunker with chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )

    def process_file(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Process a single markdown file into chunks.

        Args:
            file_path (Union[str, Path]): Path to the markdown file.

        Returns:
            List[Dict[str, Any]]: List of chunk dictionaries containing metadata and chunk text.

        Raises:
            FileNotFoundError: If the file does not exist.
            UnicodeDecodeError: If file content cannot be decoded.
            Exception: For other unforeseen errors.
        """
        file_path = Path(file_path)

        if not file_path.exists() or not file_path.is_file():
            logger.error(f"File not found or not a file: {file_path}")
            raise FileNotFoundError(f"File not found or not a file: {file_path}")

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as err:
            logger.error(f"Error reading file {file_path}: {err}")
            raise

        if not content.strip():
            logger.warning(f"File {file_path} is empty. Skipping chunking.")
            return []

        chunks: List[Dict[str, Any]] = []
        try:
            # Step 1: Split document by markdown headers
            header_sections = self.header_splitter.split_text(content)

            for section in header_sections:
                title = " / ".join(
                    str(v).strip() for v in section.metadata.values() if v
                ) or "No Title"
                text = section.page_content.strip()

                if not text:
                    logger.debug(f"Skipping empty section under title: {title}")
                    continue

                # Step 2: If text is large, split further recursively
                if len(text) > self.chunk_size:
                    sub_chunks = self.recursive_splitter.split_text(text)
                    for sub_chunk in sub_chunks:
                        chunks.append(
                            {
                                "title": title,
                                "file_path": str(file_path.resolve()),
                                "chunk": sub_chunk,
                                "dir_name": file_path.parent.name,
                                "size": len(sub_chunk),
                            }
                        )
                else:
                    chunks.append(
                        {
                            "title": title,
                            "file_path": str(file_path.resolve()),
                            "chunk": text,
                            "dir_name": file_path.parent.name,
                            "size": len(text),
                        }
                    )
        except Exception as err:
            logger.error(f"Error during chunking file {file_path}: {err}")
            raise
        return chunks

    def process_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        pattern: str = "*.md",
    ) -> List[Dict[str, Any]]:
        """
        Process all markdown files in a directory and return all chunks.

        Args:
            directory_path (Union[str, Path]): Directory path to search markdown files.
            recursive (bool): Whether to search subdirectories recursively. Defaults to True.
            pattern (str): File matching pattern. Defaults to "*.md".

        Returns:
            List[Dict[str, Any]]: Aggregated list of chunk dictionaries from all files.

        Raises:
            NotADirectoryError: If the provided path is not a directory.
        """
        directory_path = Path(directory_path)
        logger.info(
            f"Processing directory: {directory_path} | Recursive: {recursive} | Pattern: {pattern}"
        )

        if not directory_path.is_dir():
            logger.error(f"Path is not a directory: {directory_path}")
            raise NotADirectoryError(f"Path is not a directory: {directory_path}")

        files = (
            list(directory_path.rglob(pattern)) if recursive else list(directory_path.glob(pattern))
        )
        logger.info(f"Found {len(files)} markdown files to process.")

        if not files:
            logger.warning(f"No markdown files found in directory: {directory_path}")
            return []

        all_chunks: List[Dict[str, Any]] = []

        for file in tqdm(files, desc="Processing markdown files", unit="file", colour="green"):
            try:
                file_chunks = self.process_file(file)
                all_chunks.extend(file_chunks)
            except Exception as err:
                logger.error(f"Failed to process file {file}: {err}")

        logger.info(f"Completed processing directory. Total chunks: {len(all_chunks)}")
        return all_chunks


def main() -> None:
    """
    Main entry point for running the MarkdownChunker standalone.
    Change `input_path` as needed before running.
    """
    input_path = Path(
        "/home/alrashid-ai/Desktop/PathRAG-LightRAG/assets/docs/Education & E-learning/English/"
    )

    chunker = MarkdownChunker(chunk_size=500, chunk_overlap=50)

    try:
        if input_path.is_file():
            logger.info(f"Input is a file: {input_path}")
            chunks = chunker.process_file(input_path)
        elif input_path.is_dir():
            logger.info(f"Input is a directory: {input_path}")
            chunks = chunker.process_directory(input_path, recursive=True)
        else:
            logger.error(f"Invalid input path: {input_path}")
            sys.exit(1)

        if not chunks:
            logger.warning("No chunks generated from input.")
            return

        # Example: Insert chunks into MongoDB - customize as per your db API
        try:
            from src.graph_db import get_mongo_client, insert_chunk_to_mongo
        except ImportError as err:
            logger.error(f"Failed to import database functions: {err}")
            return

        mongo_client = get_mongo_client()
        db = mongo_client["PathRAG-MongoDB"]
        logger.info("MongoDB client initialized.")

        # Insert chunks into MongoDB
        for chunk_data in tqdm(chunks, desc="Inserting chunks into MongoDB", unit="chunk", colour="blue"):
            try:
                insert_chunk_to_mongo(
                    db=db,
                    chunk=chunk_data.get("chunk"),
                    file=chunk_data.get("file_path"),
                    data_name=chunk_data.get("title"),
                    size=chunk_data.get("size"),
                )
            except Exception as err:
                logger.error(f"Failed to insert chunk into MongoDB: {err}")

        logger.info("All chunks processed and inserted.")
        if len(chunks) > 1:
            from pprint import pprint
            pprint(chunks[1])

    except Exception as err:
        logger.error(f"Unexpected error in main execution: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
