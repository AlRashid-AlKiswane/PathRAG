"""
Module: text_cleaner.py

Provides a robust TextCleaner class for preprocessing raw text for NLP applications,
especially suitable for RAG/PathRAG pipelines.

Cleans operations include:
- Unicode normalization
- HTML stripping
- Lowercasing (optional)
- URL removal
- File path removal (Unix/Windows style)
- Removal of bracketed citations (e.g., [1], [citation])
- Non-ASCII character filtering (e.g., emoji removal)
- Whitespace normalization and newline removal

Author: [Your Name]
Created: 2025-07-22
"""

import logging
import os
import sys
import re
import unicodedata
from bs4 import BeautifulSoup

# Setup project root path
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.getLogger(__name__).error("Failed to set main directory path: %s", e)
    sys.exit(1)

# pylint: disable=wrong-import-position
from src.infra import setup_logging
from src.helpers import get_settings, Settings

# Initialize logger
logger = setup_logging(name="TEXT-CLEANER")
app_settings: Settings = get_settings()


class TextCleaner:
    """
    A robust text cleaning utility for NLP preprocessing in RAG pipelines.

    This class provides functionality to clean raw input text by:
    - Removing HTML tags
    - Removing URLs and file paths
    - Normalizing Unicode characters
    - Removing non-ASCII characters (emojis, special symbols)
    - Removing citations and bracketed expressions
    - Collapsing whitespace and removing newlines
    """

    def __init__(self,
                 lowercase: bool = False,
                 remove_html: bool = True,
                 remove_non_ascii: bool = True):
        """
        Initialize the text cleaner with customizable options.

        Args:
            lowercase (bool): If True, converts all text to lowercase.
            remove_html (bool): If True, removes HTML tags using BeautifulSoup.
            remove_non_ascii (bool): If True, removes characters outside ASCII range (e.g., emojis).
        """
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.remove_non_ascii = remove_non_ascii

        logger.info("TextCleaner initialized with lowercase=%s, remove_html=%s, remove_non_ascii=%s",
                    lowercase, remove_html, remove_non_ascii)

    def clean(self, text: str) -> str:
        """
        Clean the input text by applying normalization, stripping tags, removing noise, etc.

        Args:
            text (str): The raw input text to be cleaned.

        Returns:
            str: Cleaned and normalized text.

        Raises:
            ValueError: If the input is not a string.
        """
        if not isinstance(text, str):
            logger.error("Input must be a string, got %s", type(text))
            raise ValueError("Input must be a string.")

        try:
            # Step 1: Unicode normalization
            text = unicodedata.normalize("NFKC", text)

            # Step 2: Optional HTML stripping
            if self.remove_html:
                text = BeautifulSoup(text, "html.parser").get_text()

            # Step 3: Lowercase (if configured)
            if self.lowercase:
                text = text.lower()

            # Step 4: Remove URLs
            text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

            # Step 5: Remove file paths (Unix & Windows)
            text = re.sub(r'([a-zA-Z]:)?(\/|\\)[\w\/\\\.\-]+', ' ', text)

            # Step 6: Remove bracketed citations like [1], [note]
            text = re.sub(r'\[\s*[^]]+\s*\]', ' ', text)

            # Step 7: Remove line breaks and tabs
            text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

            # Step 8: Remove non-ASCII (optional)
            if self.remove_non_ascii:
                text = re.sub(r'[^\x00-\x7F]+', ' ', text)

            # Step 9: Collapse redundant whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        except Exception as e:
            logger.exception("Text cleaning failed due to unexpected error.")
            raise RuntimeError(f"Failed during text cleaning: {e}") from e

if __name__ == "__main__":
    cleaner = TextCleaner(lowercase=True)
    sample_text = """
        Visit our site: https://example.com/docs.
        File saved at /home/user/project/file.txt
        <html><body><b>Example</b> content 😀</body></html>
        [1] Reference needed.
    """
    result = cleaner.clean(sample_text)
    print(result)
