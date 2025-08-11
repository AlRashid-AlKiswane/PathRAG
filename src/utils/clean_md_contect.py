# bin/evn/ python3
"""

"""
import re

def clear_markdown(text: str) -> str:
    """
    Remove Markdown formatting from the given text.
    
    Args:
        text (str): The Markdown-formatted string.
        
    Returns:
        str: The plain text without Markdown.
    """
    # Remove code blocks ```
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    
    # Remove inline code `...`
    text = re.sub(r"`([^`]*)`", r"\1", text)
    
    # Remove images ![alt](url)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    
    # Remove links [text](url)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    
    # Remove headings (e.g., ### Heading)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    
    # Remove bold and italic **, __, *, _
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
    
    # Remove blockquotes >
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    
    # Remove unordered list markers (-, *, +) and numbered lists
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    
    # Remove horizontal rules ---
    text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)
    
    # Collapse multiple newlines into one
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()
