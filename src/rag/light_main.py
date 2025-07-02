"""

"""
import logging
import os
import sys
# Setup project base path
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
    logging.debug("Project base path set to: %s", MAIN_DIR)
except (ImportError, OSError) as e:
    logging.critical(
        "[Startup Critical] Failed to set up project base path. "
        "Error: %s. System paths: %s", e,
        exc_info=True
    )
    sys.exit(1)

import torch
from transformers import AutoModel, AutoTokenizer, pipeline

from src.rag import LightRAG
from src.utils import setup_logging

logger = setup_logging()

def load_models():
    """Load small, efficient models from Hugging Face"""
    # Use tiny models for CPU-friendly operation
    models = {
        # Small sentence transformer (17MB)
        "embedding": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        # Tiny NER model (43MB)
        "ner": {
            "model": "dslim/bert-base-NER",
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        # Small LLM (1.5GB - smallest practical for generation)
        "llm": {
            "model": "google/flan-t5-small",
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
    }
    
    logger.info("Loading models from Hugging Face...")
    
    # Load embedding model
    embedding_tokenizer = AutoTokenizer.from_pretrained(models["embedding"]["model"])
    embedding_model = AutoModel.from_pretrained(models["embedding"]["model"]).to(models["embedding"]["device"])
    
    # Load NER pipeline
    ner_pipeline = pipeline(
        "ner",
        model=models["ner"]["model"],
        tokenizer=models["ner"]["model"],
        device=models["ner"]["device"]
    )
    
    # Load LLM
    llm_tokenizer = AutoTokenizer.from_pretrained(models["llm"]["model"])
    llm_model = AutoModelForSeq2SeqLM.from_pretrained(models["llm"]["model"]).to(models["llm"]["device"])
    
    logger.info("All models loaded successfully")
    return {
        "embedding": (embedding_model, embedding_tokenizer),
        "ner": ner_pipeline,
        "llm": (llm_model, llm_tokenizer)
    }

def main():
    """
    LightRAG Demo with Small Hugging Face Models
    
    Demonstrates:
    - Loading efficient CPU-friendly models
    - Document processing
    - Knowledge graph construction
    - Query answering
    """
    logger = setup_logging()
    
    try:
        # Load models
        models = load_models()
        
        # Initialize RAG with small models
        rag = LightRAG(
            embedding_model=models["embedding"],
            ner_model=models["ner"],
            llm_model=models["llm"]
        )
        
        # Sample medical text (small for demonstration)
        document = (
            "Cardiology is the study of heart disorders. "
            "A cardiologist treats conditions like arrhythmia and heart failure. "
            "The heart pumps blood through arteries and veins."
        )
        
        # Process document
        logger.info("Processing document...")
        rag.process_document(document)
        
        # Sample queries
        queries = [
            "What does a cardiologist do?",
            "Name some heart conditions",
            "How does the heart work?"
        ]
        
        # Answer questions
        for query in queries:
            logger.info(f"Query: {query}")
            response = rag.query(query)
            print(f"Q: {query}\nA: {response}\n")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
