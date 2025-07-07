"""
Lightweight Retrieval-Augmented Generation (RAG) Pipeline

Demonstrates a retrieval-focused RAG system using efficient models suitable for CPU environments.

Features:
- Document ingestion with entity/relation extraction
- Dual-level retrieval (chunks + knowledge graph)
- Structured results without LLM generation
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

# Setup project base path
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
    logging.debug("Project base path set to: %s", MAIN_DIR)
except (ImportError, OSError) as e:
    logging.critical(
        "Failed to set up project base path: %s. System paths: %s", 
        e, sys.path
    )
    sys.exit(1)

from sentence_transformers import SentenceTransformer
from src.utils import setup_logging
from src.rag import LightRAG
from src.llms_providers import NERModel

logger = setup_logging()

def run_case_example():
    """Demonstrate the retrieval-focused RAG pipeline."""
    # Initialize system without LLM components
    rag = LightRAG()

    # Ingest the document
    document_path = Path("/home/alrashid/Desktop/PathRAG-LightRAG/asesst/sample.txt")
    try:
        entities_count, relations_count = rag.ingest_document(document_path)
        print(f"Ingested Document - Entities: {entities_count}, Relations: {relations_count}")
    except Exception as e:
        print(f"Document ingestion failed: {str(e)}")
        return

    # Ask questions and retrieve results
    questions = [
    # Fundamental Concepts
    "What is the difference between artificial intelligence, machine learning, and deep learning?",
    "Explain the bias-variance tradeoff in machine learning models",
    "What are the three main types of machine learning?",
    # Traditional Machine Learning
    "How does a decision tree algorithm work?",
    "What is the difference between bagging and boosting?",
    "Explain how support vector machines work for classification",
    "Why is feature engineering important in machine learning?",
    "What are the assumptions of linear regression?",
    "How does k-nearest neighbors algorithm make predictions?",
    "What is the curse of dimensionality in machine learning?",
    # Deep Learning Fundamentals
    "What is the basic structure of an artificial neural network?",
    "Explain the concept of backpropagation in neural networks",
    "What are activation functions and why are they important?",
    "How does dropout regularization work in neural networks?",
    "What is the vanishing gradient problem?",
    # CNN & Computer Vision
    "How do convolutional neural networks process image data?",
    "What are pooling layers in CNNs and why are they used?",
    "Explain how transfer learning works in computer vision",
    "What are some common architectures for image classification?",
    "How does object detection differ from image classification?",
    # RNN & NLP
    "What are the limitations of traditional RNNs?",
    "How do LSTMs solve the vanishing gradient problem?",
    "Explain the attention mechanism in neural networks",
    "What is word embedding in natural language processing?",
    "How does the transformer architecture work?",
    # LLM Fundamentals
    "What makes large language models different from traditional NLP models?",
    "Explain the concept of tokenization in LLMs",
    "What is the difference between autoregressive and autoencoding models?",
    "How does the GPT architecture generate text?",
    "What are the key components of the transformer architecture?",
    # LLM Training
    "What is pretraining in the context of LLMs?",
    "Explain the difference between pretraining and fine-tuning",
    "What is instruction tuning for language models?",
    "How does reinforcement learning from human feedback (RLHF) work?",
    "What is parameter-efficient fine-tuning?",
    # LLM Applications
    "How can LLMs be used for text summarization?",
    "What are some techniques for question answering with LLMs?",
    "Explain how retrieval-augmented generation (RAG) works",
    "How do LLMs handle code generation tasks?",
    "What are some challenges in multilingual NLP with LLMs?",
    # Ethics & Challenges
    "What are some ethical concerns with large language models?",
    "How can LLMs exhibit bias in their outputs?",
    "What is the problem of hallucination in LLMs?",
    "Explain the concept of AI alignment",
    "What are some techniques for making LLMs more interpretable?",
    # Optimization & Deployment
    "What techniques are used to optimize LLMs for inference?",
    "How does model quantization work for LLMs?",
    "What are some challenges in deploying LLMs in production?",
    "Explain the concept of prompt engineering",
    "What are some methods for reducing computational costs of LLMs?"
]

    for question in questions:
        try:
            result = rag.query(question)

            print("\n--- Retrieval Results ---")
            print(f"Question: {result['question']}")
            
            print("\nTop Relevant Text Chunks:")
            for i, chunk in enumerate(result['top_chunks'], 1):
                print(f"{i}. {chunk['text'][:100]}... (Score: {chunk['score']:.2f})")
            
            print("\nRelated Entities:")
            for i, entity in enumerate(result['top_entities'], 1):
                print(f"{i}. {entity['name']} ({entity['type']}) - {entity['description'][:50]}...")
            
            print("\nDiscovered Relationships:")
            for i, rel in enumerate(result['relations'], 1):
                print(f"{i}. {rel['source']} -> {rel['type']} -> {rel['target']}")
            
            print("\nFull Context Available:", len(result['combined_context'].split()), "words")
        
        except Exception as e:
            print(f"Error processing question '{question}': {str(e)}")

if __name__ == "__main__":
    run_case_example()