

from .pathrag  import PathRAG
from .light_rag import LightRAG

document = (
    "Machine Learning (ML) is a field of artificial intelligence focused on building systems that learn from data. "
    "It includes various types of learning: supervised, unsupervised, semi-supervised, and reinforcement learning. "
    "In supervised learning, models are trained on labeled data and learn to map inputs to outputs. "
    "Common algorithms include Linear Regression, Logistic Regression, Support Vector Machines (SVM), Random Forests, and Gradient Boosting Machines. "
    "Unsupervised learning is used when labels are not available. Algorithms like K-Means Clustering, DBSCAN, and Principal Component Analysis (PCA) are used to find hidden patterns and structures. "
    "Reinforcement learning (RL) involves training agents to make decisions by interacting with an environment and receiving rewards or penalties. "
    "RL has achieved remarkable success in games like Go and environments like OpenAI Gym."

    "Deep Learning is a subfield of ML that uses artificial neural networks with many layers (deep networks). "
    "It is especially powerful for unstructured data like images, audio, and natural language. "
    "Convolutional Neural Networks (CNNs) are specialized for image and video data. "
    "Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM), and Gated Recurrent Units (GRUs) are designed for sequential data such as time series and text. "
    "However, these models have largely been replaced by Transformer-based architectures, which allow for better parallelization and capture of long-term dependencies."

    "Transformers introduced the concept of self-attention, which allows the model to weigh the importance of each token in a sequence when making predictions. "
    "The Transformer architecture serves as the foundation of modern Large Language Models (LLMs), including BERT, GPT, T5, and LLaMA. "
    "BERT is a bidirectional encoder used mainly for understanding tasks like classification and named entity recognition, while GPT is a generative decoder-only model suited for text generation."

    "Large Language Models (LLMs) are deep learning models trained on massive corpora of internet text using billions of parameters. "
    "They are capable of zero-shot and few-shot learning, thanks to their scale and pretraining. "
    "Applications of LLMs include chatbots, summarization, translation, question answering, programming assistance, and creative writing. "
    "Popular LLMs include OpenAI's GPT-4, Google's Gemini, Meta's LLaMA, Anthropic's Claude, and Cohere's Command R+. "
    "Fine-tuning, prompt engineering, and instruction tuning are common strategies to adapt LLMs to domain-specific tasks or behaviors."

    "Retrieval-Augmented Generation (RAG) is an advanced technique that combines a retriever module with an LLM. "
    "Instead of relying solely on the model's internal knowledge, RAG retrieves relevant documents from external sources (e.g., vector databases like FAISS or Chroma) at query time. "
    "These documents are then used as additional context for the LLM to generate a more accurate and up-to-date response. "
    "This makes RAG especially useful for enterprise use cases, technical documentation, healthcare, finance, and legal domains where up-to-date and verifiable information is critical."

    "Lightweight RAG (LightRAG) is a simplified and efficient variation of RAG, suitable for scenarios where low latency and resource constraints are important. "
    "It typically uses smaller models and faster vector retrieval engines while maintaining the benefit of grounding responses in external data. "
    "LightRAG is often used in edge deployments, educational platforms, and local private assistants where deploying massive LLMs may not be feasible."
)