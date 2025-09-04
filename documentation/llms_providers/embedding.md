# Embedding Module (`embedding.py`)

---

## Overview

The **Embedding Module** provides a wrapper class, `HuggingFaceModel`, around HuggingFace’s `SentenceTransformer` models.  

It enables **high-quality, dense semantic embeddings** for single texts or batches of texts, which are essential for:

- Semantic search  
- Retrieval-Augmented Generation (RAG) pipelines  
- Graph-based reasoning  
- Any NLP pipeline requiring consistent vector representations  

The module focuses on **robustness, flexibility, and operational transparency** by providing:

- Automatic GPU/CPU device selection  
- Config-driven model loading  
- Optional normalization and tensor conversion  
- Detailed logging of embedding operations  
- Error handling for failed embeddings  

---

## Dependencies

- `sentence-transformers` → Core embedding model library  
- `numpy` → Data manipulation and tensor-to-array conversion  
- `torch` → Optional backend for GPU acceleration  
- `logging` → Operational transparency  
- `src.infra.setup_logging` → Custom logger initialization  
- `src.helpers.get_settings` → Configuration-driven model selection  

Install the required packages via:

```bash
pip install sentence-transformers torch numpy
````

---

## Class: `HuggingFaceModel`

The `HuggingFaceModel` class encapsulates all operations related to **loading a SentenceTransformer model** and **generating embeddings**.

---

### Initialization

```python
model = HuggingFaceModel(model_name="all-MiniLM-L6-v2")
```

* **Parameters:**

  * `model_name` *(Optional\[str])* → Name of the pre-trained SentenceTransformer model.
    If `None`, the model name is loaded from `app_settings.EMBEDDING_MODEL`.

* **Behavior:**

  * Loads the model on GPU if available; otherwise, falls back to CPU.
  * Initializes a logger for embedding operations.
  * Raises an exception if model initialization fails.

* **Logging:**

  * INFO: Successful model initialization with device info
  * ERROR: Initialization failure with exception details

---

### Method: `embed_texts()`

Generates embeddings for single or multiple text inputs.

```python
embedding = model.embed_texts(
    texts="This is a test sentence.",
    convert_to_tensor=False,
    convert_to_numpy=True,
    normalize_embeddings=False,
    show_progress_bar=True
)
```

* **Parameters:**

  * `texts` *(str or List\[str])* → Input text(s) to embed.
  * `convert_to_tensor` *(bool, default=False)* → Return a PyTorch tensor if `True`.
  * `convert_to_numpy` *(bool, default=True)* → Return a NumPy array if `True`.
  * `normalize_embeddings` *(bool, default=False)* → Apply L2 normalization if `True`.
  * `show_progress_bar` *(bool, default=True)* → Display progress bar during batch embedding.

* **Returns:**

  * `np.ndarray` or tensor of embeddings
  * `None` if embedding generation fails

* **Error Handling:**

  * Raises `ValueError` if `texts` is empty or invalid
  * Logs detailed errors in case of embedding failure

* **Notes:**

  * If `convert_to_numpy` is `True` and output is a PyTorch tensor, the tensor is automatically moved to CPU and converted to NumPy.
  * Ensures consistent output format for downstream pipelines.

---

### Method: `get_model_info()`

Retrieves metadata about the loaded SentenceTransformer model.

```python
info = model.get_model_info()
```

* **Returns:** Dictionary with keys:

  * `model_name` → Name of the loaded model
  * `max_seq_length` → Maximum sequence length supported by the model
  * `embedding_dimension` → Dimensionality of the embedding vector

* **Logging:**

  * INFO: Model metadata retrieval

* **Example Output:**

```json
{
  "model_name": "all-MiniLM-L6-v2",
  "max_seq_length": 256,
  "embedding_dimension": 384
}
```

---

## Usage Example

```python
from embedding import HuggingFaceModel

# Initialize model
model = HuggingFaceModel(model_name="all-MiniLM-L6-v2")

# Generate embedding for a single sentence
embedding = model.embed_texts("This is a test sentence.")

# Generate embeddings for a batch of sentences
texts = ["Sentence 1", "Sentence 2", "Sentence 3"]
batch_embeddings = model.embed_texts(
    texts, normalize_embeddings=True, show_progress_bar=False
)

# Retrieve model information
info = model.get_model_info()
print(info)
```

---

## Notes

* GPU acceleration is automatically enabled if `torch.cuda.is_available()` returns `True`.
* Supports both **single-string** and **list-of-strings** input seamlessly.
* Embeddings can be normalized for **cosine similarity** calculations in semantic search pipelines.
* Logging provides operational transparency, helpful for debugging or monitoring batch embedding pipelines.

---

## Best Practices

1. **Batch Processing:**
   Large text lists should be embedded in batches to optimize GPU memory usage.

2. **Normalization:**
   Use `normalize_embeddings=True` if embeddings will be compared with cosine similarity.

3. **Device Management:**
   Ensure GPU memory availability when embedding large datasets.

4. **Error Handling:**
   Always check for `None` return values in case of embedding failures.

5. **Configuration:**
   Use `app_settings.EMBEDDING_MODEL` for centralized model selection in production pipelines.

---

## Common Errors

| Error                                                         | Cause                                  | Resolution                                                |
| ------------------------------------------------------------- | -------------------------------------- | --------------------------------------------------------- |
| `ValueError: Input texts for embedding are empty or invalid.` | Empty or `None` input                  | Provide a non-empty string or list                        |
| `Failed to initialize SentenceTransformer model`              | Invalid model name or download failure | Verify model name or internet connectivity                |
| `Failed to generate embedding`                                | Runtime error during encoding          | Inspect logs; reduce batch size if GPU memory is exceeded |

---

## Conclusion

The `embedding.py` module provides a **robust, flexible, and production-ready** interface for generating semantic embeddings using HuggingFace SentenceTransformer models.

It ensures:

* Seamless GPU/CPU operation
* Optional tensor or NumPy outputs
* L2 normalization for similarity pipelines
* Centralized configuration and logging

This module is ideal for integration into **semantic search, RAG systems, knowledge graphs, and ML pipelines** that require **consistent and reliable embeddings**.

```

