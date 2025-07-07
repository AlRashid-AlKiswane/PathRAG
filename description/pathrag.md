## 🔍 Key Concepts

### **Graph (Directed Graph with Chunks)**

A graph consists of:

* **Nodes** (vertices): represent **text chunks**.
* **Edges**: represent **semantic similarity** between chunks, with a **weight** equal to the similarity score.

### **Path**

A **path** in a graph is a sequence of connected nodes (chunks) such that you can go from one chunk to another through intermediate chunks.

Example:

```text
Node A → Node B → Node C
```

This is a **path of 3 nodes**, meaning **2 hops**.

### **Hop**

A **hop** is a step from one node to another via an edge.

* Path: `[0, 2, 5]` → has **2 hops**
* Hops = number of edges in the path = `len(path) - 1`

---

## 🧠 What is `PathRAG` doing?

### 🔹 Goal:

1. Embed all chunks using `SentenceTransformer`.
2. Build a **graph** of similar chunks using cosine similarity.
3. Given a query:

   * Embed it
   * Find the most similar nodes (chunks)
   * Search for all possible paths between these nodes (max 4 hops)
   * **Score** each path using decay + similarity
   * Generate a **prompt** with the query + top-ranked paths.

---

## 🧱 Step-by-step Code Breakdown

---

### `__init__` → Initialize the class

```python
self.embed_model = SentenceTransformer(embedding_model)
```

* Loads a sentence transformer model for creating embeddings.

```python
self.g = nx.DiGraph()
```

* Creates a **directed graph** to store nodes (text chunks) and weighted edges.

```python
self.decay_rate = decay_rate  # e.g., 0.8
self.prune_thresh = prune_thresh  # e.g., 0.01
self.sim_should = sim_should  # similarity threshold
```

---

### `build_graph(self, chunks)`

1. **Embeds each chunk**
2. **Adds nodes** to the graph with their embeddings
3. **Adds edges** between chunks with cosine similarity > `sim_should`

```python
self.g.add_node(idx, text=text, emb=emb)
```

```python
sim = cosine_similarity(embeddings[i], embeddings[j])
if sim > self.sim_should:
    self.g.add_edge(i, j, weight=sim)
```

* Adds a **bidirectional edge** (i.e., both `i→j` and `j→i`) if similarity is strong.

---

### `retrieve_nodes(self, query, top_k)`

1. **Embeds the user query**
2. **Finds top\_k most similar nodes (chunks)**

```python
q_emb = self.embed_model.encode(query)
similarity = cosine_similarity(q_emb, node_emb)
```

* Stores top similar nodes (chunks) as `top_nodes`.

---

### `prune_paths(self, nodes, max_hops=4)`

1. Finds **all paths** between pairs of `top_nodes` up to `max_hops`.
2. **Computes a “flow” score** for each path using:

   ```python
   flow = product of edge weights * (decay_rate ^ hops)
   ```

```python
for path in nx.all_simple_paths(self.g, source=u, target=v, cutoff=max_hops):
```

* It uses `networkx` to get all simple paths between `u` and `v` of length ≤ `max_hops`.

---

### `_compute_flow(self, path)`

Calculates the total strength of a path:

```python
flow = (weight1 * weight2 * ...) * (decay_rate ** hops)
```

* Each edge’s similarity is multiplied.
* The longer the path, the more it decays.

---

### `score_paths(self, paths)`

* Computes the flow score of each path.
* Sorts paths by score.

```python
scored.sort(key=lambda x: x["score"])
```

---

### `generate_prompt(self, query, scored_paths)`

* Builds the final **prompt** for the LLM:

```text
QUERY: What is AI?
RELATED EVIDENCE PATHS:
- [Score: 0.123] chunk1 → chunk2 → chunk3
```

* Each path is printed as a line with its flow score and texts.

---

## 📌 Example Workflow

```python
rag = PathRAG()
rag.build_graph(chunks)  # Create the graph
top_nodes = rag.retrieve_nodes("What is AI?")
paths = rag.prune_paths(top_nodes)
scored = rag.score_paths(paths)
prompt = rag.generate_prompt("What is AI?", scored)
print(prompt)
```

---

## ✅ Summary

| Concept  | Meaning                                                           |
| -------- | ----------------------------------------------------------------- |
| **Node** | A text chunk                                                      |
| **Edge** | A connection between two chunks with a similarity score           |
| **Path** | A sequence of connected chunks                                    |
| **Hop**  | A transition from one node to the next in a path                  |
| **Flow** | A score for how strong a path is, based on edge weights and decay |

---