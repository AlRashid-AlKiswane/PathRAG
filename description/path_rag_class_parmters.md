### Parameters Explanation for `PathRAG` Initialization

```python
PathRAG(decay_rate=0.9, prune_thresh=0.3, sim_should=0.3)
```

| Parameter          | Type    | Default | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------ | ------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`decay_rate`**   | `float` | `0.8`   | The decay factor applied to the path flow score for each edge (hop) along a path. Controls how much longer paths are penalized. Must be between 0 and 1 (exclusive). <br><br> **Effect:** Higher values (closer to 1) mean less penalty for longer paths, allowing longer evidence chains to be more influential. Lower values (closer to 0) strongly penalize longer paths, favoring short, direct connections. <br><br> For example, `decay_rate=0.9` means the flow score is multiplied by `0.9^(number_of_edges)`. |
| **`prune_thresh`** | `float` | `0.01`  | The minimum flow score threshold for pruning paths during retrieval. Paths with flow scores below this value are discarded. <br><br> **Effect:** Higher thresholds make path pruning more aggressive, keeping only stronger, more relevant paths. Lower thresholds keep more paths, which may increase recall but also noise. <br><br> Example: `prune_thresh=0.3` means only paths with a flow score ≥ 0.3 are retained.                                                                                              |
| **`sim_should`**   | `float` | `0.1`   | The minimum cosine similarity threshold for creating edges between nodes in the semantic graph. Edges between chunks with similarity below this threshold are not created. <br><br> **Effect:** Higher values result in a sparser graph with fewer edges connecting only strongly similar chunks. Lower values create a denser graph with many weaker connections. <br><br> Example: `sim_should=0.3` means only chunk pairs with similarity > 0.3 get connected.                                                      |

---

### Summary

* **`decay_rate`** controls how the path relevance score decreases as the path length grows.
* **`prune_thresh`** filters out weak or less relevant paths to focus on the most meaningful chains.
* **`sim_should`** controls graph connectivity by deciding which chunk pairs are semantically close enough to link.

These parameters allow fine-tuning of the graph-based retrieval and reasoning behavior, balancing between recall, precision, and reasoning depth.