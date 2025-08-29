
from dataclasses import dataclass


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint files."""
    processed_samples: int
    total_samples: int
    timestamp: str
    graph_nodes: int
    graph_edges: int

