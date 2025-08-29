
from enum import Enum

class GraphBuildMethod(Enum):
    KNN = "knn"
    HIERARCHICAL = "hierarchical"
    APPROXIMATE = "approximate"
    MULTI_LEVEL = "multi_level"
    HYBRID = "hybrid"
    LSH = "lsh"  # Locality Sensitive Hashing
    SPECTRAL = "spectral"  # Spectral clustering