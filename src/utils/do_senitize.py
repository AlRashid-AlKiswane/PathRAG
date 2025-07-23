import numpy as np
from collections.abc import Mapping, Iterable

def sanitize(obj):
    """
    Recursively sanitize any object for JSON serialization.
    Converts NumPy types, sets, and custom objects to basic Python types.
    """
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [sanitize(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    elif isinstance(obj, Mapping):
        return dict(obj)
    elif hasattr(obj, "__dict__"):
        return sanitize(vars(obj))
    else:
        return obj
