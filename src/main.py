

from rag import PathRAG
from utils import setup_logging

logger = setup_logging()





def main():
    document = """
    Canada is a country in North America. It has ten provinces and three territories.
    Ottawa is the capital city of Canada. The country is known for its natural beauty.
    """

    # Improved chunking to ensure better connectivity
    chunks = [
        "Canada is a country in North America with ten provinces and three territories.",
        "Ottawa is the capital city of Canada.",
        "Canada is known for its natural beauty and vast landscapes."
    ]

    query = "What is the capital of Canada?"

    try:
        # Initialize with adjusted parameters for better connectivity
        path_rag = PathRAG(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            decay_rate=0.7,  # Lower decay for longer paths
            prune_thresh=0.05,  # Lower threshold to keep more paths
            sim_should=0.2  # Higher threshold for semantic similarity

        )

        # Build graph with error handling
        path_rag.build_graph(chunks)
        logger.info(f"Graph built with {path_rag.g.number_of_nodes()} nodes and {path_rag.g.number_of_edges()} edges")

        # Visualize connections for debugging
        for u, v, data in path_rag.g.edges(data=True):
            logger.debug(f"Edge {u}->{v} with weight {data['weight']:.3f}")

        # Retrieve more nodes to increase path possibilities
        top_nodes = path_rag.retrieve_nodes(query, top_k=min(4, len(chunks)))
        logger.info(f"Retrieved nodes: {top_nodes}")

        # Try different hop counts if no paths found
        max_hops_options = [2, 3]
        scored_paths = []

        for max_hops in max_hops_options:
            paths = path_rag.prune_paths(top_nodes, max_hops=max_hops)
            logger.info(f"Found {len(paths)} paths with {max_hops} hops")

            if paths:
                scored_paths = path_rag.score_paths(paths)
                break

        if not scored_paths:
            logger.warning("No valid paths found. Using fallback strategy...")
            # Fallback to direct node text if no paths
            prompt = f"QUERY: {query}\nDIRECT RESULT: {path_rag.g.nodes[top_nodes[0]]['text']}"
        else:
            prompt = path_rag.generate_prompt(query, scored_paths)

        print("\n=== FINAL OUTPUT ===")
        print(prompt)

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()