from langgraph.graph import END

# Nodes
THINKING_NODE: str = "thinking_node"
SAVE_RECOMMENDED_BOOKS: str = "save_recommended_books"

# Initial router tags
INITIAL_ROUTER_TAGS: dict = {
    "recommendation": SAVE_RECOMMENDED_BOOKS,
    "talk": THINKING_NODE,
    "end": END
}
