from langgraph.graph import END

# Nodes
THINKING_NODE: str = "thinking_node"
SAVE_RECOMMENDED_BOOKS: str = "save_recommended_books_node"
SAVE_PREFERENCES: str = "save_preferences_node"
SAVE_READ_BOOKS: str = "read_books_node"

# Initial router tags
INITIAL_ROUTER_TAGS: dict = {
    "recommendation": SAVE_RECOMMENDED_BOOKS,
    "preferences": SAVE_PREFERENCES,
    "talk": THINKING_NODE,
    "read": SAVE_READ_BOOKS,
    "end": END
}
