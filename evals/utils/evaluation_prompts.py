"""
Prompts used for LLM-based evaluation in the book recommendation system.
"""

GENRE_RELEVANCE_EVALUATION_PROMPT = """You are evaluating book recommendations for genre relevance.

User requested books in these genres/themes: {requested_genres}

Recommended books:
{recommended_books}

Question: Do the recommended books match the requested genres/themes?

Instructions:
- Consider if the books are generally appropriate for the requested genres
- Don't be overly strict - books can fit multiple genres
- Focus on whether someone interested in the requested genres would likely enjoy these books
- Answer only "YES" or "NO"

Answer:"""

UNWANTED_GENRES_EVALUATION_PROMPT = """You are evaluating book recommendations to check if they avoid unwanted genres.

User wants to AVOID these genres: {unwanted_genres}

Recommended books:
{recommended_books}

Question: Do the recommended books successfully avoid the unwanted genres?

Instructions:
- Check if any of the recommended books clearly belong to the genres to avoid
- Be reasonable - some overlap is acceptable if the primary genre is different
- Focus on whether the recommendations respect the user's preferences
- Answer only "YES" or "NO"

Answer:"""

CONTEXTUAL_RELEVANCE_EVALUATION_PROMPT = """You are evaluating book recommendations for contextual relevance.

User's reading history:
{read_books}

User's stated preferences:
{preferences}

User's request: {user_request}

Recommended books:
{recommended_books}

Question: Are the recommended books contextually relevant given the user's history and preferences?

Instructions:
- Consider if the recommendations make sense given what the user has read and liked
- Check if they align with stated preferences
- Evaluate if they appropriately respond to the specific request
- Answer only "YES" or "NO"

Answer:"""
