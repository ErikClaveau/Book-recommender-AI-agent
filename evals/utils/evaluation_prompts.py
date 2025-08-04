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

Question: Are the recommended books contextually relevant to the user's reading history, preferences, and specific request?

Instructions:
- Consider how well the recommendations align with the user's demonstrated preferences
- Check if the books match the user's reading patterns and interests
- Evaluate if the recommendations respond appropriately to the specific request
- Answer only "YES" or "NO"

Answer:"""

PREFERENCES_MATCH_EVALUATION_PROMPT = """You are evaluating whether extracted preferences accurately capture the user's stated reading preferences.

User's original message:
{user_message}

Extracted preferences:
{extracted_preferences}

Expected preferences:
{expected_preferences}

Question: Do the extracted preferences accurately capture the user's stated reading preferences from their message?

Instructions:
- Consider semantic meaning, not just exact word matches
- Check if the extracted preferences capture the same concepts as the expected ones
- Allow for different phrasings that mean the same thing (e.g., "sci-fi" vs "science fiction")
- Focus on whether someone with the extracted preferences would have similar reading tastes to someone with the expected preferences
- Answer only "YES" or "NO"

Answer:"""

DATA_ACCURACY_EVALUATION_PROMPT = """You are evaluating the accuracy of data retrieval in a book assistant's response.

User's query: {user_query}

Available data:
- Read books: {read_books}
- Recommended books: {recommended_books}
- Preferences: {preferences}

Assistant's response: {response}

Question: Does the assistant's response accurately reflect the available data and correctly answer the user's query?

Instructions:
- Check if the response includes the correct books, authors, preferences, or other data elements
- Verify that the assistant doesn't hallucinate or make up information not in the data
- Ensure the response directly addresses what the user asked for
- Answer only "YES" or "NO"

Answer:"""

RESPONSE_COHERENCE_EVALUATION_PROMPT = """You are evaluating the coherence and helpfulness of a book assistant's response.

User's query: {user_query}

Assistant's response: {response}

Question: Is the assistant's response coherent, well-structured, and helpful in addressing the user's query?

Instructions:
- Check if the response is logically organized and easy to understand
- Verify that the response directly addresses the user's question
- Ensure the tone is appropriate and helpful
- Look for clear communication without unnecessary complexity
- Answer only "YES" or "NO"

Answer:"""

COMPLETENESS_EVALUATION_PROMPT = """You are evaluating the completeness of a book assistant's response.

User's query: {user_query}

Available data:
- Read books: {read_books}
- Recommended books: {recommended_books}
- Preferences: {preferences}

Assistant's response: {response}

Expected content type: {expected_content_type}

Question: Does the assistant's response provide complete information relevant to the user's query?

Instructions:
- Check if the response includes all relevant information available in the data
- Verify that the assistant doesn't omit important details that would help the user
- For queries asking for lists or summaries, ensure comprehensive coverage
- Consider whether the response fully satisfies the user's information need
- Answer only "YES" or "NO"

Answer:"""

QUERY_UNDERSTANDING_EVALUATION_PROMPT = """You are evaluating whether a book assistant correctly understood the user's query.

User's query: {user_query}

Assistant's response: {response}

Question: Did the assistant correctly understand and respond to what the user was asking for?

Instructions:
- Check if the response type matches what the user requested
- Verify that the assistant didn't misinterpret the query
- Ensure the response addresses the specific aspect of data the user asked about
- Look for appropriate handling of edge cases (empty data, specific filters, etc.)
- Answer only "YES" or "NO"

Answer:"""
