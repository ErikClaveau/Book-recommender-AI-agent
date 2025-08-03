import asyncio
import json
from typing import List, Dict
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langsmith import Client

from evals.utils.constants import RECOMMEND_BOOKS_GROUND_TRUTH_DATASET
from evals.utils.paths import RECOMMEND_BOOKS_GROUND_TRUTH
from evals.utils.evaluation_prompts import (
    GENRE_RELEVANCE_EVALUATION_PROMPT,
    UNWANTED_GENRES_EVALUATION_PROMPT,
    CONTEXTUAL_RELEVANCE_EVALUATION_PROMPT
)

from src.agent.states import InternalState
from src.agent.nodes import save_recommended_books
from src.agent.data_types import Book

load_dotenv()

client = Client()


def _create_judge_llm() -> ChatOpenAI:
    """
    Creates a consistent LLM instance for evaluation tasks.

    Returns:
        ChatOpenAI: Configured LLM for evaluation.
    """
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


def _format_books_for_evaluation(books: List[Dict[str, str]]) -> str:
    """
    Formats a list of books for LLM evaluation.

    Args:
        books: List of book dictionaries with 'name' and 'author' keys.

    Returns:
        Formatted string of books for evaluation.
    """
    if not books:
        return "No books were recommended."

    return "\n".join([
        f"- {book.get('name', 'Unknown')} by {book.get('author', 'Unknown')}"
        for book in books
    ])


def _evaluate_with_llm(prompt: str) -> bool:
    """
    Evaluates a prompt using LLM and returns a boolean result.

    Args:
        prompt: The evaluation prompt to send to the LLM.

    Returns:
        True if LLM responds with "YES", False otherwise or on error.
    """
    try:
        judge_llm = _create_judge_llm()
        response = judge_llm.invoke([HumanMessage(content=prompt)])
        answer = response.content.strip().upper()
        return answer == "YES"
    except Exception:
        # If LLM call fails, return False (conservative approach)
        return False


def load_dataset() -> None:
    """
    Loads the dataset from the ground truth file and creates a LangSmith dataset if it doesn't exist.
    """
    with open(RECOMMEND_BOOKS_GROUND_TRUTH, "r", encoding="utf-8") as f:
        examples = json.load(f)

    if not client.has_dataset(dataset_name=RECOMMEND_BOOKS_GROUND_TRUTH_DATASET):
        dataset = client.create_dataset(dataset_name=RECOMMEND_BOOKS_GROUND_TRUTH_DATASET)
        client.create_examples(
            inputs=[{
                "messages": ex["messages"], 
                "context": ex["context"]
            } for ex in examples],
            outputs=[{"expected_criteria": ex["expected_criteria"]} for ex in examples],
            dataset_id=dataset.id
        )


async def run_save_recommended_books(inputs: dict) -> dict:
    """
    Runs the save_recommended_books node for a given input.

    Args:
        inputs: A dictionary containing the input messages and context.

    Returns:
        A dictionary with the recommended books and additional context for evaluation.
    """
    context = inputs.get("context", {})
    user_message = inputs["messages"][0]["content"]

    params = {
        "messages": [HumanMessage(content=user_message)],
        "recommended_books": [Book(**book) for book in context.get("recommended_books", [])],
        "read_books": [Book(**book) for book in context.get("read_books", [])],
        "preferences": context.get("preferences", [])
    }

    result = save_recommended_books(InternalState(**params))

    # Extract the recommended_books from the result, or return empty list if none found
    recommended_books = result.get("recommended_books", [])

    # Convert Book objects to dictionaries for analysis
    books_as_dicts = []
    for book in recommended_books:
        if isinstance(book, Book):
            books_as_dicts.append({"name": book.name, "author": book.author})
        else:
            # If it's already a dict
            books_as_dicts.append(book)

    return {
        "recommended_books": books_as_dicts,
        "user_request": user_message,
        "context": context
    }


def recommendation_count_accuracy(outputs: dict, reference_outputs: dict) -> bool:
    """
    Checks if the number of recommendations is within the expected range.

    Args:
        outputs: The outputs from the save_recommended_books node.
        reference_outputs: The reference outputs containing expected criteria.

    Returns:
        True if the recommendation count is within the expected range, False otherwise.
    """
    recommended = outputs.get("recommended_books", [])
    criteria = reference_outputs.get("expected_criteria", {})
    
    min_recs = criteria.get("min_recommendations", 1)
    max_recs = criteria.get("max_recommendations", 10)
    
    count = len(recommended)
    return min_recs <= count <= max_recs


def has_recommendations(outputs: dict, reference_outputs: dict) -> bool:
    """
    Checks if any recommendations were generated at all.

    Args:
        outputs: The outputs from the save_recommended_books node.
        reference_outputs: The reference outputs.

    Returns:
        True if at least one recommendation was generated, False otherwise.
    """
    recommended = outputs.get("recommended_books", [])
    return len(recommended) > 0


def genre_relevance_score(outputs: dict, reference_outputs: dict) -> bool:
    """
    Uses an LLM as a judge to evaluate if the recommendations match the expected genre relevance.

    Args:
        outputs: The outputs from the save_recommended_books node.
        reference_outputs: The reference outputs containing expected criteria.

    Returns:
        True if recommendations are genre-relevant, False otherwise.
    """
    recommended = outputs.get("recommended_books", [])
    criteria = reference_outputs.get("expected_criteria", {})
    relevant_genres = criteria.get("genre_relevance", [])

    if not recommended or not relevant_genres:
        return False

    # Format the recommendations for evaluation
    books_text = _format_books_for_evaluation(recommended)

    # Create evaluation prompt using the template
    evaluation_prompt = GENRE_RELEVANCE_EVALUATION_PROMPT.format(
        requested_genres=', '.join(relevant_genres),
        recommended_books=books_text
    )

    return _evaluate_with_llm(evaluation_prompt)


def avoids_unwanted_genres(outputs: dict, reference_outputs: dict) -> bool:
    """
    Uses an LLM as a judge to check if recommendations avoid genres that should be avoided.

    Args:
        outputs: The outputs from the save_recommended_books node.
        reference_outputs: The reference outputs containing expected criteria.

    Returns:
        True if unwanted genres are successfully avoided, False otherwise.
    """
    recommended = outputs.get("recommended_books", [])
    criteria = reference_outputs.get("expected_criteria", {})
    avoid_genres = criteria.get("should_avoid_genres", [])

    if not recommended or not avoid_genres:
        return True  # If no restrictions or no recommendations, return True

    # Format the recommendations for evaluation
    books_text = _format_books_for_evaluation(recommended)

    # Create evaluation prompt using the template
    evaluation_prompt = UNWANTED_GENRES_EVALUATION_PROMPT.format(
        unwanted_genres=', '.join(avoid_genres),
        recommended_books=books_text
    )

    return _evaluate_with_llm(evaluation_prompt)


def recommendation_diversity(outputs: dict, reference_outputs: dict) -> float:
    """
    Measures the diversity of recommendations (different authors).

    Args:
        outputs: The outputs from the save_recommended_books node.
        reference_outputs: The reference outputs.

    Returns:
        Float between 0 and 1 representing recommendation diversity.
    """
    recommended = outputs.get("recommended_books", [])
    
    if len(recommended) <= 1:
        return 1.0  # Single or no recommendation gets perfect diversity score
    
    # Count unique authors
    authors = set()
    for book in recommended:
        author = book.get("author", "").strip()
        if author:
            authors.add(author.lower())
    
    # Diversity = unique authors / total books
    return len(authors) / len(recommended) if len(recommended) > 0 else 0.0


def contextual_relevance(outputs: dict, reference_outputs: dict) -> bool:
    """
    Uses an LLM as a judge to check if recommendations are contextually relevant based on user's reading history.

    Args:
        outputs: The outputs from the save_recommended_books node.
        reference_outputs: The reference outputs.

    Returns:
        True if recommendations are contextually appropriate, False otherwise.
    """
    recommended = outputs.get("recommended_books", [])
    user_request = outputs.get("user_request", "")
    context = outputs.get("context", {})

    if not recommended:
        return False

    # Format the data for evaluation
    read_books = context.get("read_books", [])
    preferences = context.get("preferences", [])

    read_books_text = _format_books_for_evaluation(read_books) if read_books else "No books read previously."
    preferences_text = ', '.join(preferences) if preferences else "No stated preferences."
    books_text = _format_books_for_evaluation(recommended)

    # Create evaluation prompt using the template
    evaluation_prompt = CONTEXTUAL_RELEVANCE_EVALUATION_PROMPT.format(
        read_books=read_books_text,
        preferences=preferences_text,
        user_request=user_request,
        recommended_books=books_text
    )

    return _evaluate_with_llm(evaluation_prompt)


def book_title_quality(outputs: dict, reference_outputs: dict) -> float:
    """
    Checks if recommended books have proper titles (not empty, reasonable length).

    Args:
        outputs: The outputs from the save_recommended_books node.
        reference_outputs: The reference outputs.

    Returns:
        Float between 0 and 1 representing title quality.
    """
    recommended = outputs.get("recommended_books", [])
    
    if not recommended:
        return 0.0
    
    valid_titles = 0
    for book in recommended:
        title = book.get("name", "").strip()
        # Check if title exists and is reasonable length (between 1 and 200 characters)
        if title and 1 <= len(title) <= 200:
            valid_titles += 1
    
    return valid_titles / len(recommended)


def author_inclusion_rate(outputs: dict, reference_outputs: dict) -> float:
    """
    Checks if recommended books include author information.

    Args:
        outputs: The outputs from the save_recommended_books node.
        reference_outputs: The reference outputs.

    Returns:
        Float between 0 and 1 representing author inclusion rate.
    """
    recommended = outputs.get("recommended_books", [])
    
    if not recommended:
        return 0.0
    
    books_with_authors = 0
    for book in recommended:
        author = book.get("author", "").strip()
        if author:
            books_with_authors += 1
    
    return books_with_authors / len(recommended)


def overall_recommendation_quality(outputs: dict, reference_outputs: dict) -> float:
    """
    Combines multiple metrics to give an overall quality score.

    Args:
        outputs: The outputs from the save_recommended_books node.
        reference_outputs: The reference outputs.

    Returns:
        Float between 0 and 1 representing overall quality.
    """
    # Weight different metrics
    weights = {
        "genre_relevance": 0.3,
        "count_accuracy": 0.2,
        "diversity": 0.2,
        "title_quality": 0.15,
        "author_inclusion": 0.15
    }
    
    scores = {
        "genre_relevance": genre_relevance_score(outputs, reference_outputs),
        "count_accuracy": 1.0 if recommendation_count_accuracy(outputs, reference_outputs) else 0.0,
        "diversity": recommendation_diversity(outputs, reference_outputs),
        "title_quality": book_title_quality(outputs, reference_outputs),
        "author_inclusion": author_inclusion_rate(outputs, reference_outputs)
    }
    
    weighted_score = sum(weights[metric] * scores[metric] for metric in weights)
    return weighted_score


async def evaluate_recommend_books() -> None:
    """
    Evaluates the recommend_books node using the ground truth dataset.
    """
    load_dataset()
    
    # Run the evaluation
    results = await client.aevaluate(
        run_save_recommended_books,
        data=RECOMMEND_BOOKS_GROUND_TRUTH_DATASET,
        evaluators=[
            recommendation_count_accuracy,
            has_recommendations,
            genre_relevance_score,
            avoids_unwanted_genres,
            recommendation_diversity,
            contextual_relevance,
            book_title_quality,
            author_inclusion_rate,
            overall_recommendation_quality
        ],
        experiment_prefix="recommend_books_evaluation"
    )
    
    return results


if __name__ == "__main__":
    asyncio.run(evaluate_recommend_books())
