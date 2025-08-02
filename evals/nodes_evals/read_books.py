import asyncio
import json
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langsmith import Client

from evals.utils.constants import READ_BOOKS_GROUND_TRUTH_DATASET
from evals.utils.paths import READ_BOOKS_GROUND_TRUTH

from src.agent.states import InternalState
from src.agent.nodes import save_read_books
from src.agent.data_types import Book

load_dotenv()

client = Client()


def load_dataset() -> None:
    """
    Loads the dataset from the ground truth file and creates a LangSmith dataset if it doesn't exist.
    """
    with open(READ_BOOKS_GROUND_TRUTH, "r", encoding="utf-8") as f:
        examples = json.load(f)

    if not client.has_dataset(dataset_name=READ_BOOKS_GROUND_TRUTH_DATASET):
        dataset = client.create_dataset(dataset_name=READ_BOOKS_GROUND_TRUTH_DATASET)
        client.create_examples(
            inputs=[{"messages": ex["messages"]} for ex in examples],
            outputs=[{"expected_books": ex["expected_books"]} for ex in examples],
            dataset_id=dataset.id
        )


async def run_save_read_books(inputs: dict) -> dict:
    """
    Runs the save_read_books node for a given input.

    Args:
        inputs: A dictionary containing the input messages.

    Returns:
        A dictionary with the extracted books.
    """
    params = {
        "messages": [HumanMessage(content=inputs["messages"][0]["content"])],
        "recommended_books": [],
        "read_books": [],
        "preferences": []
    }

    result = save_read_books(InternalState(**params))

    # Extract the read_books from the result, or return empty list if none found
    extracted_books = result.get("read_books", [])

    # Convert Book objects to dictionaries for comparison
    books_as_dicts = []
    for book in extracted_books:
        if isinstance(book, Book):
            books_as_dicts.append({"name": book.name, "author": book.author})
        else:
            # If it's already a dict
            books_as_dicts.append(book)

    return {"extracted_books": books_as_dicts}


def exact_match_accuracy(outputs: dict, reference_outputs: dict) -> bool:
    """
    Calculates exact match accuracy for extracted books.

    Args:
        outputs: The outputs from the save_read_books node.
        reference_outputs: The reference outputs.

    Returns:
        True if the extracted books exactly match the expected books, False otherwise.
    """
    extracted = outputs.get("extracted_books", [])
    expected = reference_outputs.get("expected_books", [])

    # Sort both lists by name and author for consistent comparison
    extracted_sorted = sorted(extracted, key=lambda x: (x.get("name", ""), x.get("author", "")))
    expected_sorted = sorted(expected, key=lambda x: (x.get("name", ""), x.get("author", "")))

    return extracted_sorted == expected_sorted


def book_count_accuracy(outputs: dict, reference_outputs: dict) -> bool:
    """
    Calculates accuracy based on the number of books extracted.

    Args:
        outputs: The outputs from the save_read_books node.
        reference_outputs: The reference outputs.

    Returns:
        True if the number of extracted books matches the expected count, False otherwise.
    """
    extracted_count = len(outputs.get("extracted_books", []))
    expected_count = len(reference_outputs.get("expected_books", []))

    return extracted_count == expected_count


def partial_match_accuracy(outputs: dict, reference_outputs: dict) -> float:
    """
    Calculates partial match accuracy - percentage of expected books that were correctly extracted.

    Args:
        outputs: The outputs from the save_read_books node.
        reference_outputs: The reference outputs.

    Returns:
        Float between 0 and 1 representing the percentage of correctly extracted books.
    """
    extracted = outputs.get("extracted_books", [])
    expected = reference_outputs.get("expected_books", [])

    if not expected:
        return 1.0 if not extracted else 0.0

    # Count how many expected books were found in extracted books
    correct_count = 0
    for exp_book in expected:
        for ext_book in extracted:
            if (exp_book.get("name", "").lower().strip() == ext_book.get("name", "").lower().strip() and
                    exp_book.get("author", "").lower().strip() == ext_book.get("author", "").lower().strip()):
                correct_count += 1
                break

    return correct_count / len(expected)


def recall_score(outputs: dict, reference_outputs: dict) -> float:
    """
    Calculates recall - percentage of expected books that were found.
    Same as partial_match_accuracy but with clearer naming.

    Args:
        outputs: The outputs from the save_read_books node.
        reference_outputs: The reference outputs.

    Returns:
        Float between 0 and 1 representing recall.
    """
    return partial_match_accuracy(outputs, reference_outputs)


def precision_score(outputs: dict, reference_outputs: dict) -> float:
    """
    Calculates precision - percentage of extracted books that were actually expected.

    Args:
        outputs: The outputs from the save_read_books node.
        reference_outputs: The reference outputs.

    Returns:
        Float between 0 and 1 representing precision.
    """
    extracted = outputs.get("extracted_books", [])
    expected = reference_outputs.get("expected_books", [])

    if not extracted:
        return 1.0 if not expected else 0.0

    # Count how many extracted books were found in expected books
    correct_count = 0
    for ext_book in extracted:
        for exp_book in expected:
            if (ext_book.get("name", "").lower().strip() == exp_book.get("name", "").lower().strip() and
                    ext_book.get("author", "").lower().strip() == exp_book.get("author", "").lower().strip()):
                correct_count += 1
                break

    return correct_count / len(extracted)


def f1_score(outputs: dict, reference_outputs: dict) -> float:
    """
    Calculates F1 score - harmonic mean of precision and recall.

    Args:
        outputs: The outputs from the save_read_books node.
        reference_outputs: The reference outputs.

    Returns:
        Float between 0 and 1 representing F1 score.
    """
    precision = precision_score(outputs, reference_outputs)
    recall = recall_score(outputs, reference_outputs)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def author_accuracy(outputs: dict, reference_outputs: dict) -> float:
    """
    Calculates accuracy specifically for author extraction.

    Args:
        outputs: The outputs from the save_read_books node.
        reference_outputs: The reference outputs.

    Returns:
        Float between 0 and 1 representing author extraction accuracy.
    """
    extracted = outputs.get("extracted_books", [])
    expected = reference_outputs.get("expected_books", [])

    if not expected:
        return 1.0 if not extracted else 0.0

    # Count how many authors were correctly extracted
    correct_authors = 0
    total_authors = len(expected)

    for exp_book in expected:
        for ext_book in extracted:
            if (exp_book.get("name", "").lower().strip() == ext_book.get("name", "").lower().strip() and
                    exp_book.get("author", "").lower().strip() == ext_book.get("author", "").lower().strip()):
                correct_authors += 1
                break

    return correct_authors / total_authors


def title_accuracy(outputs: dict, reference_outputs: dict) -> float:
    """
    Calculates accuracy specifically for title extraction.

    Args:
        outputs: The outputs from the save_read_books node.
        reference_outputs: The reference outputs.

    Returns:
        Float between 0 and 1 representing title extraction accuracy.
    """
    extracted = outputs.get("extracted_books", [])
    expected = reference_outputs.get("expected_books", [])

    if not expected:
        return 1.0 if not extracted else 0.0

    # Count how many titles were correctly extracted (ignoring author)
    correct_titles = 0
    total_titles = len(expected)

    for exp_book in expected:
        for ext_book in extracted:
            if exp_book.get("name", "").lower().strip() == ext_book.get("name", "").lower().strip():
                correct_titles += 1
                break

    return correct_titles / total_titles


def complex_query_accuracy(outputs: dict, reference_outputs: dict) -> bool:
    """
    Calculates accuracy for complex queries (3+ books).

    Args:
        outputs: The outputs from the save_read_books node.
        reference_outputs: The reference outputs.

    Returns:
        True if it's a complex query and books were extracted correctly, False otherwise.
    """
    expected_count = len(reference_outputs.get("expected_books", []))

    # Only evaluate complex cases (3+ books)
    if expected_count < 3:
        return True  # Skip simple cases for this metric

    return exact_match_accuracy(outputs, reference_outputs)


async def evaluate_read_books() -> None:
    """
    Evaluates the read_books node using the ground truth dataset.
    """
    load_dataset()

    # Run the evaluation
    results = await client.aevaluate(
        run_save_read_books,
        data=READ_BOOKS_GROUND_TRUTH_DATASET,
        evaluators=[
            exact_match_accuracy,
            book_count_accuracy,
            partial_match_accuracy,
            recall_score,
            precision_score,
            f1_score,
            author_accuracy,
            title_accuracy,
            complex_query_accuracy
        ],
        experiment_prefix="read_books_evaluation"
    )


if __name__ == "__main__":
    asyncio.run(evaluate_read_books())
