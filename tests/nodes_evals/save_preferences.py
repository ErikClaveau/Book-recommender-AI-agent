import json
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langsmith import Client

from tests.utils.constants import SAVE_PREFERENCES_GROUND_TRUTH_DATASET
from tests.utils.paths import SAVE_PREFERENCES_GROUND_TRUTH
from tests.utils.evaluation_prompts import PREFERENCES_MATCH_EVALUATION_PROMPT

from app.graph.states import InternalState
from app.graph.nodes import save_preferences
from app.utils.logger import get_logger

load_dotenv()

client = Client()
logger = get_logger(__name__)


def load_dataset() -> None:
    """
    Loads the dataset from the ground truth file and creates a LangSmith dataset if it doesn't exist.
    """
    with open(SAVE_PREFERENCES_GROUND_TRUTH, "r", encoding="utf-8") as f:
        examples = json.load(f)

    if not client.has_dataset(dataset_name=SAVE_PREFERENCES_GROUND_TRUTH_DATASET):
        dataset = client.create_dataset(dataset_name=SAVE_PREFERENCES_GROUND_TRUTH_DATASET)
        client.create_examples(
            inputs=[{"messages": ex["messages"]} for ex in examples],
            outputs=[{"expected_preferences": ex["expected_preferences"]} for ex in examples],
            dataset_id=dataset.id
        )


def run_save_preferences(inputs: dict) -> dict:
    """
    Runs the save_preferences node for a given input.

    Args:
        inputs: A dictionary containing the input messages.

    Returns:
        A dictionary with the extracted preferences.
    """
    params = {
        "messages": [HumanMessage(content=inputs["messages"][0]["content"])],
        "recommended_books": [],
        "read_books": [],
        "preferences": [],
    }

    result = save_preferences(InternalState(**params))

    # Extract preferences from the result, or return empty list if not found
    extracted_preferences = result.get("preferences", [])

    return {"extracted_preferences": extracted_preferences}


def semantic_preferences_match(outputs: dict, reference_outputs: dict) -> bool:
    """
    Evaluates whether extracted preferences semantically capture the expected preferences using an LLM as judge.

    Args:
        outputs: The outputs from the save_preferences node.
        reference_outputs: The reference outputs.

    Returns:
        True if extracted preferences semantically capture the expected ones, False otherwise.
    """
    extracted = outputs.get("extracted_preferences", [])
    expected = reference_outputs.get("expected_preferences", [])

    # If there are no expected or extracted preferences, it's correct
    if not expected and not extracted:
        return True

    # If there are expected preferences but no extracted ones, or vice versa, it's incorrect
    if not expected or not extracted:
        return False

    # Use an LLM to evaluate semantic matching
    evaluator = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Get the original user message
    user_message = "N/A"  # Default, although ideally we would have access to the original message

    prompt = PREFERENCES_MATCH_EVALUATION_PROMPT.format(
        user_message=user_message,
        extracted_preferences="\n".join(f"- {pref}" for pref in extracted),
        expected_preferences="\n".join(f"- {pref}" for pref in expected)
    )

    response = evaluator.invoke([HumanMessage(content=prompt)])

    # Parse the response to get YES/NO
    answer = response.content.strip().upper()
    return "YES" in answer


def partial_match_accuracy(outputs: dict, reference_outputs: dict) -> float:
    """
    Calculates partial match accuracy for extracted preferences.

    Args:
        outputs: The outputs from the save_preferences node.
        reference_outputs: The reference outputs.

    Returns:
        Partial accuracy score between 0 and 1.
    """
    extracted = outputs.get("extracted_preferences", [])
    expected = reference_outputs.get("expected_preferences", [])

    if not expected:
        return 1.0 if not extracted else 0.0

    # Normalize for comparison
    extracted_normalized = [pref.strip().lower() for pref in extracted]
    expected_normalized = [pref.strip().lower() for pref in expected]

    matched_preferences = 0
    for exp_pref in expected_normalized:
        for ext_pref in extracted_normalized:
            # Look for exact or partial matches (one contains the other)
            if (exp_pref == ext_pref or
                exp_pref in ext_pref or
                ext_pref in exp_pref):
                matched_preferences += 1
                break

    return matched_preferences / len(expected)


def preferences_count_accuracy(outputs: dict, reference_outputs: dict) -> bool:
    """
    Verifies if the number of extracted preferences matches the expected count.

    Args:
        outputs: The outputs from the save_preferences node.
        reference_outputs: The reference outputs.

    Returns:
        True if the number of preferences matches, False otherwise.
    """
    extracted = outputs.get("extracted_preferences", [])
    expected = reference_outputs.get("expected_preferences", [])

    return len(extracted) == len(expected)


def has_preferences_detected(outputs: dict, reference_outputs: dict) -> bool:
    """
    Verifies if preferences were detected when expected.

    Args:
        outputs: The outputs from the save_preferences node.
        reference_outputs: The reference outputs.

    Returns:
        True if preferences were detected when there were expected preferences, False otherwise.
    """
    extracted = outputs.get("extracted_preferences", [])
    expected = reference_outputs.get("expected_preferences", [])

    # If preferences were expected, there must be at least one extracted
    if expected:
        return len(extracted) > 0
    # If no preferences were expected, none should be extracted
    else:
        return len(extracted) == 0


def evaluate_save_preferences() -> None:
    """
    Runs the evaluation of the save preferences node using LangSmith.
    """
    results = client.evaluate(
        run_save_preferences,
        data=SAVE_PREFERENCES_GROUND_TRUTH_DATASET,
        evaluators=[
            semantic_preferences_match,
            partial_match_accuracy,
            preferences_count_accuracy,
            has_preferences_detected,
        ],
        experiment_prefix="save-preferences-eval",
        num_repetitions=1,
    )

    logger.info("Preferences evaluation completed:")
    logger.info(f"Semantic preferences match: {results['semantic_preferences_match']}")
    logger.info(f"Partial match accuracy: {results['partial_match_accuracy']}")
    logger.info(f"Preferences count accuracy: {results['preferences_count_accuracy']}")
    logger.info(f"Preferences detection: {results['has_preferences_detected']}")


if __name__ == "__main__":
    load_dataset()
    evaluate_save_preferences()
