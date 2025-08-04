import asyncio
import json
from typing import Dict, List
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langsmith import Client

from evals.utils.constants import TALK_WITH_DATA_GROUND_TRUTH_DATASET
from evals.utils.paths import TALK_WITH_DATA_GROUND_TRUTH
from evals.utils.evaluation_prompts import (
    DATA_ACCURACY_EVALUATION_PROMPT,
    RESPONSE_COHERENCE_EVALUATION_PROMPT,
    COMPLETENESS_EVALUATION_PROMPT,
    QUERY_UNDERSTANDING_EVALUATION_PROMPT
)

from src.agent.states import InternalState
from src.agent.nodes import thinking_node

load_dotenv()

client = Client()


def _create_judge_llm() -> ChatOpenAI:
    """
    Creates a consistent LLM instance for evaluation tasks.

    Returns:
        ChatOpenAI: Configured LLM for evaluation.
    """
    return ChatOpenAI(model="gpt-4", temperature=0)


def _format_data_for_evaluation(data: List[Dict]) -> str:
    """
    Formats a list of data items for LLM evaluation.

    Args:
        data: List of data dictionaries (books or preferences).

    Returns:
        Formatted string for evaluation.
    """
    if not data:
        return "None"

    if isinstance(data[0], dict) and 'name' in data[0]:
        # Format books
        return "\n".join([
            f"- {item.get('name', 'Unknown')} by {item.get('author', 'Unknown')}"
            for item in data
        ])
    else:
        # Format preferences or other string data
        return "\n".join([f"- {str(item)}" for item in data])


def load_dataset() -> None:
    """
    Loads the dataset from the ground truth file and creates a LangSmith dataset if it doesn't exist.
    """
    with open(TALK_WITH_DATA_GROUND_TRUTH, "r", encoding="utf-8") as f:
        examples = json.load(f)

    if not client.has_dataset(dataset_name=TALK_WITH_DATA_GROUND_TRUTH_DATASET):
        dataset = client.create_dataset(dataset_name=TALK_WITH_DATA_GROUND_TRUTH_DATASET)
        client.create_examples(
            inputs=[{
                "user_query": ex["user_query"],
                "state": ex["state"]
            } for ex in examples],
            outputs=[{
                "expected_content_type": ex["expected_content_type"],
                "should_mention_books": ex.get("should_mention_books", []),
                "should_mention_authors": ex.get("should_mention_authors", []),
                "should_mention_preferences": ex.get("should_mention_preferences", []),
                "should_be_relevant": ex["should_be_relevant"]
            } for ex in examples],
            dataset_id=dataset.id
        )


async def run_talk_with_data(inputs: dict) -> dict:
    """
    Runs the thinking node (talk with data) for a given input.

    Args:
        inputs: A dictionary containing the user query and state data.

    Returns:
        A dictionary with the generated response and metadata.
    """
    user_query = inputs["user_query"]
    state_data = inputs["state"]

    # Create the state with the user query as a message
    state = InternalState(
        messages=[HumanMessage(content=user_query)],
        recommended_books=state_data.get("recommended_books", []),
        read_books=state_data.get("read_books", []),
        preferences=state_data.get("preferences", [])
    )

    # Run the thinking node
    result = thinking_node(state)

    return {
        "response": result["messages"].content,
        "user_query": user_query,
        "state_data": state_data
    }


def comprehensive_evaluation(run, example) -> dict:
    """
    Comprehensive evaluation using multiple LLM judges.

    Args:
        run: LangSmith run object containing outputs.
        example: LangSmith example object containing inputs and reference outputs.

    Returns:
        Dictionary with evaluation results for each metric.
    """
    outputs = run.outputs
    reference_outputs = example.outputs

    user_query = outputs.get("user_query", "")
    response = outputs.get("response", "")
    state_data = outputs.get("state_data", {})
    expected_content_type = reference_outputs.get("expected_content_type", "")

    # Run all evaluations
    results = {
        "data_accuracy": data_accuracy_score_with_llm(user_query, state_data, response),
        "response_coherence": response_coherence_score(user_query, response),
        "completeness": completeness_score(user_query, state_data, response, expected_content_type),
        "query_understanding": query_understanding_score(user_query, response)
    }

    # Calculate overall score
    results["overall_score"] = sum(results.values()) / len(results)

    return results


def data_accuracy_score_with_llm(user_query: str, state_data: dict, response: str) -> bool:
    """
    Uses LLM judge to evaluate data accuracy.

    Args:
        user_query: The user's original query.
        state_data: The available data (books, preferences, etc.).
        response: The assistant's response.

    Returns:
        True if the response is accurate, False otherwise.
    """
    judge_llm = _create_judge_llm()

    read_books = _format_data_for_evaluation(state_data.get("read_books", []))
    recommended_books = _format_data_for_evaluation(state_data.get("recommended_books", []))
    preferences = _format_data_for_evaluation(state_data.get("preferences", []))

    prompt = DATA_ACCURACY_EVALUATION_PROMPT.format(
        user_query=user_query,
        read_books=read_books,
        recommended_books=recommended_books,
        preferences=preferences,
        response=response
    )

    result = judge_llm.invoke([{"role": "user", "content": prompt}])
    return result.content.strip().upper() == "YES"


def response_coherence_score(user_query: str, response: str) -> bool:
    """
    Evaluates the coherence and helpfulness of the response.

    Args:
        user_query: The user's original query.
        response: The assistant's response.

    Returns:
        True if the response is coherent and helpful, False otherwise.
    """
    judge_llm = _create_judge_llm()

    prompt = RESPONSE_COHERENCE_EVALUATION_PROMPT.format(
        user_query=user_query,
        response=response
    )

    result = judge_llm.invoke([{"role": "user", "content": prompt}])
    return result.content.strip().upper() == "YES"


def completeness_score(user_query: str, state_data: dict, response: str, expected_content_type: str) -> bool:
    """
    Evaluates the completeness of the response.

    Args:
        user_query: The user's original query.
        state_data: The available data.
        response: The assistant's response.
        expected_content_type: The type of content expected.

    Returns:
        True if the response is complete, False otherwise.
    """
    judge_llm = _create_judge_llm()

    read_books = _format_data_for_evaluation(state_data.get("read_books", []))
    recommended_books = _format_data_for_evaluation(state_data.get("recommended_books", []))
    preferences = _format_data_for_evaluation(state_data.get("preferences", []))

    prompt = COMPLETENESS_EVALUATION_PROMPT.format(
        user_query=user_query,
        read_books=read_books,
        recommended_books=recommended_books,
        preferences=preferences,
        response=response,
        expected_content_type=expected_content_type
    )

    result = judge_llm.invoke([{"role": "user", "content": prompt}])
    return result.content.strip().upper() == "YES"


def query_understanding_score(user_query: str, response: str) -> bool:
    """
    Evaluates whether the assistant correctly understood the user's query.

    Args:
        user_query: The user's original query.
        response: The assistant's response.

    Returns:
        True if the query was understood correctly, False otherwise.
    """
    judge_llm = _create_judge_llm()

    prompt = QUERY_UNDERSTANDING_EVALUATION_PROMPT.format(
        user_query=user_query,
        response=response
    )

    result = judge_llm.invoke([{"role": "user", "content": prompt}])
    return result.content.strip().upper() == "YES"


def response_length_appropriateness(run, example) -> dict:
    """
    Evaluates if the response length is appropriate (not too short or excessively long).

    Args:
        run: LangSmith run object containing outputs.
        example: LangSmith example object containing inputs and reference outputs.

    Returns:
        Dictionary with the evaluation result.
    """
    response = run.outputs.get("response", "")
    word_count = len(response.split())

    # Response should be between 10 and 500 words for most queries
    is_appropriate = 10 <= word_count <= 500

    return {
        "key": "response_length_appropriate",
        "score": is_appropriate,
        "comment": f"Response length: {word_count} words"
    }


def data_accuracy_evaluation(run, example) -> dict:
    """
    Evaluates data accuracy using LLM judge.

    Args:
        run: LangSmith run object containing outputs.
        example: LangSmith example object containing inputs and reference outputs.

    Returns:
        Dictionary with the evaluation result.
    """
    outputs = run.outputs
    user_query = outputs.get("user_query", "")
    response = outputs.get("response", "")
    state_data = outputs.get("state_data", {})

    is_accurate = data_accuracy_score_with_llm(user_query, state_data, response)

    return {
        "key": "data_accuracy",
        "score": is_accurate,
        "comment": "Data accuracy evaluated by LLM judge"
    }


def response_coherence_evaluation(run, example) -> dict:
    """
    Evaluates response coherence using LLM judge.

    Args:
        run: LangSmith run object containing outputs.
        example: LangSmith example object containing inputs and reference outputs.

    Returns:
        Dictionary with the evaluation result.
    """
    outputs = run.outputs
    user_query = outputs.get("user_query", "")
    response = outputs.get("response", "")

    is_coherent = response_coherence_score(user_query, response)

    return {
        "key": "response_coherence",
        "score": is_coherent,
        "comment": "Response coherence evaluated by LLM judge"
    }


async def main():
    """
    Main function to run the evaluation.
    """
    load_dataset()

    # Run evaluation using LangSmith
    await client.aevaluate(
        run_talk_with_data,
        data=TALK_WITH_DATA_GROUND_TRUTH_DATASET,
        evaluators=[
            data_accuracy_evaluation,
            response_coherence_evaluation,
            response_length_appropriateness
        ],
        experiment_prefix="talk_with_data_evaluation"
    )


if __name__ == "__main__":
    asyncio.run(main())
