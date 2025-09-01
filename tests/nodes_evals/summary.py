import asyncio
import json
import os
from typing import Dict, List
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langsmith import Client

from tests.utils.constants import SUMMARY_GROUND_TRUTH_DATASET
from tests.utils.paths import SUMMARY_GROUND_TRUTH
from tests.utils.evaluation_prompts import (
    SUMMARY_COVERAGE_PROMPT,
    SUMMARY_PERSONALIZATION_PROMPT,
    SUMMARY_NO_HALLUCINATION_PROMPT,
    SUMMARY_TONE_CLOSURE_PROMPT,
)

from app.graph.states import InternalState
from app.graph.nodes import do_summary

load_dotenv()

client = Client()


def _get_judge_models() -> List[str]:
    env_models = os.getenv("SUMMARY_JUDGE_MODELS", "").strip()
    if env_models:
        return [m.strip() for m in env_models.split(",") if m.strip()]
    return ["gpt-4o-mini", "gpt-4o-mini"]


def _create_judge_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=0)


def _format_books(data: List[Dict]) -> str:
    if not data:
        return "None"
    return "\n".join([f"- {item.get('name', 'Unknown')} by {item.get('author', 'Unknown')}" for item in data])


def load_dataset() -> None:
    with open(SUMMARY_GROUND_TRUTH, "r", encoding="utf-8") as f:
        examples = json.load(f)

    if not client.has_dataset(dataset_name=SUMMARY_GROUND_TRUTH_DATASET):
        dataset = client.create_dataset(dataset_name=SUMMARY_GROUND_TRUTH_DATASET)
        client.create_examples(
            inputs=[{"state": ex["state"]} for ex in examples],
            outputs=[{
                "expected_accomplishments": ex["expected_accomplishments"],
                "should_reference_preferences": ex["should_reference_preferences"]
            } for ex in examples],
            dataset_id=dataset.id
        )


async def run_summary(inputs: dict) -> dict:
    state_data = inputs["state"]

    # Build InternalState messages list
    messages = []
    for m in state_data.get("messages", []):
        if m.get("role") == "user":
            messages.append(HumanMessage(content=m.get("content", "")))
        else:
            messages.append(AIMessage(content=m.get("content", "")))

    state = InternalState(
        messages=messages,
        recommended_books=state_data.get("recommended_books", []),
        read_books=state_data.get("read_books", []),
        preferences=state_data.get("preferences", []),
        intents=state_data.get("intents", [])
    )

    result = do_summary(state)

    return {
        "summary": result["messages"].content,
        "state_data": state_data
    }


def _judge_yes_no(prompt: str) -> bool:
    models = _get_judge_models()
    votes = []
    for m in models:
        llm = _create_judge_llm(m)
        try:
            response = llm.invoke([{"role": "user", "content": prompt}])
            vote = response.content.strip().upper()
            votes.append(vote)
        except Exception:
            votes.append("ERROR")
    valid_votes = [v for v in votes if v in ("YES", "NO")]
    if not valid_votes:
        return False
    yes_count = sum(1 for v in valid_votes if v == "YES")
    return yes_count >= (len(valid_votes) / 2)


def coverage_evaluation(outputs, reference_outputs) -> dict:
    expected_accomplishments = reference_outputs.get("expected_accomplishments", [])
    summary = outputs.get("summary", "")

    prompt = SUMMARY_COVERAGE_PROMPT.format(
        expected_accomplishments="\n".join([f"- {a}" for a in expected_accomplishments]) or "None",
        summary=summary
    )

    if not expected_accomplishments:
        score = True
    else:
        score = _judge_yes_no(prompt)

    return {"key": "summary_coverage", "score": score}


def personalization_evaluation(outputs, reference_outputs) -> dict:
    state_data = outputs.get("state_data", {})

    preferences = state_data.get("preferences", [])
    summary = outputs.get("summary", "")
    should_reference = reference_outputs.get("should_reference_preferences", False)

    prompt = SUMMARY_PERSONALIZATION_PROMPT.format(
        preferences="\n".join([f"- {p}" for p in preferences]) or "None",
        summary=summary
    )

    if not preferences and not should_reference:
        score = True
    else:
        score = _judge_yes_no(prompt)

    return {"key": "summary_personalization", "score": score}


def no_hallucination_evaluation(outputs, reference_outputs) -> dict:
    state_data = outputs.get("state_data", {})
    summary = outputs.get("summary", "")

    prompt = SUMMARY_NO_HALLUCINATION_PROMPT.format(
        read_books=_format_books(state_data.get("read_books", [])),
        recommended_books=_format_books(state_data.get("recommended_books", [])),
        preferences="\n".join([f"- {p}" for p in state_data.get("preferences", [])]) or "None",
        summary=summary
    )

    score = _judge_yes_no(prompt)

    return {"key": "summary_no_hallucination", "score": score}


def tone_closure_evaluation(outputs, reference_outputs) -> dict:
    summary = outputs.get("summary", "")

    prompt = SUMMARY_TONE_CLOSURE_PROMPT.format(summary=summary)
    score = _judge_yes_no(prompt)

    return {"key": "summary_tone_closure", "score": score}


def overall_evaluation(outputs, reference_outputs) -> dict:
    # This aggregator will be computed post-hoc by averaging existing metrics if needed.
    return {"key": "summary_overall_placeholder", "score": 1}


async def main():
    load_dataset()

    await client.aevaluate(
        run_summary,
        data=SUMMARY_GROUND_TRUTH_DATASET,
        evaluators=[
            coverage_evaluation,
            personalization_evaluation,
            no_hallucination_evaluation,
            tone_closure_evaluation,
        ],
        experiment_prefix="summary_evaluation"
    )


if __name__ == "__main__":
    asyncio.run(main())
