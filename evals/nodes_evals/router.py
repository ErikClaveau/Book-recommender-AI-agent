import asyncio
import json
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langsmith import Client
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)

from evals.utils.constants import ROUTER_GROUND_TRUTH_DATASET, ROUTER_EVALUATION_RUN
from evals.utils.paths import ROUTER_GROUND_TRUTH

from src.agent.states import InternalState
from src.agent.nodes import get_intention
from src.utils.constants import INITIAL_ROUTER_TAGS

load_dotenv()

client = Client()


def load_dataset() -> None:
    with open(ROUTER_GROUND_TRUTH, "r", encoding="utf-8") as f:
        examples = json.load(f)

    if not client.has_dataset(dataset_name=ROUTER_GROUND_TRUTH_DATASET):
        dataset = client.create_dataset(dataset_name=ROUTER_GROUND_TRUTH_DATASET)
        client.create_examples(
            inputs=[{"messages": ex["messages"]} for ex in examples],
            outputs=[{"route": ex["route"]} for ex in examples],
            dataset_id=dataset.id
        )


async def run_intent(inputs: dict) -> dict:
    params = {"messages": [HumanMessage(content=inputs["messages"][0]["content"])]}
    intent = get_intention(InternalState(**params))

    return {"route": intent}


def accuracy(outputs: dict, reference_outputs: dict) -> bool:
    return outputs["route"] == reference_outputs["route"]


def hamming_accuracy(outputs: dict, reference_outputs: dict) -> float:
    correct = 0

    for lbl in INITIAL_ROUTER_TAGS.keys():
        if (lbl in reference_outputs["route"]) == (lbl in outputs["route"]):
            correct += 1
    return correct / len(INITIAL_ROUTER_TAGS.keys())


def jaccard_index(outputs: dict, reference_outputs: dict) -> float:
    y_true = set(reference_outputs["route"])
    y_hat = set(outputs["route"])

    if not y_true and not y_hat:
        return 1.0
    return len(y_true & y_hat) / len(y_true | y_hat)


async def run_evals():
    load_dataset()

    experiment_results = await client.aevaluate(
        run_intent,
        data=ROUTER_GROUND_TRUTH_DATASET,
        experiment_prefix="initial_version",
        evaluators=[accuracy, hamming_accuracy, jaccard_index],
        max_concurrency=4,
    )
    experiment_results.to_pandas()


if __name__ == "__main__":
    asyncio.run(run_evals())
