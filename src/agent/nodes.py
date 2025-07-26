from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import MessagesState, END
from langchain_openai import ChatOpenAI

from src.agent.data_types import RecommendedBooks
from src.agent.prompts import system_recommender_prompt, initial_router, talk_with_data
from src.agent.states import InternalState


def thinking_node(state: InternalState) -> InternalState:
    """
    Node that generates book recommendations based on the conversation history.

    This node invokes an OpenAI chat model with a system prompt and the last user message,
    then wraps the LLM response in an AIMessage for downstream states.

    Args:
        state (MessagesState): The current messages state, containing conversation history
            under the key "messages" and any previous recommendations under "recommended_books".

    Returns:
        InternalState: A new state dict with a single key "messages" mapping to an AIMessage
            containing the model's response content.
    """
    print("Executing thinking node")

    # Initialize chat chain
    chain = ChatOpenAI(model="gpt-3.5-turbo")

    # Prepare prompt with context of previously recommended books
    prompt_text = talk_with_data.format(
        previous_books=str(state.get("recommended_books", [])),
        future_books=str(state.get("future_books", [])),
        preferences=str(state.get("preferences", [])),
        user_query=state["messages"][-1].content
    )

    # Invoke the model
    result = chain.invoke([SystemMessage(content=prompt_text)])

    # Return the AIMessage for next state
    return {"messages": AIMessage(content=result.content)}


def save_recommended_books(state: InternalState) -> dict:
    """
    Node that extracts and saves structured book recommendations from the LLM output.

    This node calls an OpenAI chat model configured for structured output,
    parses the RecommendedBooks schema, and returns any extracted recommendations.

    Args:
        state (InternalState): The state containing the latest AIMessage under "messages".

    Returns:
        dict: A dict with key "recommended_books" mapping to a list of book recommendations,
            if any are present; otherwise, an empty dict.
    """
    print("Saving Recommended Books")

    # Initialize deterministic chat model for structured output
    model = ChatOpenAI(model="gpt-3.5-turbo",
                       temperature=0)
    model_with_structure = model.with_structured_output(RecommendedBooks)

    # Invoke and parse structured recommendations
    recommended_books = model_with_structure.invoke([state["messages"][-1]])

    if recommended_books.recommended_books:
        return {"recommended_books": recommended_books.recommended_books}
    return {}


def get_intention(state: InternalState) -> str or END:
    model = ChatOpenAI(model="gpt-3.5-turbo",
                       temperature=0,
                       max_tokens=2)

    prompt_text = initial_router.format(
        user_intention=state["messages"][-1].content
    )

    result = model.invoke([SystemMessage(content=prompt_text)])

    print(result)
    return result.content


def initial_router_node(state: InternalState) -> dict:
    return {}
