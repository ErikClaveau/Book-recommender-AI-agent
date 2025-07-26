"""
Module defining graph nodes for the book recommendation agent.

Each node is a step in the workflow, interacting with OpenAI chat models
and passing structured data through the InternalState.
"""
from typing import Dict, List, Union

from langchain_core.messages import AIMessage, SystemMessage, BaseMessage
from langgraph.graph import END
from langchain_openai import ChatOpenAI

from src.agent.data_types import RecommendedBooks, Book
from src.agent.prompts import initial_router, talk_with_data, recommend_feedback
from src.agent.states import InternalState


def thinking_node(state: InternalState) -> Dict[str, AIMessage]:
    """
    Generate a new recommendation message based on current conversation state.

    Args:
        state (InternalState): Contains:
            - messages (List[BaseMessage]): Conversation history.
            - recommended_books (List[Book]): Previously recommended books.
            - future_books (List[Book]): Books queued for future recommendation.
            - preferences (List[str]): User reading preferences.

    Returns:
        Dict[str, AIMessage]: New state mapping:
            - "messages": AIMessage with the model's recommendation response.
    """
    print("Executing thinking node")

    chain: ChatOpenAI = ChatOpenAI(model="gpt-3.5-turbo")
    prompt_text: str = talk_with_data.format(
        previous_books=str(state.get("recommended_books", [])),
        future_books=str(state.get("future_books", [])),
        preferences=str(state.get("preferences", [])),
        user_query=state["messages"][-1].content
    )
    system_msg: SystemMessage = SystemMessage(content=prompt_text)

    result = chain.invoke([system_msg])
    return {"messages": AIMessage(content=result.content)}


def _recommended_feedback(
    state: InternalState,
    recommended_books: List[Book]
) -> BaseMessage:
    """
    Create a feedback message based on structured recommendations.

    Args:
        state (InternalState): Current state with conversation history.
        recommended_books (List[Book]): Books extracted from save_recommended_books.

    Returns:
        BaseMessage: The LLM's feedback response.
    """
    talk_model: ChatOpenAI = ChatOpenAI(model="gpt-3.5-turbo")
    prompt_text: str = recommend_feedback.format(
        previous_books=str(recommended_books),
        future_books=str(state.get("future_books", [])),
        preferences=str(state.get("preferences", [])),
        user_query=state["messages"][-1].content
    )
    result = talk_model.invoke([SystemMessage(content=prompt_text)])
    return result


def save_recommended_books(
    state: InternalState
) -> Dict[str, Union[List[Book], AIMessage]]:
    """
    Extract and persist structured book recommendations, then generate follow-up feedback.

    Args:
        state (InternalState): Contains last AIMessage under "messages".

    Returns:
        Dict[str, Union[List[Book], AIMessage]]: New state entries:
            - "recommended_books": List[Book] if any recommendations.
            - "messages": AIMessage with feedback if recommendations.
    """
    print("Saving Recommended Books")

    recommender_model: ChatOpenAI = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )
    structured_chain = recommender_model.with_structured_output(RecommendedBooks)

    recommended_books_struct = structured_chain.invoke([state["messages"][-1]])
    output: Dict[str, Union[List[Book], AIMessage]] = {}

    if recommended_books_struct.recommended_books:
        feedback: BaseMessage = _recommended_feedback(
            state=state,
            recommended_books=recommended_books_struct.recommended_books
        )
        output["recommended_books"] = recommended_books_struct.recommended_books
        output["messages"] = AIMessage(content=feedback.content)

    return output


def get_intention(state: InternalState) -> Union[str, END]:
    """
    Determine the routing intention based on the user's last message.

    Args:
        state (InternalState): Contains last message under "messages".

    Returns:
        Union[str, END]: The next state token or END to terminate.
    """
    print("Getting intention")

    router_model: ChatOpenAI = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=2
    )
    prompt_text: str = initial_router.format(
        user_intention=state["messages"][-1].content
    )

    result = router_model.invoke([SystemMessage(content=prompt_text)])
    return result.content
