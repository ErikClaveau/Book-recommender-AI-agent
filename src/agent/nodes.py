"""
Module defining graph nodes for the book recommendation agent.

Each node interacts with OpenAI chat models, processes user inputs,
and passes structured data through the InternalState.
"""
from typing import Dict, List, Union

from langchain_core.messages import AIMessage, SystemMessage, BaseMessage
from langgraph.graph import END
from langchain_openai import ChatOpenAI

from src.agent.data_types import RecommendedBooks, Book, Preferences
from src.agent.prompts import initial_router, talk_with_data, recommend_feedback, preferences_feedback
from src.agent.states import InternalState


def thinking_node(state: InternalState) -> Dict[str, AIMessage]:
    """
    Generate a recommendation message based on the current conversation state.

    Args:
        state (InternalState): Contains:
            - messages (List[BaseMessage]): Conversation history.
            - recommended_books (List[Book]): Previously recommended books.
            - future_books (List[Book]): Queued recommendations for next turn.
            - preferences (List[str]): User reading preferences.

    Returns:
        Dict[str, AIMessage]: A mapping with key "messages" for the next node.
    """
    print("Executing thinking node")

    # Initialize chat model for generating recommendations
    chain: ChatOpenAI = ChatOpenAI(model="gpt-3.5-turbo")

    # Build the system prompt including conversation context and user data
    prompt_text: str = talk_with_data.format(
        previous_books=str(state.get("recommended_books", [])),
        future_books=str(state.get("future_books", [])),
        preferences=str(state.get("preferences", [])),
        user_query=state["messages"][-1].content
    )
    system_msg: SystemMessage = SystemMessage(content=prompt_text)

    # Invoke the model and wrap the response
    result = chain.invoke([system_msg])
    return {"messages": AIMessage(content=result.content)}


def _recommended_feedback(
    state: InternalState,
    recommended_books: List[Book]
) -> BaseMessage:
    """
    Generate a follow-up feedback message using structured recommendations.

    Args:
        state (InternalState): Current graph state.
        recommended_books (List[Book]): Parsed list from model output.

    Returns:
        BaseMessage: Feedback message from the LLM.
    """
    # Initialize chat model for feedback
    talk_model: ChatOpenAI = ChatOpenAI(model="gpt-3.5-turbo")

    # Build feedback prompt
    prompt_text: str = recommend_feedback.format(
        previous_books=str(recommended_books),
        future_books=str(state.get("future_books", [])),
        preferences=str(state.get("preferences", [])),
        user_query=state["messages"][-1].content
    )
    return talk_model.invoke([SystemMessage(content=prompt_text)])


def save_recommended_books(
    state: InternalState
) -> Dict[str, Union[List[Book], AIMessage]]:
    """
    Extract structured book recommendations and generate feedback.

    Args:
        state (InternalState): Contains last AIMessage under "messages".

    Returns:
        Dict[str, Union[List[Book], AIMessage]]: If recommendations exist:
            - "recommended_books": Parsed list of Book objects.
            - "messages": AIMessage containing feedback.
    """
    print("Saving recommended books")

    # Deterministic model for structured output
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    structured_chain = model.with_structured_output(RecommendedBooks)

    # Parse recommendations
    parsed = structured_chain.invoke([state["messages"][-1]])
    output: Dict[str, Union[List[Book], AIMessage]] = {}

    if parsed.recommended_books:
        feedback: BaseMessage = _recommended_feedback(
            state=state,
            recommended_books=parsed.recommended_books
        )
        output["recommended_books"] = parsed.recommended_books
        output["messages"] = AIMessage(content=feedback.content)

    return output


def _preference_feedback(
    state: InternalState,
    preferences: Preferences
) -> BaseMessage:
    """
    Generate a follow-up message after capturing user preferences.

    Args:
        state (InternalState): Current graph state.
        preferences (Preferences): Parsed preferences object.

    Returns:
        BaseMessage: Feedback message from the LLM.
    """
    # Initialize chat model for preference feedback
    pref_model: ChatOpenAI = ChatOpenAI(model="gpt-3.5-turbo")

    prompt_text: str = preferences_feedback.format(
        previous_books=str(state.get("recommended_books", [])),
        future_books=str(state.get("future_books", [])),
        preferences=str(preferences),
        user_query=state["messages"][-1].content
    )
    return pref_model.invoke([SystemMessage(content=prompt_text)])


def save_preferences(
    state: InternalState
) -> Dict[str, Union[List[str], AIMessage]]:
    """
    Extract structured user preferences and generate feedback.

    Args:
        state (InternalState): Contains last AIMessage under "messages".

    Returns:
        Dict[str, Union[List[str], AIMessage]]: If preferences exist:
            - "preferences": Parsed list of preference strings.
            - "messages": AIMessage containing feedback.
    """
    print("Saving user preferences")

    # Deterministic model for structured output
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    structured_chain = model.with_structured_output(Preferences)

    # Parse preferences
    parsed = structured_chain.invoke([state["messages"][-1]])
    output: Dict[str, Union[List[str], AIMessage]] = {}

    if parsed.preferences:
        feedback: BaseMessage = _preference_feedback(
            state=state,
            preferences=parsed
        )
        output["preferences"] = parsed.preferences
        output["messages"] = AIMessage(content=feedback.content)

    return output


def get_intention(state: InternalState) -> Union[str, END]:
    """
    Determine routing intention based on the user's last message.

    Args:
        state (InternalState): Contains last AIMessage under "messages".

    Returns:
        Union[str, END]: Next state tag or END to terminate.
    """
    print("Getting intention")

    # Model returns a tag corresponding to INITIAL_ROUTER_TAGS or END
    router = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=2)
    prompt_text: str = initial_router.format(
        user_intention=state["messages"][-1].content
    )
    result = router.invoke([SystemMessage(content=prompt_text)])
    return result.content
