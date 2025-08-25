"""
Module defining graph nodes for the book recommendation agent.

Each node interacts with OpenAI chat models, processes user inputs,
and passes structured data through the InternalState.
"""
from typing import Dict, List, Union

from langchain_core.messages import AIMessage, SystemMessage, BaseMessage, HumanMessage, RemoveMessage
from langchain_openai import ChatOpenAI

from app.graph.data_types import RecommendedBooks, Book, Preferences, ReadBooks, IntentClassification
from app.graph.prompts import (
    initial_router,
    talk_with_data,
    recommend_feedback,
    preferences_feedback,
    system_recommender_prompt,
    summarizing_prompt,
)
from app.graph.states import InternalState


def thinking_node(state: InternalState) -> Dict[str, AIMessage]:
    """
    Generate a recommendation message based on the current conversation state.

    Args:
        state (InternalState): Contains:
            - messages (List[BaseMessage]): Conversation history.
            - recommended_books (List[Book]): Previously recommended books.
            - read_books (List[Book]): Books user has read.
            - preferences (List[str]): User reading preferences.

    Returns:
        Dict[str, AIMessage]: Mapping with key "messages" for the next node.
    """
    print("Executing thinking node")

    # Initialize chat model for generating recommendations - upgraded to GPT-4o-mini
    chain: ChatOpenAI = ChatOpenAI(model="gpt-4o-mini")

    # Build prompt with full context
    prompt_text: str = talk_with_data.format(
        previous_books=str(state.get("recommended_books", [])),
        read_books=str(state.get("read_books", [])),
        preferences=str(state.get("preferences", [])),
        user_query=state["messages"][-1].content
    )
    system_msg: SystemMessage = SystemMessage(content=prompt_text)

    # Invoke the model and return AIMessage
    result = chain.invoke([system_msg])
    return {"messages": AIMessage(content=result.content)}


def _recommended_feedback(
    state: InternalState,
    recommended_books: List[Book],
    last_human_message: HumanMessage
) -> BaseMessage:
    """
    Generate follow-up feedback using structured recommendations.

    Args:
        state (InternalState): Current graph state.
        recommended_books (List[Book]): Parsed book list.

    Returns:
        BaseMessage: Feedback message from LLM.
    """
    talk_model: ChatOpenAI = ChatOpenAI(model="gpt-3.5-turbo")
    prompt_text: str = recommend_feedback.format(
        previous_books=str(recommended_books),
        read_books=str(state.get("read_books", [])),
        preferences=str(state.get("preferences", [])),
        user_query=last_human_message.content
    )
    return talk_model.invoke([SystemMessage(content=prompt_text)])


def save_recommended_books(
    state: InternalState
) -> Dict[str, Union[List[Book], AIMessage]]:
    """
    Extract structured recommendations and generate feedback.

    Args:
        state (InternalState): Contains last AIMessage under "messages".

    Returns:
        Dict[str, Union[List[Book], AIMessage]]: If recommendations exist:
            - "recommended_books": Parsed Book list.
            - "messages": AIMessage with feedback.
    """
    print("Saving recommended books")

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    structured_chain = model.with_structured_output(RecommendedBooks)
    last_human_message: HumanMessage = _get_last_human_message(state=state)

    # Use system prompt for structured parsing
    prompt_text: str = system_recommender_prompt.format(
        previous_books=str(state.get("recommended_books", [])),
        read_books=str(state.get("read_books", [])),
        preferences=str(state.get("preferences", [])),
        user_query=last_human_message.content
    )
    parsed = structured_chain.invoke([SystemMessage(content=prompt_text)])
    output: Dict[str, Union[List[Book], AIMessage]] = {}

    if parsed.recommended_books:
        feedback: BaseMessage = _recommended_feedback(
            state=state,
            recommended_books=parsed.recommended_books,
            last_human_message=last_human_message
        )
        output["recommended_books"] = parsed.recommended_books
        output["messages"] = AIMessage(content=feedback.content)

    return output


def _preference_feedback(
    state: InternalState,
    preferences: Preferences,
    last_human_message: HumanMessage
) -> BaseMessage:
    """
    Generate follow-up message after capturing user preferences.

    Args:
        state (InternalState): Current graph state.
        preferences (Preferences): Parsed preferences.

    Returns:
        BaseMessage: Feedback from LLM.
    """
    pref_model: ChatOpenAI = ChatOpenAI(model="gpt-3.5-turbo")
    prompt_text: str = preferences_feedback.format(
        previous_books=str(state.get("recommended_books", [])),
        read_books=str(state.get("read_books", [])),
        preferences=str(preferences),
        user_query=last_human_message.content
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
            - "preferences": List of preferences.
            - "messages": AIMessage with feedback.
    """
    print("Saving user preferences")

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    structured_chain = model.with_structured_output(Preferences)
    last_human_message: HumanMessage = _get_last_human_message(state=state)

    parsed = structured_chain.invoke([last_human_message])
    output: Dict[str, Union[List[str], AIMessage]] = {}

    if parsed.preferences:
        feedback: BaseMessage = _preference_feedback(
            state=state,
            preferences=parsed,
            last_human_message=last_human_message
        )
        output["preferences"] = parsed.preferences
        output["messages"] = AIMessage(content=feedback.content)

    return output


def _read_feedback(
    state: InternalState,
    read_books: List[Book],
    last_human_message: HumanMessage
) -> BaseMessage:
    """
    Generate feedback message after capturing read books.

    Args:
        state (InternalState): Current graph state.
        read_books (List[Book]): Parsed list of books user has read.

    Returns:
        BaseMessage: Feedback from LLM.
    """
    read_model: ChatOpenAI = ChatOpenAI(model="gpt-3.5-turbo")
    prompt_text: str = recommend_feedback.format(
        previous_books=str(state.get("recommended_books", [])),
        read_books=str(read_books),
        preferences=str(state.get("preferences", [])),
        user_query=last_human_message.content
    )
    return read_model.invoke([SystemMessage(content=prompt_text)])


def save_read_books(
    state: InternalState
) -> Dict[str, Union[List[Book], AIMessage]]:
    """
    Extract structured read books and generate feedback.

    Args:
        state (InternalState): Contains last AIMessage under "messages".

    Returns:
        Dict[str, Union[List[Book], AIMessage]]: If read_books exist:
            - "read_books": List of books read.
            - "messages": AIMessage with feedback.
    """
    print("Saving read books")

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    structured_chain = model.with_structured_output(ReadBooks)
    last_human_message: HumanMessage = _get_last_human_message(state=state)

    parsed = structured_chain.invoke([last_human_message])
    output: Dict[str, Union[List[Book], AIMessage]] = {}

    if parsed.read_books:
        feedback: BaseMessage = _read_feedback(
            state=state,
            read_books=parsed.read_books,
            last_human_message=last_human_message
        )
        output["read_books"] = parsed.read_books
        output["messages"] = AIMessage(content=feedback.content)

    return output


def empty_node(state: InternalState) -> dict:
    return {}


def _get_last_human_message(state: InternalState) -> HumanMessage:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return message

    return HumanMessage(content="")


def get_intention(state: InternalState) -> list[str]:
    """
    Determine routing intention based on the user's last message using structured output.

    Uses an improved intent classifier with Pydantic validation to ensure
    accurate intent detection and proper output formatting.

    Args:
        state (InternalState): Contains last message under "messages".

    Returns:
        list[str]: List of detected intent tags for routing decisions.
    """
    print("Getting intention")

    # Use GPT-4o for better intent classification
    router = ChatOpenAI(model="gpt-4o", temperature=0)

    # Create structured chain with Pydantic validation
    structured_router = router.with_structured_output(IntentClassification)

    prompt_text: str = initial_router.format(
        user_intention=state["messages"][-1].content
    )

    try:
        # Get structured output with validation
        classification = structured_router.invoke([SystemMessage(content=prompt_text)])

        # Convert enum values to strings for compatibility
        result = [intent.value for intent in classification.intents]

        state["intents"] = result

        print(f"Detected intents: {result}")
        return result

    except Exception as e:
        print(f"Error in intent classification: {e}")
        # Fallback to 'end' if classification fails
        return ["end"]


def do_summary(state: InternalState) -> Dict[str, AIMessage]:
    print("Summarizing info")

    # Initialize chat model for generating recommendations - upgraded to GPT-4o-mini
    chain: ChatOpenAI = ChatOpenAI(model="gpt-4")

    # Build prompt with full context
    prompt_text: str = summarizing_prompt.format(
        intents=str(state.get("intents", [])),
        message_history=str(state.get("messages", [])[1:]),
        previous_books=str(state.get("recommended_books", [])),
        read_books=str(state.get("read_books", [])),
        preferences=str(state.get("preferences", [])),
        user_query=state["messages"][0].content
    )
    system_msg: SystemMessage = SystemMessage(content=prompt_text)

    # Invoke the model and return AIMessage
    result = chain.invoke([system_msg])
    return {"messages": AIMessage(content=result.content)}


def clean_message_history(state: InternalState) -> Dict[str, List]:
    intents = get_intention(state=state)

    if len(state["messages"]) > 1:
        messages = [RemoveMessage(id=message.id) for message in state["messages"][:-2]]

        return {"messages": messages,
                "intents": intents}

    return {"intents": intents}


def get_intents(state: InternalState) -> List[str]:
    """
    Retrieve the list of intents from the current state.

    Args:
        state (InternalState): The current graph state containing intents.

    Returns:
        List[str]: List of intent tags.
    """
    return state.get("intents", [])
