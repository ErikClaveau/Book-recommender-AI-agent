from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState

from src.agent.prompts import system_recommender_prompt

load_dotenv()


def thinking_node(state: MessagesState):
    print("Executing thinking node")

    chain = ChatOpenAI(model="gpt-3.5-turbo")

    result = chain.invoke([SystemMessage(content=system_recommender_prompt), state["messages"][-1]])

    return {"messages": AIMessage(content=result.content)}


builder = StateGraph(MessagesState)

builder.add_node("thinking_node", thinking_node)

builder.add_edge(START, "thinking_node")
builder.add_edge("thinking_node", END)

graph = builder.compile()
