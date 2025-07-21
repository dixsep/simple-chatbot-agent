
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from typing_extensions import TypedDict

load_dotenv()   # loads env variables


llm = init_chat_model(
    "anthropic:claude-3-5-sonnet-latest"
)  # can be other llm models also

class State(TypedDict):
    messages : Annotated[list, add_messages]


def chatbot(state : State) -> State:

    last_message = state["messages"][-1]

    messages = [{
        "role" : "user",
        "content" : last_message.content
    }]

    reply = llm.invoke(messages)

    return {"messages" : [{"role" : "agent", "content" : reply.content}]}


graph_builder = StateGraph(State)

# add nodes
graph_builder.add_node("chatbot", chatbot)

# add edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

def run_bot():

    state = {"messages" : []}

    while True:

        user_input = input("Enter ur query : ")

        if user_input == "exit":
            break

        state["messages"] = state.get("messages") + [
            {
                "role" : "user",
                "content" : user_input
            }
        ]

        state = graph.invoke(state);
        print(state["messages"][-1].content)


'''
LLM's knowledge is limited upto recent training
it will not give daily news, stock prices etc

LLM must be able to call API's (tools) or some python function
to ge the latest data
'''
