from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()   # loads env variables

@tool
def get_stock_price(symbol : str) -> float:
    # doc string is very important (llm calls tools intelligently by doc string only)

    ''' Return the current price of stock given the stock symbol
    :param symbol: staock symbol
    :return: current price of the stock
    '''

    return {
        "BK" : 98.4,
        "JPMC" : 289.43,
        "GS" : 650
    }.get(symbol, 0.0)


tools = [get_stock_price]


llm = init_chat_model(
    "anthropic:claude-3-5-sonnet-latest"
)

llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):

    messages : Annotated[list, add_messages]


def chatbot(state : State) -> State:
    pass


graph_builder = StateGraph(State)

#nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))
#edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

'''
 1 . what is stock price of GS (works , call to tool)
 2 . What is theory of relativity  (works )
 3 . what will be the total price if i want to buy 20APPL STock and 15GS stocks (just only gives stock price, DOESNIOW WORK)
    No Math is done by the LLM 
    Fix : recursive call to Chatbot by tool (to send data)

'''

