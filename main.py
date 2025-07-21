
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from typing_extensions import TypedDict

load_dotenv()   # loads env variables

'''
MULTI AGENT LLM
'''


llm = init_chat_model(
    "anthropic:claude-3-5-sonnet-latest"
)

class MessageClassifier(BaseModel):

    # response from llm should be in either of the literals

    message_type : Literal["emotional", "logical"] = Field(
        ...,
        description = "Classify if the message requires emotional agent or logical agent"
    )

class State(TypedDict):

    messages : Annotated[list, add_messages]
    message_type : str | None     #type of message
    next : str | None    #updates to most recent value


def classify_message(state : State):

    last_message = state["messages"][-1]  #user input
    classifier_llm = llm.with_structured_output(MessageClassifier)

    # giving instructions to classifier
    result = classifier_llm.invoke([

        {
            "role" : "system",
            "content" : """ Classify the user message as either :
             - 'emotional' : if it asks for emotional support, deals with feelings
             - 'logical' : if it asks for logical analysis, facts, information or practical solutions
            """
        },
        {
            "role" : "user",
            "content" : last_message.content
        }
    ])

    # matches the state and updates.
    return {"message_type" : result.message_type}



def router(state : State):

    message_type = state.get("message_type", "logical")

    # we can chose which node should the agent should go from router using next
    if message_type == "emotional":
        return {"next" : "therapist"}

    return {"next" : "logical"}

def therapist_agent(state : State):

    last_message = state["messages"][-1]

    messages = [
        {
            "role" : "system",
            "content" : """ You are a compasionate therapist
             Focus on emotional aspects, validate their feelings, help them to 
             explore feelings by asking questions, avoid giving logical answers
            """
        },
        {
            "role" : "user",
            "content" : last_message.content
        }
    ]

    reply = llm.invoke(messages)

    return {"messages" : [{"role" : "assistant", "content" : reply.content}]}



def logical_agent(state : State):
    last_message = state["messages"][-1]

    messages = [
        {
            "role" : "system",
            "content" : """ You are a logical assistant
                provide solutions to their questions which are logical and practical
                based on their circumstances, and also ask questions to know their circumstances before giving
                logical solutions to their queries
                """
        },
        {
            "role" : "user",
            "content" : last_message.content
        }
    ]

    reply = llm.invoke(messages)

    return {"messages" : [{"role" : "assistant", "content" : reply.content}]}



graph_builder = StateGraph(State)

#build the graph
graph_builder.add_node("classifier", classify_message);
graph_builder.add_node("router", router);
graph_builder.add_node("logical", logical_agent);
graph_builder.add_node("therapist", therapist_agent);

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

graph_builder.add_conditional_edges(
    "router",
    lambda state : state.get("next"),
    {"therapist" : "therapist", "logical" : "logical"}

)

graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)

# compile the graph
graph = graph_builder.compile()


def run_bot():

    state = {"messages" : [], "message_type" : None}

    while True:
        user_input = input("Enter ur query: ")

        if user_input == "exit":
            break

        state["messages"] = state.get("messages") + [{
            "role" : "user",
            "content" : user_input
        }]

        # state gets updated following the path in the graph
        state = graph.invoke(state)
        # state["messages"][-1] gives final output by LLM.

        print(state["messages"][-1].content)


if __name__ == "main":
    run_bot()
