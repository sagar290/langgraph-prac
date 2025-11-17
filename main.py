from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


load_dotenv()


# Using local LLM via Ollama
llm = ChatOllama(model="mistral", temperature=0)

class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requiress an emotional (therapist) or lgical response."
    )


class State (TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None
    next: str | None

def classify_message(state: State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as either:
            - 'emotional': if it ask for emotional support, therapy, deals with feelings or personal problem.
            - 'logical': if it ask for facts, intformation, logical analysis or practical solution.
               """
        },
        {
            "role": "user", 
            "content": last_message.content
        }
    ])

    return {"message_type": result.message_type}



def router(state: State):
     message_type = state.get("message_type", "logical")

     if message_type == "emotional":
         return {"next": "therapist"}
     
     return {"next": "logical"} 

def therapist_agent(state: State):
    last_message = state["messages"][-1]

    messages =  [
        {
            "role": "system",
            "content": """You are a compensationate therapist. focus on a emotional aspect of the user message.
                Show empathy validate their feelings and help them to process their emotions.
                Ask thoughtful qustion to help them explore their feelings more deeply.
                Avoid giving logical solution unless explicitly asked.
            """
        }, 
        {
            "role": "user",
            "content": last_message.content
        }
    ]

    reply = llm.invoke(messages)
    return {"messages": [
        {
            "role": "assistant",
            "content": reply.content
        }
    ]}

def logical_agent(state: State):
    last_message = state["messages"][-1]

    messages =  [
        {
            "role": "system",
            "content": """You are a purely logical assistant. Focus only on facts and information.
            Provide clear, concisr answere based on logical and evidance.
            Do not address emotions or provide emotional support.
            Be direct and straightforward in your responses.
            """
        }, 
        {
            "role": "user",
            "content": last_message.content
        }
    ]

    reply = llm.invoke(messages)
    return {"messages": [
        {
            "role": "assistant",
            "content": reply.content
        }
    ]}



graph_builder = StateGraph(State)


graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)


graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")
graph_builder.add_conditional_edges(
    "router",
    lambda state: state.gate("next"),
    {"therapist": "therapist", "logical": "logical"}
)

graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)


graph = graph_builder.compile()

def run_chatbot():
    state = {"message": [], "message_type": None}

    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("Bye")
            break

        state["message"] = state.get("messages", []) + [
            {
                "role": ":user", "content": user_input
            }
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message =  state["messages"][-1]
            print(f"Assistant: {last_message.content}")


if __name__ == "__main__":
    run_chatbot()

# from IPython.display import Image, display
# png = graph.get_graph().draw_mermaid_png()
# out_path = "graph.png"
# with open(out_path, "wb") as f:
#     f.write(png)
# print(f"Saved graph to {out_path}")
# display(Image(data=png))