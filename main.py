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
    # defensive access: messages may be empty or contain dicts/objects
    messages = state.get("messages") or []
    if not messages:
        # nothing to classify; default to logical
        return {"message_type": "logical"}

    last_message = messages[-1]

    # support both dict-like and object-like message representations
    content = getattr(last_message, "content", None) or last_message.get("content") if isinstance(last_message, dict) else None
    if content is None:
        # fallback if message shape is unexpected
        return {"message_type": "logical"}

    classifier_llm = llm.with_structured_output(MessageClassifier)

    try:
        result = classifier_llm.invoke([
            {
                "role": "system",
                "content": """Classify the user message as either:
            - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
            - 'logical': if it asks for facts, information, logical analysis, or practical solutions
            """
            },
            {"role": "user", "content": content}
        ])
        return {"message_type": result.message_type}
    except Exception:
        # if the classifier fails, return a sensible default
        return {"message_type": "logical"}



def router(state: State):
     message_type = state.get("message_type", "logical")

     if message_type == "emotional":
         return {"next": "therapist"}
     
     return {"next": "logical"} 

def therapist_agent(state: State):
    messages_list = state.get("messages") or []
    if not messages_list:
        return {"messages": []}

    last_message = messages_list[-1]

    # support dict or object
    content = getattr(last_message, "content", None) or (last_message.get("content") if isinstance(last_message, dict) else None)
    if content is None:
        return {"messages": []}

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
            "content": content
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
    messages_list = state.get("messages") or []
    if not messages_list:
        return {"messages": []}

    last_message = messages_list[-1]

    content = getattr(last_message, "content", None) or (last_message.get("content") if isinstance(last_message, dict) else None)
    if content is None:
        return {"messages": []}

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
            "content": content
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
    lambda state: state.get("next"),
    {"therapist": "therapist", "logical": "logical"}
)

graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)


graph = graph_builder.compile()

def run_chatbot():
    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("Bye")
            break

        state["messages"] = state.get("messages", []) + [
            {
                "role": "user", "content": user_input
            }
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            # support object or dict message
            assistant_content = getattr(last_message, "content", None) or (last_message.get("content") if isinstance(last_message, dict) else None)
            if assistant_content:
                print(f"Assistant: {assistant_content}")


if __name__ == "__main__":
    run_chatbot()

# from IPython.display import Image, display
# png = graph.get_graph().draw_mermaid_png()
# out_path = "graph.png"
# with open(out_path, "wb") as f:
#     f.write(png)
# print(f"Saved graph to {out_path}")
# display(Image(data=png))