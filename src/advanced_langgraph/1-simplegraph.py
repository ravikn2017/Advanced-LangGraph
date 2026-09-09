from typing_extensions import TypedDict
import random
from typing import Literal
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    graph_info:str

## Nodes
#Nodes are just Python functions
# The first positional argument is the state , as defined above.

def start_play(state: State):
    print("Start_Play has been called ")
    return {"graph_info":state['graph_info'] + "I am planning to play"}

def cricket(state:State):
    print("My Cricket node has been called")
    return {"graph_info":state['graph_info'] + " Cricket"}

def badminton(state:State):
    print("My Badminton node has been called")
    return {"graph_info":state['graph_info'] + " Badminton"}

def random_play(state:State)-> Literal['cricket', 'badminton']:
    graph_info=state['graph_info']

    if random.random()>0.5:
        return "cricket"
    else:
        return "badminton"

# Graph Construction
# We build the graph from our components defined above
# The StateGraph class is the graph class that we can use
# First, we initialize a StateGraph with the State class we defined above
# Then we add our nodes and edges
# We use the START Node, a special node that sends user input to the graph, 
# to indicate where to start our graph
# The END node is a special node that represents a terminal node
# Finally, we compile our graph to perform a few basic checks on the graph structure
# We can visualize the graph as a Mermaid Diagram

## Build Graph
graph=StateGraph(State)

## Adding the Nodes
graph.add_node("start_play", start_play)
graph.add_node("cricket", cricket)
graph.add_node("badminton", badminton)

## Schedule the flow of the graph
graph.add_edge(START, "start_play")
graph.add_conditional_edges("start_play",random_play)
graph.add_edge("cricket",END)
graph.add_edge("badminton", END)

## Compile the graph
graph_builder = graph.compile()

## View the graph
#graph_builder.get_graph().draw_mermaid_png(output_file_path="Simple-graph.png")

## Graph Invocation
graph_builder.invoke({"graph_info":"My name is Ravi"})