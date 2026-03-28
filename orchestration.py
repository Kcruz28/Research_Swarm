from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from agents import Agents



def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}

class Orchestration:
    def __init__(self):
        self.graph = StateGraph(MessagesState)
        self.agents = {}

    def add_node(self, node_name, node_func):
        self.graph.add_node(node_name, node_func)
        self.agents[node_name] = node_func

    def add_edge(self, agent1, agent2, agent3):
        self.graph.add_edge(START, agent1)
        self.graph.add_edge(agent1, agent2)
        self.graph.add_edge(agent2, agent3)
        self.graph.add_edge(agent3, END)

    def run(self):
        # Compile the graph - finalizes the structure
        graph = self.graph.compile()
        
        # Create initial state with a user message
        # This is the starting point for the graph execution
        initial_state = {"messages": [{"role": "user", "content": "Analyze this research paper"}]}
        
        # invoke() executes the entire graph:
        # START -> analysist -> critic -> refiner -> END
        # Each node receives the current state and adds/modifies messages
        result = graph.invoke(initial_state)
        
        # Return the final state with all messages from all nodes
        return result