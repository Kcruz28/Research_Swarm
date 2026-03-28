from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END



class Agents:
    def __init__(self):
        self.model = OllamaLLM(
            model="qwen2.5-coder:1.5b",
            temperature=0.7,
            base_url="http://localhost:11434"
        )

    #nodes
    def analysist(self, state):
        # Extract the user message from state
        # state["messages"][-1] is a Message object, use .content attribute
        user_input = state["messages"][-1].content if state.get("messages") else "No input"
        prompt = "You are an analyst. Analyze the following research paper and make a " \
        "detailed summary of the paper. The paper is: " + user_input
        response = self.model.invoke(prompt)
        return {"messages": [{"role": "ai", "content": f"Analyst: {response}"}]}

    def critic(self, state):
        # Extract the last message using .content attribute (not dictionary subscripting)
        user_input = state["messages"][-1].content if state.get("messages") else "No input"
        prompt = "You are a critic. Critique the following research paper summary. Summary: " + user_input
        response = self.model.invoke(prompt)
        return {"messages": [{"role": "ai", "content": f"Critic: {response}"}]}

    def refiner(self, state):
        # Extract the last message using .content attribute (not dictionary subscripting)
        user_input = state["messages"][-1].content if state.get("messages") else "No input"
        prompt = "You are a refiner. Refine the following research paper summary based on critique. Summary: " + user_input
        response = self.model.invoke(prompt)
        return {"messages": [{"role": "ai", "content": f"Refiner: {response}"}]}



