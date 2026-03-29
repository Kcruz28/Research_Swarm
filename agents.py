from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from rich.console import Console
from langchain_core.messages import AIMessage
import time


class Agents:
    def __init__(self, data):
        self.model = ChatOllama(
            model="qwen3.5:0.8b",
            temperature=1.0,
            presence_penalty=1.5, # The key to stopping the hang!
            base_url="http://localhost:11434",
            additional_kwargs={"think": False}
        )
        self.data = data

    #nodes
    def analysist(self, state):
        console = Console()
        data_text = "\n".join([doc.page_content for doc in self.data]) if isinstance(self.data, list) else self.data
        prompt = "Summarize in 10 words. Be direct: " + data_text
        
        print(f"DEBUG: Extracted {len(data_text)} chars from document for analysis")
        print(f"DEBUG: Sending {len(prompt)} chars to {self.model.model}")
        
        with console.status("[bold blue]Analyst is deconstructing the paper...", spinner="dots"):
            start = time.time()
            response = self.model.invoke(prompt)
            elapsed = time.time() - start
            console.print(f"[blue]✓ Analyst done in {elapsed:.1f}s")
        
        # Debug: Check what response is
        print(f"DEBUG Analyst response type: {type(response)}")
        print(f"DEBUG Analyst response: {response}")
        
        # Extract content - handle both string and object responses
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        return {"messages": state["messages"] + [AIMessage(content=f"Analyst: {response_content}")]}

    def critic(self, state):
        console = Console()
        # Extract the last message using .content attribute (not dictionary subscripting)
        analysist_message = state["messages"][-1].content
        prompt = "Criticize in 10 words. Be direct: " \
        + analysist_message
        
        with console.status("[bold red]Critic is looking for flaws...", spinner="dots"):
            start = time.time()
            response = self.model.invoke(prompt)
            elapsed = time.time() - start
            console.print(f"[red]✓ Critic done in {elapsed:.1f}s")
        
        # Debug: Check what response is
        print(f"DEBUG Critic response type: {type(response)}")
        print(f"DEBUG Critic response: {response}")
        
        # Extract content - handle both string and object responses
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        return {"messages": state["messages"] + [AIMessage(content=f"Critic: {response_content}")]}

    def refiner(self, state):
        console = Console()
        # Extract the last message using .content attribute (not dictionary subscripting)
        critic_message = state["messages"][-1].content
        prompt = "Improve in 10 words. Be direct: " \
        + critic_message
        
        with console.status("[bold green]Refiner is polishing the final summary...", spinner="dots"):
            start = time.time()
            response = self.model.invoke(prompt)
            elapsed = time.time() - start
            console.print(f"[green]✓ Refiner done in {elapsed:.1f}s")
        
        # Debug: Check what response is
        print(f"DEBUG Refiner response type: {type(response)}")
        print(f"DEBUG Refiner response: {response}")
        
        # Extract content - handle both string and object responses
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        return {"messages": state["messages"] + [AIMessage(content=f"Refiner: {response_content}")]} 



