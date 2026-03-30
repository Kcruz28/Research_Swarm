from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from rich.console import Console
from langchain_core.messages import AIMessage, HumanMessage
import time


class Agents:
    def __init__(self, data):
        self.model = ChatOllama(
            model="qwen3.5:9b",  # Or whatever specific 9B model you have pulled, like "llama3.1:8b" or "qwen2.5"
            temperature=0.7,
            base_url="http://localhost:11434",
            num_ctx=32000,  # Expand the model's memory to allow reading large documents
        )
        self.data = data

    #nodes
    def analysist(self, state):
        console = Console()
        data_text = "\n".join([doc.page_content for doc in self.data]) if isinstance(self.data, list) else self.data
        
        # Since you are using a 9B model with 32k context, we can massively expand the limit
        # 100,000 characters is roughly 25,000 tokens, which easily fits inside 32k context!
        max_chars = 100000 
        truncated_text = data_text[:max_chars] + ("...[truncated]" if len(data_text) > max_chars else "")
        
        print(f"--- RAW TEXT PREVIEW ---\n{truncated_text[:500]}\n-----------------------")
        prompt = "Write a one-sentence summary of this text: " + truncated_text
        
        print(f"DEBUG: Extracted {len(data_text)} chars, truncated to {len(truncated_text)} chars for analysis")
        print(f"DEBUG: Sending {len(prompt)} chars to {self.model.model}")
        
        with console.status("[bold blue]Analyst is deconstructing the paper...", spinner="dots"):
            start = time.time()
            response = self.model.invoke([HumanMessage(content=prompt)])
            elapsed = time.time() - start
            console.print(f"[blue]✓ Analyst done in {elapsed:.1f}s")
        
        # Debug: Check what response is
        print(f"DEBUG Analyst response type: {type(response)}")
        print(f"DEBUG Analyst response: {response}")
        
        # Extract content - handle both string and object responses
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        return {"messages": [AIMessage(content=f"Analyst: {response_content}")]}

    def critic(self, state):
        console = Console()
        # Extract the last message using .content attribute (not dictionary subscripting)
        analysist_message = state["messages"][-1].content
        prompt = "Name one missing detail from this summary: " \
        + analysist_message
        
        with console.status("[bold red]Critic is looking for flaws...", spinner="dots"):
            start = time.time()
            response = self.model.invoke([HumanMessage(content=prompt)])
            elapsed = time.time() - start
            console.print(f"[red]✓ Critic done in {elapsed:.1f}s")
        
        # Debug: Check what response is
        print(f"DEBUG Critic response type: {type(response)}")
        print(f"DEBUG Critic response: {response}")
        
        # Extract content - handle both string and object responses
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        return {"messages": [AIMessage(content=f"Critic: {response_content}")]}

    def refiner(self, state):
        console = Console()
        # Extract the last message using .content attribute (not dictionary subscripting)
        critic_message = state["messages"][-1].content
        prompt = "Rewrite this summary to be better: " \
        + critic_message
        
        with console.status("[bold green]Refiner is polishing the final summary...", spinner="dots"):
            start = time.time()
            response = self.model.invoke([HumanMessage(content=prompt)])
            elapsed = time.time() - start
            console.print(f"[green]✓ Refiner done in {elapsed:.1f}s")
        
        # Debug: Check what response is
        print(f"DEBUG Refiner response type: {type(response)}")
        print(f"DEBUG Refiner response: {response}")
        
        # Extract content - handle both string and object responses
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        return {"messages": [AIMessage(content=f"Refiner: {response_content}")]}



