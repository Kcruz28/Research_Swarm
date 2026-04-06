from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from rich.console import Console
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
import time


class SummaryOutput(BaseModel):
    summary: str = Field(description="The comprehensive summary of the paper, adhering strictly to all requested formatting including bullet points and paragraphs")
    confidence: float = Field(description="0.0 to 1.0")

class CriticOutput(BaseModel):
    is_approved: bool = Field(description="True if perfect, False if missing details")
    critique: str = Field(description="Specific flaws or 'PASS'")
    confidence: float = Field(description="0.0 to 1.0")

class Agents:
    def __init__(self, data):
        self.raw_model = ChatOllama(
            model="qwen3.5:4b",  # Or whatever specific 9B model you have pulled, like "llama3.1:8b" or "qwen2.5"
            # temperature=0.7,
            base_url="http://localhost:11434",
            # num_ctx=32000,  # Expand the model's memory to allow reading large documents
        )
        self.summary_writer = self.raw_model.with_structured_output(SummaryOutput)
        self.critic_reviewer = self.raw_model.with_structured_output(CriticOutput)
        self.data = data

    #nodes
    def analysist(self, state):
        console = Console()
        data_text = "\n".join([doc.page_content for doc in self.data]) if isinstance(self.data, list) else self.data
        
        # Grab the last message. If it's a REJECTION, show it to the Analyst.
        last_message = state["messages"][-1].content if state["messages"] else ""
        feedback = f"\n\nCRITIC FEEDBACK: {last_message}" if "REJECTED" in last_message else ""
        
        max_chars = 100000 
        truncated_text = data_text[:max_chars] + ("...[truncated]" if len(data_text) > max_chars else "")
        print(f"--- RAW TEXT PREVIEW ---\n{truncated_text[:500]}\n-----------------------")
        
        with open("analyst_prompt.txt", "r") as f:
            base_prompt = f.read()
        prompt = base_prompt.replace("{text}", truncated_text).replace("{feedback}", feedback)
        
        with console.status("[bold blue]Analyst is deconstructing the paper...", spinner="dots"):
            start = time.time()
            response = self.summary_writer.invoke([HumanMessage(content=prompt)])
            elapsed = time.time() - start
            console.print(f"[blue]✓ Analyst done in {elapsed:.1f}s")
        
        return {"messages": [AIMessage(content=f"Analyst: {response.summary}")]}

    def critic(self, state):
        console = Console()
        # Extract the last message using .content attribute (not dictionary subscripting)

        # stoping infinite loop
        if len(state["messages"]) > 4: 
            console.print("[yellow]! Overriding Critic: Maximum attempts reached, forcing approval.")
            return {"messages": [AIMessage(content="APPROVED: Maximum revision limit reached.")]}

        analysist_message = state["messages"][-1].content
        paper_text = "\n".join([doc.page_content for doc in self.data]) if isinstance(self.data, list) else self.data
        max_chars = 100000 
        truncated_text = paper_text[:max_chars] + ("...[truncated]" if len(paper_text) > max_chars else "")
        
        with open("critic_prompt.txt", "r") as f:
            base_prompt = f.read()
        prompt = base_prompt.replace("{summary}", analysist_message).replace("{text}", truncated_text)
        
        with console.status("[bold red]Critic is looking for flaws...", spinner="dots"):
            start = time.time()
            response = self.critic_reviewer.invoke([HumanMessage(content=prompt)])
            elapsed = time.time() - start
            console.print(f"[red]✓ Critic done in {elapsed:.1f}s")
        
        status = "REJECTED" if not response.is_approved else "APPROVED"
        return {"messages": [AIMessage(content=f"{status}: {response.critique}")]}



    def refiner(self, state):
        console = Console()
        full_history = "\n".join([m.content for m in state["messages"]])
        
        with open("refiner_prompt.txt", "r") as f:
            base_prompt = f.read()
        prompt = base_prompt.replace("{history}", full_history)
        
        with console.status("[bold green]Refiner is polishing the final summary...", spinner="dots"):
            start = time.time()
            # We use the raw model without structured output for perfect Markdown formatting
            response = self.raw_model.invoke([HumanMessage(content=prompt)])
            elapsed = time.time() - start
            console.print(f"[green]✓ Refiner done in {elapsed:.1f}s")
        
        response_content = response.content if hasattr(response, 'content') else str(response)
        return {"messages": [AIMessage(content=f"Refiner: {response_content}")]}



