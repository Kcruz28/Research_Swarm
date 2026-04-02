from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from rich.console import Console
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
import time


class SummaryOutput(BaseModel):
    summary: str = Field(description="The 1-sentence summary of the paper")
    confidence: float = Field(description="0.0 to 1.0")

class CriticOutput(BaseModel):
    is_approved: bool = Field(description="True if perfect, False if missing details")
    critique: str = Field(description="Specific flaws or 'PASS'")
    confidence: float = Field(description="0.0 to 1.0")

class Agents:
    def __init__(self, data):
        raw_model = ChatOllama(
            model="qwen3.5:4b",  # Or whatever specific 9B model you have pulled, like "llama3.1:8b" or "qwen2.5"
            # temperature=0.7,
            base_url="http://localhost:11434",
            # num_ctx=32000,  # Expand the model's memory to allow reading large documents
        )
        self.summary_writer = raw_model.with_structured_output(SummaryOutput)
        self.critic_reviewer = raw_model.with_structured_output(CriticOutput)
        self.data = data

    #nodes
    def analysist(self, state):
        console = Console()
        data_text = "\n".join([doc.page_content for doc in self.data]) if isinstance(self.data, list) else self.data
        
        # Grab the last message. If it's a REJECTION, show it to the Analyst.
        last_message = state["messages"][-1].content if state["messages"] else ""
        feedback = f"\n\nCRITIC FEEDBACK: {last_message}" if "REJECTED" in last_message else ""
        
        # print(f"--- RAW TEXT PREVIEW ---\n{data_text[:500]}\n-----------------------")

        prompt = f"""
        Summarize the following research paper. 
        Focus on the TECHNICAL ARCHITECTURE and the QUANTITATIVE RESULTS.
        
        STRUCTURE:
        - Core Contribution: (5-6 sentences)
        - Key Metrics/Results: (Specific percentages or improvements)
        - Methodology: (Algorithm names used) {data_text} {feedback}
        If there is feedback above, ensure you incorporate the missing details into your new summary."""
        
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
        paper_text = "\n".join([doc.page_content for doc in self.data])

        prompt = f"""Compare this SUMMARY to the ORIGINAL PAPER.
        SUMMARY: {analysist_message}
        ORIGINAL PAPER: {paper_text} 
        You are a harsh Peer Reviewer. Compare this summary to the original text. \
        If the summary is missing specific metrics, names of algorithms, or actual results, \
        you MUST set is_approved to False. Be pedantic."""
        
        with console.status("[bold red]Critic is looking for flaws...", spinner="dots"):
            start = time.time()
            response = self.critic_reviewer.invoke([HumanMessage(content=prompt)])
            elapsed = time.time() - start
            console.print(f"[red]✓ Critic done in {elapsed:.1f}s")
        
        status = "REJECTED" if not response.is_approved else "APPROVED"
        return {"messages": [AIMessage(content=f"{status}: {response.critique}")]}



    def refiner(self, state):
        console = Console()
        # Extract the last message using .content attribute (not dictionary subscripting)
        critic_message = state["messages"][-1].content

        full_history = "\n".join([m.content for m in state["messages"]])

        prompt = f"""
        Review the summary and the Critic's feedback below.
        Produce a final technical summary that is dense with information.
        
        CRITICAL RULES:
        1. REMOVE 'filler' phrases (e.g., 'In this paper', 'The researchers found').
        2. START with the technology itself.
        3. ENSURE all metrics mentioned by the Critic are included.
        4. Use 5-10 sentences to ensure depth.
        
        HISTORY: 
        {full_history}"""
        
        with console.status("[bold green]Refiner is polishing the final summary...", spinner="dots"):
            start = time.time()
            response = self.summary_writer.invoke([HumanMessage(content=prompt)])
            elapsed = time.time() - start
            console.print(f"[green]✓ Refiner done in {elapsed:.1f}s")
        
        return {"messages": [AIMessage(content=f"Refiner: {response.summary}")]}



