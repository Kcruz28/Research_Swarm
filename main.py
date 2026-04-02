from langchain_docling.loader import DoclingLoader
from orchestration import Orchestration
from agents import Agents
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from langchain_core.messages import HumanMessage
from pdf_reader import PDFReader


def main():

    FILE_PATH = "IEEE Xplore Full-Text PDF_.pdf"
    print(f"Loading document from: {FILE_PATH}")
    data = PDFReader(file_path=FILE_PATH).load_pdf()
    agents = Agents(data)
    orchestration = Orchestration()
    orchestration.add_node('analysist', agents.analysist)
    orchestration.add_node('critic', agents.critic)
    orchestration.add_node('refiner', agents.refiner)
    orchestration.add_edges('analysist', 'critic', 'refiner')
    
    # Run the graph - flows through all nodes
    result = orchestration.run()
    
    console = Console()

    print("\n" + "="*60)
    print("RESEARCH SWARM EXECUTION TRACE")
    print("="*60)

    # 1. Create a DYNAMIC table that shows the full "argument" history
    table = Table(title="Agent Conversation History", show_lines=True)
    table.add_column("Step", style="cyan", no_wrap=True)
    table.add_column("Agent", style="bold green")
    table.add_column("Content Preview", style="magenta")

    step_num = 1
    final_refiner_text = ""

    for msg in result["messages"]:
        # Skip the initial user prompt
        if isinstance(msg, HumanMessage):
            continue
        
        content = msg.content
        
        # Identify the agent and set colors
        if "Analyst:" in content:
            agent_name = "Analyst"
            style = "blue"
        elif "REJECTED:" in content:
            agent_name = "Critic (REJECT)"
            style = "red"
        elif "APPROVED:" in content:
            agent_name = "Critic (PASS)"
            style = "green"
        elif "Refiner:" in content:
            agent_name = "Refiner"
            style = "bold yellow"
            final_refiner_text = content.replace("Refiner: ", "") # Store for Markdown
        else:
            agent_name = "System"
            style = "white"

        # Add to the trace table (truncating long text for the history view)
        preview = (content[:150] + "...") if len(content) > 150 else content
        table.add_row(f"Step {step_num}", f"[{style}]{agent_name}[/{style}]", preview)
        step_num += 1

    console.print(table)

    # 2. THE MARKDOWN LOGIC: Display the polished final product
    if final_refiner_text:
        console.print("\n" + "═"*60, style="bold yellow")
        console.print("  FINAL RESEARCH SUMMARY (DISTILLED)", style="bold yellow")
        console.print("═"*60 + "\n", style="bold yellow")
        
        # This renders the bold headers and structured text beautifully
        md = Markdown(final_refiner_text)
        console.print(md)
        console.print("\n" + "═"*60, style="bold yellow")
    else:
        console.print("[bold red]Error: No Refiner output found in the message history.[/bold red]")

if __name__ == "__main__":
    main()