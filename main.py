from langchain_docling.loader import DoclingLoader
from orchestration import Orchestration
from agents import Agents
from rich.console import Console
from rich.table import Table
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
    
    # Print all messages from the result
    print("\n" + "="*60)
    print("GRAPH EXECUTION RESULTS")
    print("="*60)
    
    # Create a table to display the results
    table = Table(title="Research Analysis Pipeline")
    table.add_column("Agent", style="cyan")
    table.add_column("Output", style="magenta")
    
    # Add rows for each agent's output (skip the initial HumanMessage)
    agent_names = ["Analyst", "Critic", "Refiner"]
    for i, name in enumerate(agent_names):
        msg_content = result["messages"][i + 1].content  # Skip first message (HumanMessage)
        # Truncate long content for display
        display_content = msg_content
        table.add_row(name, display_content)
    
    console = Console()
    console.print(table)

if __name__ == "__main__":
    main()