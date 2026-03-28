from langchain_docling.loader import DoclingLoader
from orchestration import Orchestration
from agents import Agents


def main():

    FILE_PATH = "test_search.pdf"
    print(f"Loading document from: {FILE_PATH}")
    loader = DoclingLoader(file_path=FILE_PATH)
    data = loader.load()
    print("Loaded document content")
    agents = Agents(data)
    orchestration = Orchestration()
    orchestration.add_node('analysist', agents.analysist)
    orchestration.add_node('critic', agents.critic)
    orchestration.add_node('refiner', agents.refiner)
    orchestration.add_edge('analysist', 'critic', 'refiner')
    
    # Run the graph - flows through all nodes
    result = orchestration.run()
    
    # Print all messages from the result
    print("\n" + "="*60)
    print("GRAPH EXECUTION RESULTS")
    print("="*60)
    for i, msg in enumerate(result["messages"]):
        # Messages are LangChain Message objects with type and content attributes
        msg_type = type(msg).__name__
        print(f"\nMessage {i}: {msg_type.upper()}")
        print(f"Content: {msg.content[:200]}...")  # Show first 200 chars


if __name__ == "__main__":
    main()