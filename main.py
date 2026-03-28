from orchestration import Orchestration
from agents import Agents


def main():
    agents = Agents()
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
        print(f"\nMessage {i}: {msg.get('role', 'unknown').upper()}")
        print(f"Content: {msg.get('content', 'No content')[:200]}...")  # Show first 200 chars


if __name__ == "__main__":
    main()