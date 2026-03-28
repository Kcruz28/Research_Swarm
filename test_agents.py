"""
Test cases for the Research Swarm agents.
Run with: pytest test_agents.py -v
"""

import pytest
from agents import Agents
from langchain_core.messages import AIMessage, HumanMessage


class TestAgents:
    """Test suite for Agents class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create mock data for testing"""
        # Simulate Document objects with page_content attribute
        class MockDocument:
            def __init__(self, content):
                self.page_content = content
        
        return [MockDocument("The Transformer model is based on attention mechanisms.")]
    
    @pytest.fixture
    def agents(self, sample_data):
        """Initialize agents with sample data"""
        return Agents(sample_data)
    
    # Test 1: Initialization
    def test_agents_initialization(self, agents):
        """Test that Agents initializes correctly"""
        assert agents.model is not None
        assert agents.data is not None
        assert agents.model.model == "qwen3.5:0.8b"
    
    # Test 2: Data extraction
    def test_data_extraction(self, agents):
        """Test that data is extracted correctly"""
        data_text = "\n".join([doc.page_content for doc in agents.data])
        assert len(data_text) > 0
        assert "Transformer" in data_text
    
    # Test 3: Analysist node returns proper format
    def test_analysist_returns_messages(self, agents):
        """Test that analysist returns proper message format"""
        state = {"messages": [HumanMessage(content="Test paper")]}
        result = agents.analysist(state)
        
        # Check structure
        assert "messages" in result
        assert isinstance(result["messages"], list)
        assert len(result["messages"]) > 0
        assert isinstance(result["messages"][-1], AIMessage)
    
    # Test 4: Critic node processes previous output
    def test_critic_uses_analyst_output(self, agents):
        """Test that critic processes analyst's output"""
        analyst_msg = AIMessage(content="Transformer uses attention mechanisms")
        state = {"messages": [HumanMessage(content="Test"), analyst_msg]}
        
        result = agents.critic(state)
        
        assert "messages" in result
        assert isinstance(result["messages"][-1], AIMessage)
        # Check that content was processed (should contain "Critic:")
        assert "Critic:" in result["messages"][-1].content
    
    # Test 5: Refiner node processes critic output  
    def test_refiner_uses_critic_output(self, agents):
        """Test that refiner processes critic's output"""
        critic_msg = AIMessage(content="Missing key details about scalability")
        state = {"messages": [HumanMessage(content="Test"), critic_msg]}
        
        result = agents.refiner(state)
        
        assert "messages" in result
        assert isinstance(result["messages"][-1], AIMessage)
        assert "Refiner:" in result["messages"][-1].content
    
    # Test 6: Error handling - empty data
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        empty_agents = Agents([])
        state = {"messages": [HumanMessage(content="Test")]}
        
        # Should not crash, should handle gracefully
        result = empty_agents.analysist(state)
        assert "messages" in result


class TestIntegration:
    """Integration tests for the full pipeline"""
    
    @pytest.fixture
    def sample_data(self):
        """Create mock data for testing"""
        class MockDocument:
            def __init__(self, content):
                self.page_content = content
        
        return [MockDocument("Deep learning uses neural networks.")]
    
    def test_full_pipeline_flow(self, sample_data):
        """Test that all agents work together"""
        agents = Agents(sample_data)
        
        # Start state
        state = {"messages": [HumanMessage(content="Analyze this")]}
        
        # Run through pipeline
        analyst_result = agents.analysist(state)
        assert len(analyst_result["messages"]) > 1
        
        critic_result = agents.critic(analyst_result)
        assert len(critic_result["messages"]) > len(analyst_result["messages"])
        
        refiner_result = agents.refiner(critic_result)
        assert len(refiner_result["messages"]) > len(critic_result["messages"])


class TestPrompts:
    """Test that prompts are formatted correctly"""
    
    @pytest.fixture
    def sample_data(self):
        class MockDocument:
            def __init__(self, content):
                self.page_content = content
        
        return [MockDocument("Test content")]
    
    def test_analyst_prompt_includes_data(self, sample_data):
        """Test that analyst prompt includes the document data"""
        agents = Agents(sample_data)
        # The prompt should be built with the data
        assert agents.data is not None
    
    def test_message_extraction(self):
        """Test that message content can be extracted correctly"""
        msg = AIMessage(content="Sample output")
        assert msg.content == "Sample output"
        assert hasattr(msg, 'content')
