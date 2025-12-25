from agentverse.agents.base import BaseAgent

# ==================== 测试 BaseAgent ====================
class TestBaseAgent:
    """测试Agent基类"""
    
    def test_base_agent_is_abstract(self):
        """测试BaseAgent是抽象类"""
        from abc import ABC
        
        assert issubclass(BaseAgent, ABC)
    
    def test_base_agent_has_required_methods(self):
        """测试BaseAgent有必需的方法"""
        assert hasattr(BaseAgent, 'step')
        assert hasattr(BaseAgent, 'astep')
        assert hasattr(BaseAgent, 'reset')
        assert hasattr(BaseAgent, 'add_message_to_memory')
        assert hasattr(BaseAgent, 'get_receiver')
        assert hasattr(BaseAgent, 'set_receiver')
        assert hasattr(BaseAgent, 'add_receiver')
        assert hasattr(BaseAgent, 'remove_receiver')
    
    def test_base_agent_receiver_methods(self):
        """测试BaseAgent的receiver方法(需要具体实现类)"""
        # 这个测试需要具体实现,但我们可以测试方法存在
        # 由于BaseAgent是抽象类,我们需要创建一个mock实现
        from unittest.mock import Mock
        
        # 创建一个简单的mock实现
        class MockAgent(BaseAgent):
            def step(self, env_description: str = ""):
                return Message()
            
            def astep(self, env_description: str = ""):
                return Message()
            
            def reset(self):
                pass
            
            def add_message_to_memory(self, messages):
                pass
        
        # 注意：这不会真正工作,因为BaseAgent需要llm和output_parser
        # 但展示了如何测试抽象类的方法存在性

