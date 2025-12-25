from agentverse.registry import Registry
from agentverse.message import Message
from agentverse.utils import AgentAction
from agentverse.llms import LLMResult
import pytest

# ==================== 集成测试 ====================
class TestIntegration:
    """集成测试：测试模块之间的协作"""
    
    def test_registry_with_message(self):
        """测试注册表与消息类的集成"""
        registry = Registry(name="MessageRegistry")
        
        @registry.register("message")
        def create_message(content: str):
            return Message(content=content)
        
        msg = registry.build("message", content="Hello")
        assert isinstance(msg, Message)
        assert msg.content == "Hello"
    
    def test_message_with_agent_action(self):
        """测试消息与AgentAction的集成"""
        action = AgentAction(tool="test", tool_input="input", log="log")
        msg = Message(
            content="Test",
            tool_response=[(action, "response")]
        )
        assert len(msg.tool_response) == 1
        assert msg.tool_response[0][0].tool == "test"
    
    def test_llm_result_structure(self):
        """测试LLMResult的结构"""
        result = LLMResult(
            content="Response",
            send_tokens=100,
            recv_tokens=50,
            total_tokens=150
        )
        # 验证所有字段
        assert result.content is not None
        assert result.send_tokens >= 0
        assert result.recv_tokens >= 0
        assert result.total_tokens == result.send_tokens + result.recv_tokens


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    