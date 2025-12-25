from agentverse.utils import AgentAction
from agentverse.message import Message
from typing import Set

# ==================== 测试 Message ====================
class TestMessage:
    """测试消息类"""
    
    def test_message_creation_default(self):
        """测试使用默认值创建消息"""
        msg = Message()
        assert msg.content == ""
        assert msg.sender == ""
        assert msg.receiver == {"all"}
        assert msg.tool_response == []
    
    def test_message_creation_custom(self):
        """测试使用自定义值创建消息"""
        action = AgentAction(tool="test_tool", tool_input="input", log="log")
        msg = Message(
            content="Hello",
            sender="Alice",
            receiver={"Bob", "Charlie"},
            tool_response=[(action, "response")]
        )
        assert msg.content == "Hello"
        assert msg.sender == "Alice"
        assert msg.receiver == {"Bob", "Charlie"}
        assert len(msg.tool_response) == 1
    
    def test_message_receiver_set(self):
        """测试接收者集合操作"""
        msg = Message(receiver={"user1"})
        assert isinstance(msg.receiver, Set)
        assert "user1" in msg.receiver

