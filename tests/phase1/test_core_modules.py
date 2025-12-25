"""
测试核心基础模块的功能
测试文件: tests/test_core_modules.py
"""
import pytest
from typing import Set
from agentverse.registry import Registry
from agentverse.message import Message
from agentverse.parser import OutputParser, OutputParserError
from agentverse.utils import AgentAction, AgentFinish
from agentverse.memory.base import BaseMemory
from agentverse.agents.base import BaseAgent
from agentverse.llms.base import BaseLLM, LLMResult, BaseModelArgs


# ==================== 测试 Registry ====================
class TestRegistry:
    """测试注册表系统"""
    
    def test_registry_creation(self):
        """测试创建注册表"""
        registry = Registry(name="TestRegistry")
        assert registry.name == "TestRegistry"
        assert registry.entries == {}
    
    def test_register_decorator(self):
        """测试注册装饰器"""
        registry = Registry(name="TestRegistry")
        
        @registry.register("test_class")
        class TestClass:
            def __init__(self, value):
                self.value = value
        
        assert "test_class" in registry.entries
        assert registry.entries["test_class"] == TestClass
    
    def test_build_registered_class(self):
        """测试构建已注册的类"""
        registry = Registry(name="TestRegistry")
        
        @registry.register("test_class")
        class TestClass:
            def __init__(self, value):
                self.value = value
        
        instance = registry.build("test_class", value=42)
        assert instance.value == 42
    
    def test_build_unregistered_class(self):
        """测试构建未注册的类应该抛出错误"""
        registry = Registry(name="TestRegistry")
        
        with pytest.raises(ValueError) as exc_info:
            registry.build("non_existent")
        
        assert "not registered" in str(exc_info.value)
    
    def test_get_all_entries(self):
        """测试获取所有条目"""
        registry = Registry(name="TestRegistry")
        
        @registry.register("class1")
        class Class1:
            pass
        
        @registry.register("class2")
        class Class2:
            pass
        
        entries = registry.get_all_entries()
        assert len(entries) == 2
        assert "class1" in entries
        assert "class2" in entries


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


# ==================== 测试 Utils ====================
class TestUtils:
    """测试工具函数"""
    
    def test_agent_action_creation(self):
        """测试创建AgentAction"""
        action = AgentAction(
            tool="search",
            tool_input="query",
            log="Searching..."
        )
        assert action.tool == "search"
        assert action.tool_input == "query"
        assert action.log == "Searching..."
    
    def test_agent_action_tuple_access(self):
        """测试AgentAction作为元组访问"""
        action = AgentAction(tool="test", tool_input={"key": "value"}, log="log")
        # NamedTuple支持索引访问
        assert action[0] == "test"
        assert action[1] == {"key": "value"}
        assert action[2] == "log"
    
    def test_agent_finish_creation(self):
        """测试创建AgentFinish"""
        finish = AgentFinish(
            return_values={"result": "success"},
            log="Completed"
        )
        assert finish.return_values == {"result": "success"}
        assert finish.log == "Completed"
    
    def test_agent_finish_tuple_access(self):
        """测试AgentFinish作为元组访问"""
        finish = AgentFinish(return_values={"a": 1}, log="done")
        assert finish[0] == {"a": 1}
        assert finish[1] == "done"


# ==================== 测试 Parser ====================
class TestParser:
    """测试解析器基类"""
    
    def test_output_parser_error(self):
        """测试OutputParserError异常"""
        error = OutputParserError("Test error message")
        assert error.message == "Test error message"
        assert "Test error message" in str(error)
    
    def test_output_parser_is_abstract(self):
        """测试OutputParser是抽象类,不能直接实例化"""
        # OutputParser是抽象基类,应该不能直接实例化
        # 但我们可以在测试中检查它需要实现parse方法
        from abc import ABC
        
        assert issubclass(OutputParser, ABC)
        assert hasattr(OutputParser, 'parse')


# ==================== 测试 BaseMemory ====================
class TestBaseMemory:
    """测试内存基类"""
    
    def test_base_memory_is_abstract(self):
        """测试BaseMemory是抽象类"""
        from abc import ABC
        
        assert issubclass(BaseMemory, ABC)
        # 不能直接实例化抽象类
        with pytest.raises(TypeError):
            BaseMemory()
    
    def test_base_memory_has_required_methods(self):
        """测试BaseMemory有必需的方法"""
        assert hasattr(BaseMemory, 'add_message')
        assert hasattr(BaseMemory, 'to_string')
        assert hasattr(BaseMemory, 'reset')


# ==================== 测试 BaseLLM ====================
class TestBaseLLM:
    """测试LLM基类"""
    
    def test_llm_result_creation(self):
        """测试创建LLMResult"""
        result = LLMResult(
            content="Test response",
            send_tokens=10,
            recv_tokens=20,
            total_tokens=30
        )
        assert result.content == "Test response"
        assert result.send_tokens == 10
        assert result.recv_tokens == 20
        assert result.total_tokens == 30
    
    def test_base_model_args(self):
        """测试BaseModelArgs"""
        args = BaseModelArgs()
        assert isinstance(args, BaseModelArgs)
    
    def test_base_llm_is_abstract(self):
        """测试BaseLLM是抽象类"""
        from abc import ABC
        
        assert issubclass(BaseLLM, ABC)
        with pytest.raises(TypeError):
            BaseLLM()
    
    def test_base_llm_has_required_methods(self):
        """测试BaseLLM有必需的方法"""
        assert hasattr(BaseLLM, 'generate_response')
        assert hasattr(BaseLLM, 'agenerate_response')
    
    def test_base_chat_model_inheritance(self):
        """测试BaseChatModel继承自BaseLLM"""
        assert issubclass(BaseChatModel, BaseLLM)
    
    def test_base_completion_model_inheritance(self):
        """测试BaseCompletionModel继承自BaseLLM"""
        assert issubclass(BaseCompletionModel, BaseLLM)


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
    