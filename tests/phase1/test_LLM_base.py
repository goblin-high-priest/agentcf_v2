from agentverse.llms.base import BaseLLM, LLMResult, BaseModelArgs, BaseChatModel, BaseCompletionModel
import pytest

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

