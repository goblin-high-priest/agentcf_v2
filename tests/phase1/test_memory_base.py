from agentverse.memory.base import BaseMemory
import pytest

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

