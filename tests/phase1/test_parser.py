from agentverse.parser import OutputParser, OutputParserError

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
