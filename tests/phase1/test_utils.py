from agentverse.utils import AgentAction, AgentFinish

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
