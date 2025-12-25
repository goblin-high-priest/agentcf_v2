import pytest
import sys
import os

# 确保能导入 agentverse 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from agentverse.llms.base import LLMResult
from agentverse.parser import OutputParserError
from agentverse.tasks.recommendation.output_parser import (
    RecommenderParser,
    UserAgentParser,
    ItemAgentParser
)

# ==========================================
# Test RecommenderParser
# ==========================================

def test_recommender_parser_parse_success():
    parser = RecommenderParser()
    # 模拟 LLM 输出，包含 Choice 和 Explanation
    content = """
    Some thought process...
    Choice: Item A
    Explanation: Because it is good.
    """
    result = LLMResult(content=content, send_tokens=0, recv_tokens=0, total_tokens=0)
    
    ans, rat = parser.parse(result)
    assert ans == "Item A"
    assert rat == "Because it is good."

def test_recommender_parser_parse_failure_missing_keywords():
    parser = RecommenderParser()
    # 缺少 Choice 或 Explanation
    content = "Just some random text without keywords."
    result = LLMResult(content=content, send_tokens=0, recv_tokens=0, total_tokens=0)
    
    with pytest.raises(OutputParserError):
        parser.parse(result)

def test_recommender_parser_backward():
    parser = RecommenderParser()
    content = """
    Analysis of the situation...
    Updated Strategy: Focus on more recent items.
    """
    result = LLMResult(content=content, send_tokens=0, recv_tokens=0, total_tokens=0)
    
    strategy = parser.parse_backward(result)
    assert strategy == "Focus on more recent items."

def test_recommender_parser_evaluation():
    parser = RecommenderParser()
    content = """
    Rank:
    Item 1
    Item 2
    Item 3
    """
    result = LLMResult(content=content, send_tokens=0, recv_tokens=0, total_tokens=0)
    
    ranking = parser.parse_evaluation(result)
    assert ranking == ["Item 1", "Item 2", "Item 3"]

# ==========================================
# Test UserAgentParser
# ==========================================

def test_user_agent_parser_simple():
    parser = UserAgentParser()
    content = "   This is a user response.   \n\n"
    result = LLMResult(content=content, send_tokens=0, recv_tokens=0, total_tokens=0)
    
    parsed = parser.parse(result)
    assert parsed == "This is a user response."

def test_user_agent_parser_update():
    parser = UserAgentParser()
    content = """
    My thought process...
    My updated self-introduction: I am a fan of rock music.
    """
    result = LLMResult(content=content, send_tokens=0, recv_tokens=0, total_tokens=0)
    
    updated_intro = parser.parse_update(result)
    assert updated_intro == "I am a fan of rock music."

# ==========================================
# Test ItemAgentParser
# ==========================================

def test_item_agent_parser_parse_success():
    parser = ItemAgentParser()
    content = (
        "The updated description of the first CD is: Description 1\n"
        "The updated description of the second CD is: Description 2"
    )
    result = LLMResult(content=content, send_tokens=0, recv_tokens=0, total_tokens=0)

    ans, rat = parser.parse(result)
    assert ans == "Description 1"
    assert rat == "Description 2"

def test_item_agent_parser_parse_failure():
    parser = ItemAgentParser()
    content = "Invalid format without proper keys."
    result = LLMResult(content=content, send_tokens=0, recv_tokens=0, total_tokens=0)
    
    with pytest.raises(OutputParserError):
        parser.parse(result)

def test_item_agent_parser_pretrain():
    parser = ItemAgentParser()
    content = "CD Description: A great album by Artist X."
    result = LLMResult(content=content, send_tokens=0, recv_tokens=0, total_tokens=0)
    
    desc = parser.parse_pretrain(result)
    assert desc == "A great album by Artist X."