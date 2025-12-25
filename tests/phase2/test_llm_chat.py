from unittest.mock import MagicMock
from agentverse.llms.openai import OpenAICompletion
from agentverse.llms.base import LLMResult

def test_completion_sync_chat_mode():
    # 1. 构造 Chat API 格式的 mock response
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Hello Answer"
    mock_response.choices = [mock_choice]

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 3
    mock_usage.completion_tokens = 7
    mock_usage.total_tokens = 10
    mock_response.usage = mock_usage

    # 2. 构造假的 pool，模拟 Chat API 调用
    fake_pool = MagicMock()
    fake_pool.client.chat.completions.create.return_value = mock_response

    # 3. 创建模型实例并注入假 pool
    llm = OpenAICompletion(api_key_list=["fake"])
    llm.pool = fake_pool

    # 4. 调用 generate_response
    result = llm.generate_response("hi")

    # 5. 断言内容正确
    assert isinstance(result, LLMResult)
    assert result.content == "Hello Answer"
    assert result.send_tokens == 3
    assert result.recv_tokens == 7
    assert result.total_tokens == 10
