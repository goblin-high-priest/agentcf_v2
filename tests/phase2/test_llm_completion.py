import pytest
from unittest.mock import MagicMock
from agentverse.llms.openai import OpenAICompletion

def test_completion_sync():
    # 1. 构造 Chat API 风格的 mock response
    mock_choice = MagicMock()
    mock_choice.message.content = "Hello"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 30

    # 2. 模拟 pool.client.chat.completions.create
    fake_pool = MagicMock()
    fake_pool.client.chat.completions.create.return_value = mock_response

    completion = OpenAICompletion(api_key_list=["dummy"])
    completion.pool = fake_pool

    result = completion.generate_response("Test")

    assert result.content == "Hello"
    fake_pool.client.chat.completions.create.assert_called_once()
