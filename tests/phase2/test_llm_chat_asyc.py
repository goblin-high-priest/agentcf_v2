import pytest
from unittest.mock import MagicMock, AsyncMock
from agentverse.llms.openai import OpenAIChat

@pytest.mark.asyncio
async def test_chat_async():
    # 1. 构造 fake Chat API response
    mock_choice = MagicMock()
    mock_choice.message.content = "Hi"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    # 2. 模拟 async_client
    fake_pool = MagicMock()
    fake_pool.async_client.chat.completions.create = AsyncMock(return_value=mock_response)

    # 3. 创建模型并注入 fake_pool
    chat = OpenAIChat(api_key_list=["dummy"])
    chat.pool = fake_pool

    # 4. 调用 async 方法
    result = await chat.agenerate_response("hi")

    # 5. 验证 async 方法被 await 调用
    fake_pool.async_client.chat.completions.create.assert_awaited_once()

    # 6. 验证返回数据符合 Chat API 的 async 格式
    assert result == ["Hi"]
