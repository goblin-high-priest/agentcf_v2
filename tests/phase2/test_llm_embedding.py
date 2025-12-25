import pytest
from unittest.mock import MagicMock
from agentverse.llms.openai import OpenAIEmbedding

def test_embedding_sync():
    # 1. 构造 embedding API 的 mock response
    mock_item = MagicMock()
    mock_item.embedding = [0.1, 0.2]

    mock_response = MagicMock()
    mock_response.data = [mock_item]

    # 2. 模拟 pool.client.embeddings.create 返回 mock_response
    fake_pool = MagicMock()
    fake_pool.client.embeddings.create.return_value = mock_response

    # 3. 注入 pool
    embedder = OpenAIEmbedding(api_key_list=["dummy"])
    embedder.pool = fake_pool

    # 4. 调用 generate_response
    result = embedder.generate_response("text")

    # 5. 断言 embedding 正确返回
    assert result.content == [0.1, 0.2]

    # 6. 断言 API 被正确调用
    fake_pool.client.embeddings.create.assert_called_once()
