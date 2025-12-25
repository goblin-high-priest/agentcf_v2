# tests/test_llm_client_pool.py
import os
import pytest
from agentverse.llms.openai import OpenAIClientPool

@pytest.fixture
def pool(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return OpenAIClientPool(["sk-test"])

def test_pool_initialization(pool):
    pool.ensure_clients()
    assert pool.client is not None
    assert pool.async_client is not None

def test_pool_rotation(pool):
    pool.api_key_list = ["key1", "key2"]
    pool.ensure_clients()
    first_client = pool.client
    pool.rotate_key()
    assert pool.client is not first_client