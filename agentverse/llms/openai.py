"""
OpenAI v1+ 客户端封装
"""
from __future__ import annotations

import os
import time
import asyncio
from typing import Dict, List, Optional, Sequence, Union

from openai import OpenAI, AsyncOpenAI
from openai import APIError, APIConnectionError, RateLimitError
from pydantic import BaseModel, Field, ConfigDict

from agentverse.llms.base import LLMResult, BaseChatModel, BaseCompletionModel, BaseModelArgs
from agentverse.llms import llm_registry

import logging

logger = logging.getLogger(__name__)

try:
    import httpx
except ImportError:  # 允许无代理环境
    httpx = None

class OpenAIChatArgs(BaseModelArgs):
    model: str = Field(default="gpt-4o")
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=1.0)
    top_p: int = Field(default=1)
    n: int = Field(default=1)
    stop: Optional[Union[str, List]] = Field(default=None)
    presence_penalty: int = Field(default=0)
    frequency_penalty: int = Field(default=0)

# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------

class OpenAIClientConfig(BaseModel):
    api_base: Optional[str] = Field(default_factory=lambda: os.environ.get("api_base"))
    http_proxy: Optional[str] = Field(default_factory=lambda: os.environ.get("http_proxy"))

    def build_http_client(self) -> Optional[httpx.Client]:
        if not self.http_proxy or not httpx:
            return None
        return httpx.Client(proxy=self.http_proxy)
    
    def build_async_http_client(self) -> Optional[httpx.AsyncClient]:
        if not self.http_proxy or not httpx:
            return None
        return httpx.AsyncClient(proxy=self.http_proxy)


class OpenAIClientPool:
    """
    管理多个 API key 的客户端池，负责：
    - 初始化同步/异步客户端
    - 轮换 key(rate limit / quota 时）
    """
    def __init__(self, api_key_list: Sequence[str]):
        if not api_key_list:
            raise ValueError("api_key_list 不能为空，请在配置中提供至少一个 OpenAI API key")
        self.api_key_list = list(api_key_list)
        self.idx = 0
        self.config = OpenAIClientConfig()

        # 当前持有客户端
        self.client: Optional[OpenAI] = None
        self.async_client: Optional[AsyncOpenAI] = None

    def _current_key(self) -> str:
        return self.api_key_list[self.idx % len(self.api_key_list)]

    def _build_clients(self, api_key: str):
        http_client = self.config.build_http_client()
        async_http_client = self.config.build_async_http_client()

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.config.api_base,
            http_client=http_client,
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.config.api_base,
            http_client=async_http_client,
        )

    def ensure_clients(self):
        if self.client is None or self.async_client is None:
            self._build_clients(self._current_key())

    def rotate_key(self):
        """切换到下一个 key, 并重建客户端"""
        self.idx = (self.idx + 1) % len(self.api_key_list)
        self._build_clients(self._current_key())

    def handle_api_error(self, error: Exception) -> bool:
        """处理常见 API 错误。返回值：是否已经处理（并可重试）"""
        error_str = str(error)
        if "quota" in error_str or "deactivated" in error_str:
            bad_key = self._current_key()
            print(f"[OpenAI] key 被封禁或额度耗尽: {bad_key}")
            if bad_key in self.api_key_list:
                self.api_key_list.remove(bad_key)
            if not self.api_key_list:
                raise ValueError("所有 API key 都不可用，请更新配置")
            self.rotate_key()
            return True
        if "context length" in error_str:
            logger.warning("上下文超出限制，可以考虑缩短 prompt")
            return False
        return False


# ---------------------------------------------------------------------------
# Completion / Embedding 基类
# ---------------------------------------------------------------------------

class OpenAIBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    args: BaseModelArgs
    max_retry: int = 3
    pool: OpenAIClientPool

    def _run_with_retry(self, func, *args, **kwargs):
        """同步调用，带重试"""
        for attempt in range(self.max_retry):
            try:
                self.pool.ensure_clients()
                return func(*args, **kwargs)
            except (APIError, APIConnectionError, RateLimitError) as e:
                should_retry = self.pool.handle_api_error(e)
                if should_retry:
                    continue
                raise
            except Exception as e:
                logger.error(f"OpenAI 调用失败: {e}")
                time.sleep(2 ** attempt)
        raise RuntimeError("多次重试后仍失败")


    async def _arun_with_retry(self, coro_builder):
        """异步调用，带重试"""
        for attempt in range(self.max_retry):
            try:
                self.pool.ensure_clients()
                coro = coro_builder()
                return await coro
            except (APIError, APIConnectionError, RateLimitError) as e:
                should_retry = self.pool.handle_api_error(e)
                if should_retry:
                    continue
                raise
            except Exception as e:
                logger.error(f"OpenAI 异步调用失败: {e}")
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError("多次重试后仍失败")


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------

class OpenAICompletionArgs(OpenAIChatArgs):
    model: str = Field(default="gpt-4o")
    best_of: int = Field(default=1)

@llm_registry.register("text-davinci-003")  # 兼容旧配置
@llm_registry.register("gpt-4o")
class OpenAICompletion(OpenAIBaseModel, BaseCompletionModel):
    args: OpenAICompletionArgs = Field(default_factory=OpenAICompletionArgs)

    def __init__(self, api_key_list: Sequence[str], max_retry: int = 3, **kwargs):

        # 1. 处理旧模型（合理）
        if kwargs.get("model") == "text-davinci-003":
            kwargs["model"] = "gpt-4o"

        # 2. 获取默认值
        default = OpenAICompletionArgs().model_dump()

        # 3. 合并参数（用户覆盖默认）
        merged = {**default, **kwargs}

        # 4. 构建最终 args（Pydantic 会自动校验）
        args = OpenAICompletionArgs(**merged)

        # 5. 调用父类构造函数
        super().__init__(
            args=args,
            max_retry=max_retry,
            pool=OpenAIClientPool(api_key_list),
        )

        self.args = args


    def generate_response(self, prompt: str) -> LLMResult:
        def _call():
            # 将 Completion prompt 转换为 Chat message
            messages = [{"role": "user", "content": prompt}]
            
            # 使用 Chat 接口
            response = self.pool.client.chat.completions.create(
                model=self.args.model,
                messages=messages,
                **{k: v for k, v in self.args.model_dump().items() if k != "model" and k != "best_of"},
            )
            return response

        response = self._run_with_retry(_call)
        return LLMResult(
            content=response.choices[0].message.content,
            send_tokens=response.usage.prompt_tokens,
            recv_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

    async def agenerate_response(self, prompt: str):
        async def _call():
            messages = [{"role": "user", "content": prompt}]
            return await self.pool.async_client.chat.completions.create(
                model=self.args.model,
                messages=messages,
                **{k: v for k, v in self.args.model_dump().items() if k != "model" and k != "best_of"},
            )

        response = await self._arun_with_retry(_call)
        return [choice.message.content for choice in response.choices]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

@llm_registry.register("embedding")
class OpenAIEmbedding(OpenAIBaseModel, BaseCompletionModel):
    def __init__(self, api_key_list: Sequence[str], max_retry: int = 3, **kwargs):
        super().__init__(
            args=BaseModelArgs(),
            max_retry=max_retry,
            pool=OpenAIClientPool(api_key_list),
        )

    def generate_response(self, prompt: str) -> LLMResult:
        def _call():
            return self.pool.client.embeddings.create(
                model="text-embedding-ada-002",
                input=prompt,
            )

        response = self._run_with_retry(_call)
        return LLMResult(
            content=response.data[0].embedding,
            send_tokens=0,
            recv_tokens=0,
            total_tokens=0,
        )

    async def agenerate_response(self, sentences: List[str]):
        async def _call():
            tasks = [
                self.pool.async_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=sentence,
                )
                for sentence in sentences
            ]
            return await asyncio.gather(*tasks)

        responses = await self._arun_with_retry(_call)
        # 返回 dict 列表，与旧代码兼容
        return [resp.model_dump() for resp in responses]


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@llm_registry.register("gpt-3.5-turbo-16k-0613")
@llm_registry.register("gpt-3.5-turbo")
@llm_registry.register("gpt-4")
class OpenAIChat(OpenAIBaseModel, BaseChatModel):
    args: OpenAIChatArgs = Field(default_factory=OpenAIChatArgs)

    def __init__(self, api_key_list: Sequence[str], max_retry: int = 3, **kwargs):
        # 1. 获取默认值
        default = OpenAIChatArgs().model_dump()
        
        # 2. 合并参数
        merged = {**default, **kwargs}
        
        # 3. 构建 Pydantic 对象
        args = OpenAIChatArgs(**merged)

        super().__init__(
            args=args,
            max_retry=max_retry,
            pool=OpenAIClientPool(api_key_list),
        )
        self.args = args

    def _build_messages(self, prompts: Sequence[str]):
        return [[{"role": "user", "content": p}] for p in prompts]

    def generate_response(self, prompt: str) -> LLMResult:
        def _call():
            messages = self._build_messages([prompt])[0]
            response = self.pool.client.chat.completions.create(
                model=self.args.model,
                messages=messages,
                **{k: v for k, v in self.args.model_dump().items() if k != "model"},
            )
            return response

        response = self._run_with_retry(_call)
        return LLMResult(
            content=response.choices[0].message.content,
            send_tokens=response.usage.prompt_tokens,
            recv_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

    async def agenerate_response(self, prompt: str):
        async def _call():
            messages = self._build_messages([prompt])[0]
            return await self.pool.async_client.chat.completions.create(
                model=self.args.model,
                messages=messages,
                **{k: v for k, v in self.args.model_dump().items() if k != "model"},
            )

        response = await self._arun_with_retry(_call)
        return [choice.message.content for choice in response.choices]

    async def agenerate_response_without_construction(self, messages: List[List[Dict[str, str]]]):
        async def _call():
            tasks = [
                self.pool.async_client.chat.completions.create(
                    model=self.args.model,
                    messages=msg,
                    **{k: v for k, v in self.args.model_dump().items() if k != "model"},
                )
                for msg in messages
            ]
            return await asyncio.gather(*tasks)

        return await self._arun_with_retry(_call)
