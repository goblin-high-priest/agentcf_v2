
## 实施阶段

### ✅ 阶段0: 基础结构 (已完成)
- [x] 创建所有目录
- [x] 创建所有 `__init__.py` 文件

### 🔄 阶段1: 核心基础模块

#### 1.1 注册表和基础类
- [ ] `agentverse/registry.py` - 注册表系统
  - 复制原文件结构
  - 保持 Registry 类不变

- [ ] `agentverse/message.py` - 消息类
  - 复制原文件结构
  - Message 类定义

- [ ] `agentverse/parser.py` - 解析器基类
  - 复制原文件结构
  - OutputParser 基类和 OutputParserError

- [ ] `agentverse/utils.py` - 工具函数
  - AgentAction NamedTuple
  - AgentFinish NamedTuple

- [ ] `agentverse/memory/base.py` - 内存基类
  - BaseMemory 抽象类

- [ ] `agentverse/agents/base.py` - Agent基类
  - BaseAgent 抽象类
  - 保持所有方法签名

- [ ] `agentverse/llms/base.py` - LLM基类
  - BaseLLM, BaseChatModel, BaseCompletionModel
  - LLMResult 类

### 🔄 阶段2: LLM模块 (核心重构)

#### 2.1 OpenAI v1+ 实现
- [ ] `agentverse/llms/openai.py` - OpenAI客户端
  - **关键修改点：**
    - 使用 `from openai import OpenAI, AsyncOpenAI`
    - 移除全局 `openai.api_key` 配置
    - 在类中创建 client 实例
    - 更新所有 API 调用：
      - `openai.Completion.create()` → `client.completions.create()`
      - `openai.ChatCompletion.create()` → `client.chat.completions.create()`
      - `openai.Embedding.create()` → `client.embeddings.create()`
    - 更新响应访问：
      - `response["choices"]` → `response.choices`
      - `response["data"]` → `response.data`
    - 更新错误处理：
      - `OpenAIError` → `APIError, APIConnectionError, RateLimitError`
    - 支持代理配置（通过 httpx）

#### 2.2 LLM测试
- [ ] `tests/test_llm_openai.py` - LLM模块测试
  - 测试同步和异步调用
  - 测试错误处理

### 🔄 阶段3: Agent系统

#### 3.1 输出解析器
- [ ] `agentverse/tasks/recommendation/output_parser.py`
  - RecommenderParser
  - UserAgentParser
  - ItemAgentParser
  - 注意：parse 方法接受字符串，不是 LLMResult

#### 3.2 Agent实现
- [ ] `agentverse/agents/conversation_agent.py`
  - EmbeddingAgent
  - RecAgent
  - UserAgent
  - ItemAgent
  - **更新点：** 确保所有 LLM 调用使用新的 client 方式

#### 3.3 初始化模块
- [ ] `agentverse/initialization.py`
  - load_llm 函数
  - load_agent 函数
  - prepare_task_config 函数（可选，推荐系统可能不需要）

#### 3.4 Agent测试
- [ ] `tests/test_agents.py` - Agent系统测试

### 🔄 阶段4: 数据处理和工具

#### 4.1 Embedding工具
- [ ] 实现自定义 `embedding_utils` 函数
  - `distances_from_embeddings()` - 计算余弦距离
  - `indices_of_nearest_neighbors_from_distances()` - 找最近邻
  - 可以放在 `utils.py` 或单独的 `embedding_utils.py`

#### 4.2 数据集模块
- [ ] `dataset.py`
  - BPRDataset 类
  - ITEMBPRDataset 类（如果需要）
  - 复制原文件结构

#### 4.3 工具函数
- [ ] `utils.py`
  - `get_model()` 函数
  - `dispatch_openai_requests()` - 更新使用 AsyncOpenAI
  - `dispatch_single_openai_requests()` - 更新使用 OpenAI
  - 其他工具函数

#### 4.4 数据处理测试
- [ ] `tests/test_embedding_utils.py` - Embedding工具测试

### 🔄 阶段5: 核心模型

#### 5.1 AgentCF模型
- [ ] `model/agentcf.py`
  - **关键修改点：**
    - 替换 `from openai.embeddings_utils import ...` 
    - 改为使用自定义的 embedding_utils
    - 更新 `generate_embedding()` 中的响应访问
    - 确保所有 LLM 调用使用新方式
    - 保持所有其他逻辑不变

#### 5.2 模型测试
- [ ] `tests/test_model_init.py` - 模型初始化测试

### 🔄 阶段6: 训练和运行

#### 6.1 训练器
- [ ] `trainer.py`
  - LanguageLossTrainer
  - 复制原文件结构

#### 6.2 主入口
- [ ] `run.py`
  - run_baseline 函数
  - 主程序入口
  - 复制原文件结构

#### 6.3 配置文件
- [ ] 复制 `props/` 目录
  - `overall.yaml`
  - `AgentCF.yaml`
  - `CDs.yaml`
  - `CDs-100-user-dense.yaml`
  - `CDs-100-user-sparse.yaml`
  - 其他需要的配置文件

#### 6.4 端到端测试
- [ ] `tests/test_end_to_end.py` - 完整流程测试

### 🔄 阶段7: 集成和验证

#### 7.1 依赖管理
- [ ] `requirements.txt`
  - openai>=1.0.0
  - httpx>=0.24.0
  - 其他依赖保持与原项目一致

#### 7.2 文档
- [ ] `README.md` - 更新说明API变更
- [ ] `CHANGELOG.md` - 记录所有变更

#### 7.3 对比测试
- [ ] 在同一数据集上运行原版和新版
- [ ] 对比输出结果
- [ ] 验证功能一致性

## 关键修改点总结

### 1. OpenAI API 迁移
**文件：** `agentverse/llms/openai.py`
- 完全重写使用 OpenAI v1+ SDK
- 使用 Client 实例而非全局配置
- 更新所有 API 调用和响应访问

### 2. Embedding工具替换
**文件：** `utils.py` 或新建 `embedding_utils.py`
- 实现自定义的 `distances_from_embeddings`
- 实现自定义的 `indices_of_nearest_neighbors_from_distances`

### 3. 模型中的更新
**文件：** `model/agentcf.py`
- 替换 `openai.embeddings_utils` 导入
- 更新 embedding 响应访问方式

### 4. 工具函数更新
**文件：** `utils.py`
- 更新 `dispatch_openai_requests` 使用 AsyncOpenAI
- 更新 `dispatch_single_openai_requests` 使用 OpenAI

## 测试策略

每个阶段完成后：
1. 运行单元测试验证模块功能
2. 对比原版输出（如果可能）
3. 集成测试已完成的模块

## 依赖版本

- openai >= 1.0.0
- httpx >= 0.24.0 (OpenAI v1+ 需要)
- 其他依赖保持与原项目一致

## 注意事项

1. **API Key 管理：** 新版本中每个 client 实例都需要单独配置 API key
2. **响应格式：** 新版本响应是 Pydantic 模型对象，可直接访问属性
3. **异步客户端：** 必须为每个异步方法创建 AsyncOpenAI 实例
4. **代理设置：** 通过 httpx 的 http_client 参数设置
5. **向后兼容：** 尽量保持接口一致，确保现有代码可以无缝迁移

## 进度跟踪

- [x] 阶段0: 基础结构
- [ ] 阶段1: 核心基础模块
- [ ] 阶段2: LLM模块
- [ ] 阶段3: Agent系统
- [ ] 阶段4: 数据处理和工具
- [ ] 阶段5: 核心模型
- [ ] 阶段6: 训练和运行
- [ ] 阶段7: 集成和验证

---

**最后更新：** 2025-01-XX
**状态：** 进行中