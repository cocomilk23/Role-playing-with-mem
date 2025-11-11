# Role-playing with Memory (Role-playing-with-mem)

本项目旨在构建一个功能与结构完整的**角色扮演智能助手框架**，其核心特点是实现了基于**角色隔离**的**记忆持久化机制**和**记忆融合逻辑**，确保智能体在跨次对话中保持身份认知、行为一致性和长期学习能力。

## 核心设计理念

框架的核心是围绕“角色”（Role）构建一个独立、隔离的“记忆空间”。每个智能体（Agent）都绑定一个角色，并拥有三层记忆系统，以实现高效、连贯且专业的交互。

## 架构概览

| 模块名称 | 核心类 | 职责描述 |
| :--- | :--- | :--- |
| **Agent** | `RolePlayingAgent` | 框架入口，协调记忆管理和LLM调用，执行记忆融合。 |
| **Role** | `Role` | 存储角色的静态配置、身份描述（System Prompt）和专业知识路径。 |
| **Memory** | `MemoryManager` | 统一管理三种记忆类型，负责记忆的存储、检索、更新和融合。 |
| **LLM** | `LLMConnector` | 抽象层，负责与底层大语言模型（LLM）的通信。 |

## 记忆系统详解

本项目实现了三种层次的记忆，以满足不同场景下的需求：

| 记忆类型 | 存储内容 | 存储机制 (当前实现) | 作用 |
| :--- | :--- | :--- | :--- |
| **专业记忆** (Professional Memory) | 角色专属的专业知识、理论依据。 | `MockRAG` 抽象层 (预留向量数据库接口) | 保证角色的专业性，作为回答的理论依据来源。 |
| **对话记忆** (Dialogue Memory) | 用户与智能体的完整对话历史、用户偏好、关键信息。 | `FilePersistenceLayer` (JSON 文件持久化) | 实现跨会话的连贯性，构建用户画像。 |
| **激活记忆** (Active Memory) | 高频、近期的关键信息（如最近查询、用户习惯的 KV 对）。 | `FilePersistenceLayer` (JSON 文件持久化) | 降低交互延迟，实现高速调用。 |

## 快速开始

### 1. 环境准备

本项目基于 Python 3.11+，并使用 `pydantic` 进行数据模型管理。

```bash
# 克隆仓库
git clone https://github.com/cocomilk23/Role-playing-with-mem.git
cd Role-playing-with-mem

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行示例

框架内置了一个示例，演示了记忆的加载、更新和融合过程。

```bash
python run.py
```

运行结果将展示智能体如何：
1.  加载角色配置（如“私人健康顾问”）。
2.  初始化并加载/创建对话记忆和激活记忆。
3.  在第一次交互中记录对话并更新记忆。
4.  在第二次交互中，通过 `MemoryManager` 融合**角色身份**、**激活记忆**（用户偏好）、**对话历史**和**专业知识**（RAG 模拟），生成一个包含所有上下文的 Prompt，并调用 `MockLLMConnector` 获得响应。
5.  将记忆持久化到 `data/memory_store` 目录下的 JSON 文件中。

### 3. 扩展与集成

#### LLM 集成

要集成真实的 LLM，您需要修改 `src/llm/connector.py` 文件，实现 `LLMConnector` 抽象类。例如，使用 OpenAI API：

```python
# src/llm/connector.py (部分)
from openai import OpenAI

class OpenAIConnector(LLMConnector):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        self.client = OpenAI(api_key=api_key)

    def generate_response(self, system_prompt: str, user_prompt: str, history: List[Dict[str, str]] = None) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        # ... 格式化历史记录和用户 Prompt ...
        messages.append({"role": "user", "content": user_prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        return response.choices[0].message.content
```

#### 专业记忆 (RAG) 集成

要实现真正的专业知识检索，您需要修改 `src/memory/persistence.py` 文件，实现 `ProfessionalMemoryRAG` 抽象类，并集成向量数据库（如 ChromaDB, Pinecone）或 RAG 框架（如 LangChain）。

```python
# src/memory/persistence.py (部分)
# ... 导入 ChromaDB 或其他 RAG 库 ...

class ChromaDBRAG(ProfessionalMemoryRAG):
    def retrieve(self, query: str, knowledge_path: str, top_k: int = 3) -> ProfessionalMemory:
        # 1. 连接到 ChromaDB 客户端
        # 2. 根据 knowledge_path (Collection Name) 检索
        # 3. 将检索结果转换为 ProfessionalMemory 实例
        pass
```

## 文件结构

```
Role-playing-with-mem/
├── src/
│   ├── agent.py          # RolePlayingAgent 核心逻辑
│   ├── role.py           # Role 类定义
│   ├── memory/
│   │   ├── manager.py    # MemoryManager 记忆管理核心
│   │   ├── persistence.py# PersistenceLayer 抽象和实现
│   │   └── types.py      # 记忆数据结构定义
│   └── llm/
│       └── connector.py  # LLMConnector 抽象和实现
├── config/
│   └── roles/
│       └── default_role.json # 角色配置示例
├── run.py                # 运行示例文件
├── ARCHITECTURE.md       # 架构设计文档
├── README.md             # 本文件
└── requirements.txt      # 项目依赖
```
