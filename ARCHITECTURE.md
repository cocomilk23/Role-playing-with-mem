# 角色扮演智能助手框架架构设计

## 1. 核心设计理念：以角色为中心的记忆架构

本框架旨在实现一个通用、可扩展的角色扮演智能助手，其核心是**“角色”**（Role）和**“记忆”**（Memory）。每个智能体（Agent）都将绑定一个角色，并拥有一个独立、隔离的记忆空间，以确保其专业性和行为一致性。

## 2. 模块划分与职责

框架将划分为以下核心模块：

| 模块名称 | 核心类/组件 | 职责描述 |
| :--- | :--- | :--- |
| **Agent** (智能体) | `RolePlayingAgent` | 框架的入口和核心执行者。负责接收用户输入，协调 `MemoryManager` 和 `LLMConnector`，并生成最终响应。 |
| **Role** (角色) | `Role` | 存储角色的静态配置和元数据，包括角色名称、身份描述（System Prompt）、专业领域等。 |
| **Memory** (记忆管理) | `MemoryManager` | 统一管理三种记忆类型，负责记忆的存储、检索、更新和融合。 |
| **LLM** (大模型连接) | `LLMConnector` | 负责与底层大语言模型（LLM）的通信，包括 Prompt 封装、API 调用和响应解析。 |
| **Persistence** (持久化) | `PersistenceLayer` | 负责将记忆数据持久化到数据库或文件系统，实现跨会话的长期存储。 |

## 3. 记忆架构详解

根据项目设想，记忆系统分为三个层次，由 `MemoryManager` 统一管理：

| 记忆类型 | 存储内容 | 存储机制/技术选型 | 作用 |
| :--- | :--- | :--- | :--- |
| **专业记忆** (Professional Memory) | 角色专属的专业知识、理论依据、上传资料。 | **向量数据库 (Vector DB)**：用于高效的 RAG 检索。 | 保证角色的专业性，作为回答的理论依据来源。 |
| **对话记忆** (Dialogue Memory) | 用户与智能体的完整对话历史、用户偏好、习惯、关键信息。 | **关系型/文档数据库 (SQL/NoSQL)**：按会话或用户ID存储。 | 实现跨会话的连贯性，构建用户画像。 |
| **激活记忆** (Active Memory) | 高频、近期的关键信息（如最近查询、用户习惯的 KV 对）。 | **内存缓存 (In-Memory Cache)**：如 Python 字典或 Redis。 | 降低交互延迟，实现高速调用。 |

## 4. 交互流程（Memory Fusion 记忆融合）

当用户输入一个查询时，`RolePlayingAgent` 的处理流程如下：

1.  **预处理**：`RolePlayingAgent` 接收用户输入。
2.  **激活记忆检索**：`MemoryManager` 检查 `Active Memory`，快速获取高频上下文。
3.  **对话记忆检索**：`MemoryManager` 检索 `Dialogue Memory`，获取最近的对话历史和用户画像。
4.  **专业记忆检索**：`MemoryManager` 根据用户查询，从 `Professional Memory` (Vector DB) 中检索相关专业知识片段（RAG）。
5.  **记忆融合**：`MemoryManager` 将 **角色身份** (`Role` System Prompt)、**激活记忆**、**对话历史** 和 **专业知识** 融合，生成一个完整的、包含所有上下文的 **最终 Prompt**。
6.  **LLM 调用**：`LLMConnector` 将最终 Prompt 发送给 LLM。
7.  **响应生成与后处理**：LLM 返回响应。`RolePlayingAgent` 将响应返回给用户，并异步更新 `Dialogue Memory` 和 `Active Memory`。

## 5. 文件结构

```
Role-playing-with-mem/
├── src/
│   ├── __init__.py
│   ├── agent.py          # RolePlayingAgent 核心逻辑
│   ├── role.py           # Role 类定义
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── manager.py    # MemoryManager 记忆管理核心
│   │   ├── persistence.py# PersistenceLayer 抽象和实现（如 File/DB）
│   │   └── types.py      # 记忆数据结构定义（专业、对话、激活）
│   └── llm/
│       ├── __init__.py
│       └── connector.py  # LLMConnector 抽象和实现（如 OpenAI/Gemini）
├── config/
│   └── roles/
│       └── default_role.json # 角色配置示例
├── tests/
├── README.md
└── requirements.txt
```
