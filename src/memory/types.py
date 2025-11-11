from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------
# 1. 对话记忆 (Dialogue Memory)
# ----------------------------------------------------------------------

class Message(BaseModel):
    """单条对话消息"""
    sender: str = Field(..., description="发送者，如 'user' 或 'assistant'")
    content: str = Field(..., description="消息内容")
    timestamp: datetime = Field(default_factory=datetime.now, description="消息时间戳")

class DialogueMemory(BaseModel):
    """用户与智能体的对话历史"""
    user_id: str = Field(..., description="用户唯一ID")
    role_id: str = Field(..., description="角色唯一ID")
    messages: List[Message] = Field(default_factory=list, description="对话消息列表")
    last_updated: datetime = Field(default_factory=datetime.now, description="最后更新时间")
    
    def add_message(self, sender: str, content: str):
        """添加一条新消息"""
        self.messages.append(Message(sender=sender, content=content))
        self.last_updated = datetime.now()

# ----------------------------------------------------------------------
# 2. 激活记忆 (Active Memory)
# ----------------------------------------------------------------------

class ActiveMemoryItem(BaseModel):
    """激活记忆中的单个键值对"""
    key: str = Field(..., description="记忆键名，如 'user_preference_food'")
    value: Any = Field(..., description="记忆值")
    last_accessed: datetime = Field(default_factory=datetime.now, description="最后访问时间")
    
class ActiveMemory(BaseModel):
    """高频、近期的关键信息缓存 (KV 形式)"""
    user_id: str = Field(..., description="用户唯一ID")
    role_id: str = Field(..., description="角色唯一ID")
    items: Dict[str, ActiveMemoryItem] = Field(default_factory=dict, description="激活记忆键值对")
    
    def get(self, key: str) -> Optional[Any]:
        """获取激活记忆项的值"""
        item = self.items.get(key)
        if item:
            item.last_accessed = datetime.now()
            return item.value
        return None

    def set(self, key: str, value: Any):
        """设置或更新激活记忆项"""
        self.items[key] = ActiveMemoryItem(key=key, value=value)

# ----------------------------------------------------------------------
# 3. 专业记忆 (Professional Memory) - 仅定义接口，具体实现依赖外部 RAG/VectorDB
# ----------------------------------------------------------------------

class ProfessionalMemoryQuery(BaseModel):
    """专业记忆查询请求"""
    query: str = Field(..., description="用户查询内容")
    role_id: str = Field(..., description="角色ID")
    knowledge_path: str = Field(..., description="专业知识路径/索引名")
    top_k: int = Field(default=3, description="检索结果数量")

class ProfessionalMemoryResult(BaseModel):
    """专业记忆检索结果"""
    content: str = Field(..., description="检索到的知识片段内容")
    source: Optional[str] = Field(None, description="知识来源")
    score: Optional[float] = Field(None, description="相关性得分")

class ProfessionalMemory(BaseModel):
    """专业记忆的抽象表示"""
    results: List[ProfessionalMemoryResult] = Field(default_factory=list, description="检索到的专业知识列表")
    
    def to_prompt_context(self) -> str:
        """将检索结果格式化为可用于 Prompt 的上下文"""
        if not self.results:
            return "无相关专业知识。"
        
        context = "以下是与用户查询相关的专业知识片段，请参考并融合到你的回答中：\n"
        for i, result in enumerate(self.results):
            context += f"--- 知识片段 {i+1} ---\n"
            context += f"{result.content}\n"
            if result.source:
                context += f"来源: {result.source}\n"
        context += "------------------------\n"
        return context
