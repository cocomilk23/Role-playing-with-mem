from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import os
from memory.types import DialogueMemory, ActiveMemory

class PersistenceLayer(ABC):
    """
    记忆持久化抽象层。负责将 DialogueMemory 和 ActiveMemory 存储到持久化存储中。
    """
    
    @abstractmethod
    def load_dialogue_memory(self, user_id: str, role_id: str) -> Optional[DialogueMemory]:
        """加载指定用户和角色的对话记忆。"""
        pass

    @abstractmethod
    def save_dialogue_memory(self, memory: DialogueMemory):
        """保存对话记忆。"""
        pass

    @abstractmethod
    def load_active_memory(self, user_id: str, role_id: str) -> Optional[ActiveMemory]:
        """加载指定用户和角色的激活记忆。"""
        pass

    @abstractmethod
    def save_active_memory(self, memory: ActiveMemory):
        """保存激活记忆。"""
        pass

class FilePersistenceLayer(PersistenceLayer):
    """
    基于文件的简单持久化实现（用于快速原型和演示）。
    在实际生产环境中应替换为数据库实现。
    """
    def __init__(self, base_path: str = "data/memory_store"):
        import os
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def _get_path(self, user_id: str, role_id: str, memory_type: str) -> str:
        """获取记忆文件的路径。"""
        return os.path.join(self.base_path, f"{role_id}_{user_id}_{memory_type}.json")

    def load_dialogue_memory(self, user_id: str, role_id: str) -> Optional[DialogueMemory]:
        path = self._get_path(user_id, role_id, "dialogue")
        return self._load_memory(path, DialogueMemory, user_id, role_id)

    def save_dialogue_memory(self, memory: DialogueMemory):
        path = self._get_path(memory.user_id, memory.role_id, "dialogue")
        self._save_memory(path, memory)

    def load_active_memory(self, user_id: str, role_id: str) -> Optional[ActiveMemory]:
        path = self._get_path(user_id, role_id, "active")
        return self._load_memory(path, ActiveMemory, user_id, role_id)

    def save_active_memory(self, memory: ActiveMemory):
        path = self._get_path(memory.user_id, memory.role_id, "active")
        self._save_memory(path, memory)

    def _load_memory(self, path: str, model_class, user_id: str, role_id: str):
        import os
        import json
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return model_class.model_validate(data)
        
        # 如果文件不存在，返回一个新的空记忆实例
        if model_class == DialogueMemory:
            return DialogueMemory(user_id=user_id, role_id=role_id)
        elif model_class == ActiveMemory:
            return ActiveMemory(user_id=user_id, role_id=role_id)
        return None

    def _save_memory(self, path: str, memory):
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(memory.model_dump(mode='json'), f, ensure_ascii=False, indent=4)

# ----------------------------------------------------------------------
# 4. 专业记忆 RAG 抽象层 (Placeholder)
# ----------------------------------------------------------------------

class ProfessionalMemoryRAG(ABC):
    """
    专业记忆检索抽象层。负责 RAG 检索。
    """
    @abstractmethod
    def retrieve(self, query: str, knowledge_path: str, top_k: int = 3) -> 'ProfessionalMemory':
        """
        根据查询和知识路径进行 RAG 检索。
        返回 ProfessionalMemory 实例。
        """
        from memory.types import ProfessionalMemory
        return ProfessionalMemory()

class MockRAG(ProfessionalMemoryRAG):
    """
    模拟 RAG 检索实现。
    """
    def retrieve(self, query: str, knowledge_path: str, top_k: int = 3) -> 'ProfessionalMemory':
        from memory.types import ProfessionalMemory, ProfessionalMemoryResult
        print(f"Mock RAG: Retrieving for query '{query}' in knowledge '{knowledge_path}'")
        
        # 模拟检索结果
        if "健康" in query or "医疗" in query:
            results = [
                ProfessionalMemoryResult(
                    content="根据世界卫生组织定义，健康不仅仅是没有疾病或虚弱，而是身体、心理和社会适应的完好状态。",
                    source="WHO Constitution",
                    score=0.95
                ),
                ProfessionalMemoryResult(
                    content="对于高血压患者，建议低盐饮食，并保持适度的有氧运动，如快走或慢跑。",
                    source="Clinical Guidelines 2024",
                    score=0.88
                )
            ]
        else:
            results = []
            
        return ProfessionalMemory(results=results[:top_k])
