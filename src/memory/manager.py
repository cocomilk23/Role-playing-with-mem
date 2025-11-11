from typing import Optional
from memory.types import DialogueMemory, ActiveMemory, ProfessionalMemory, ProfessionalMemoryQuery
from memory.persistence import PersistenceLayer, FilePersistenceLayer, ProfessionalMemoryRAG, MockRAG
from role import Role

class MemoryManager:
    """
    记忆管理核心类。负责加载、保存、检索和融合所有类型的记忆。
    """
    def __init__(self, user_id: str, role: Role, 
                 persistence_layer: Optional[PersistenceLayer] = None,
                 rag_system: Optional[ProfessionalMemoryRAG] = None):
        
        self.user_id = user_id
        self.role = role
        
        # 1. 持久化层：用于 Dialogue Memory 和 Active Memory 的长期存储
        self.persistence = persistence_layer if persistence_layer else FilePersistenceLayer(
            base_path=f"/home/ubuntu/Role-playing-with-mem/data/memory_store/{role.role_id}"
        )
        
        # 2. RAG 系统：用于 Professional Memory 的检索
        self.rag_system = rag_system if rag_system else MockRAG()
        
        # 3. 内存中的记忆实例
        self.dialogue_memory: DialogueMemory = self.persistence.load_dialogue_memory(user_id, role.role_id)
        self.active_memory: ActiveMemory = self.persistence.load_active_memory(user_id, role.role_id)

    def add_dialogue(self, sender: str, content: str):
        """
        添加一条对话记录到 Dialogue Memory。
        """
        self.dialogue_memory.add_message(sender, content)
        self.persistence.save_dialogue_memory(self.dialogue_memory)
        
        # TODO: 可以在这里实现一个机制，将关键信息提取并更新到 Active Memory

    def get_recent_dialogue(self, n: int = 5) -> str:
        """
        获取最近 n 条对话记录，格式化为 Prompt 字符串。
        """
        recent_messages = self.dialogue_memory.messages[-n:]
        
        formatted_dialogue = "--- 最近对话历史 ---\n"
        for msg in recent_messages:
            formatted_dialogue += f"[{msg.timestamp.strftime('%H:%M')}] {msg.sender.capitalize()}: {msg.content}\n"
        formatted_dialogue += "----------------------\n"
        
        return formatted_dialogue

    def get_active_memory_context(self) -> str:
        """
        获取 Active Memory 的上下文，格式化为 Prompt 字符串。
        """
        if not self.active_memory.items:
            return "无高频激活记忆。"
        
        context = "--- 高频激活记忆 (用户偏好/近期信息) ---\n"
        for key, item in self.active_memory.items.items():
            context += f"{key}: {item.value} (最后访问: {item.last_accessed.strftime('%Y-%m-%d %H:%M')})\n"
        context += "--------------------------------------\n"
        
        return context

    def retrieve_professional_memory(self, query: str) -> ProfessionalMemory:
        """
        检索 Professional Memory。
        """
        if not self.role.professional_knowledge_path:
            return ProfessionalMemory()
            
        return self.rag_system.retrieve(
            query=query,
            knowledge_path=self.role.professional_knowledge_path
        )

    def fuse_memory_for_prompt(self, user_query: str) -> str:
        """
        记忆融合：将所有记忆类型融合为一个完整的 Prompt 上下文。
        """
        # 1. 角色身份 (System Prompt)
        role_context = f"你的身份和核心指令：\n{self.role.system_prompt}\n\n"
        
        # 2. 激活记忆 (高频/近期)
        active_context = self.get_active_memory_context()
        
        # 3. 对话记忆 (历史)
        dialogue_context = self.get_recent_dialogue(n=5)
        
        # 4. 专业记忆 (RAG 检索)
        professional_memory = self.retrieve_professional_memory(user_query)
        professional_context = professional_memory.to_prompt_context()
        
        # 5. 最终融合 Prompt
        fused_prompt = (
            f"{role_context}"
            f"{active_context}"
            f"{dialogue_context}"
            f"{professional_context}"
            f"用户当前的问题是：{user_query}\n\n"
            "请根据上述所有信息，以你设定的角色身份，给出专业、个性化且连贯的回答。"
        )
        
        return fused_prompt
