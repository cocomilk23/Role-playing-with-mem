from typing import Optional
from role import Role
from memory.manager import MemoryManager
from llm.connector import LLMConnector, MockLLMConnector, OpenAIConnector

class RolePlayingAgent:
    """
    角色扮演智能体核心类。
    负责接收用户输入，协调记忆管理和LLM连接器，生成响应。
    """
    def __init__(self, user_id: str, role: Role, llm_connector: Optional[LLMConnector] = None):
        """
        初始化智能体。
        
        :param user_id: 用户的唯一ID。
        :param role: 绑定的 Role 实例。
        :param llm_connector: LLMConnector 实例。
        """
        self.user_id = user_id
        self.role = role
        self.memory_manager = MemoryManager(user_id=user_id, role=role)
        self.llm_connector = llm_connector if llm_connector else MockLLMConnector()

    def process_query(self, user_query: str) -> str:
        """
        处理用户查询，生成响应。
        
        :param user_query: 用户的输入文本。
        :return: 智能体的响应文本。
        """
        # 1. 记录用户输入到对话记忆
        self.memory_manager.add_dialogue("user", user_query)
        
        # 2. 记忆融合：生成包含所有上下文的最终 Prompt
        fused_prompt = self.memory_manager.fuse_memory_for_prompt(user_query)
        
        # 3. 调用 LLM 生成响应
        # 注意：这里将 fused_prompt 作为 user_prompt 传递给 LLM，
        # 因为 fused_prompt 已经包含了 system_prompt 的内容（角色身份和指令）。
        # 实际生产中，可以根据 LLM API 的要求调整传递方式。
        response = self.llm_connector.generate_response(
            system_prompt=self.role.system_prompt, # 也可以将 system_prompt 单独传递
            user_prompt=fused_prompt
        )
        
        # 4. 记录智能体响应到对话记忆
        self.memory_manager.add_dialogue("assistant", response)
        
        # 5. TODO: 响应后处理，如关键信息提取并更新到 Active Memory
        
        return response

# ----------------------------------------------------------------------

