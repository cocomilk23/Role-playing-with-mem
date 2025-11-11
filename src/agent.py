from typing import Optional
from role import Role
from memory.manager import MemoryManager
from llm.connector import LLMConnector, MockLLMConnector

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
# 示例运行
# ----------------------------------------------------------------------

def run_example():
    # 1. 确保角色配置文件存在
    from role import create_default_role_config
    default_config_path = '/home/ubuntu/Role-playing-with-mem/config/roles/default_role.json'
    create_default_role_config(default_config_path)
    
    # 2. 加载角色
    role = Role.from_config(default_config_path)
    
    # 3. 初始化智能体
    user_id = "test_user_001"
    agent = RolePlayingAgent(user_id=user_id, role=role)
    
    print(f"--- 智能体初始化成功 ---")
    print(f"角色: {agent.role.name}")
    print(f"用户ID: {agent.user_id}")
    print("-" * 30)
    
    # 4. 第一次交互：设置激活记忆
    print("--- 第一次交互：设置用户偏好 ---")
    agent.memory_manager.active_memory.set("user_preference_food", "清淡少油")
    agent.memory_manager.active_memory.set("user_recent_trip", "下周去上海出差")
    agent.memory_manager.persistence.save_active_memory(agent.memory_manager.active_memory)
    
    query1 = "我最近总是感觉疲惫，有什么健康建议吗？"
    response1 = agent.process_query(query1)
    print(f"用户: {query1}")
    print(f"智能体: {response1}")
    print("-" * 30)
    
    # 5. 第二次交互：测试记忆融合（对话历史、激活记忆、专业记忆）
    print("--- 第二次交互：测试记忆融合 ---")
    query2 = "根据我的情况，我应该在上海出差期间注意什么饮食？"
    response2 = agent.process_query(query2)
    print(f"用户: {query2}")
    print(f"智能体: {response2}")
    print("-" * 30)
    
    # 6. 检查持久化文件
    print("--- 检查记忆持久化文件 ---")
    import os
    memory_path = f"/home/ubuntu/Role-playing-with-mem/data/memory_store/{role.role_id}"
    print(f"记忆存储路径: {memory_path}")
    print(f"文件列表: {os.listdir(memory_path)}")

if __name__ == '__main__':
    # 确保运行环境正确
    import os
    os.chdir('/home/ubuntu/Role-playing-with-mem')
    
    # 运行示例
    run_example()
