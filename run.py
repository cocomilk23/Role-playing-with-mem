import sys
import os

# 将项目根目录添加到 Python 路径，以便正确解析相对导入
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from agent import run_example, Role, RolePlayingAgent
from memory.rag_utils import index_documents_to_chroma # 导入索引工具
from llm.connector import OpenAIConnector, MockLLMConnector # 导入 LLM 连接器

def run_openai_example():
    """
    使用 OpenAIConnector 运行示例。
    """
    # 1. 确保角色配置文件存在
    from role import create_default_role_config
    default_config_path = '/home/ubuntu/Role-playing-with-mem/config/roles/default_role.json'
    create_default_role_config(default_config_path)
    
    # 2. 加载角色
    role = Role.from_config(default_config_path)
    
    # 3. 初始化 LLM 连接器
    # 注意：在沙箱环境中，OPENAI_API_KEY 环境变量已配置。
    # 如果在本地运行，请确保设置了 OPENAI_API_KEY。
    try:
        llm_connector = OpenAIConnector(model_name="gpt-4o-mini", base_url="https://hk.uniapi.io")
    except Exception as e: # 捕获更广泛的异常，例如 API Key 缺失或连接错误
        print(f"警告: {e}。将回退到 MockLLMConnector。")
        llm_connector = MockLLMConnector()
    
    # 4. 初始化智能体
    user_id = "test_user_002_openai" # 使用新的用户ID以隔离记忆
    agent = RolePlayingAgent(user_id=user_id, role=role, llm_connector=llm_connector)
    
    print(f"--- 智能体初始化成功 (使用 {llm_connector.model_name}) ---")
    print(f"角色: {agent.role.name}")
    print(f"用户ID: {agent.user_id}")
    print("-" * 30)
    
    # 5. 第一次交互：设置激活记忆
    print("--- 第一次交互：设置用户偏好 ---")
    agent.memory_manager.active_memory.set("user_preference_food", "清淡少油")
    agent.memory_manager.active_memory.set("user_recent_trip", "下周去上海出差")
    agent.memory_manager.persistence.save_active_memory(agent.memory_manager.active_memory)
    
    query1 = "我最近总是感觉疲惫，有什么健康建议吗？"
    response1 = agent.process_query(query1)
    print(f"用户: {query1}")
    print(f"智能体: {response1}")
    print("-" * 30)
    
    # 6. 第二次交互：测试记忆融合（对话历史、激活记忆、专业记忆）
    print("--- 第二次交互：测试记忆融合 ---")
    query2 = "根据我的情况，我应该在上海出差期间注意什么饮食？"
    response2 = agent.process_query(query2)
    print(f"用户: {query2}")
    print(f"智能体: {response2}")
    print("-" * 30)
    
    # 7. 检查持久化文件
    print("--- 检查记忆持久化文件 ---")
    memory_path = f"/home/ubuntu/Role-playing-with-mem/data/memory_store/{role.role_id}"
    print(f"记忆存储路径: {memory_path}")
    print(f"文件列表: {os.listdir(memory_path)}")


if __name__ == '__main__':
    # 确保运行环境正确
    os.chdir('/home/ubuntu/Role-playing-with-mem')
    
    # 1. 确保 data 目录存在
    os.makedirs('data/memory_store/default_medical_assistant', exist_ok=True)
    
    # 2. 索引专业知识文档到 ChromaDB
    KNOWLEDGE_PATH = "medical_knowledge_index_v1"
    KNOWLEDGE_FILE = "/home/ubuntu/Role-playing-with-mem/data/medical_knowledge.txt"
    
    # 确保知识文件存在
    if not os.path.exists(KNOWLEDGE_FILE):
        print(f"错误：知识文件 {KNOWLEDGE_FILE} 不存在。请先创建该文件。")
    else:
        index_documents_to_chroma(
            file_path=KNOWLEDGE_FILE,
            collection_name=KNOWLEDGE_PATH, db_path="/home/ubuntu/Role-playing-with-mem/data/chroma_db"
        )
    
    # 3. 运行示例
    # run_example() # 原始 Mock 示例
    run_openai_example() # 新的 OpenAI 示例
