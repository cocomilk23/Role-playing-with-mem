import sys
import os

# 将项目根目录添加到 Python 路径，以便正确解析相对导入
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from agent import run_example, Role, RolePlayingAgent
from memory.rag_utils import index_documents_to_chroma # 导入索引工具

if __name__ == '__main__':
    # 1. 确保 data 目录存在
    os.makedirs('data/memory_store/default_medical_assistant', exist_ok=True)
    
    # 2. 索引专业知识文档到 ChromaDB
    # 角色配置中的 professional_knowledge_path 对应 ChromaDB 的 Collection Name
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
    run_example()
