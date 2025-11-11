#!/usr/bin/env python3
"""
简洁易懂的角色扮演智能体示例
功能：专业记忆检索 + 对话记忆 + LLM生成回复
作者：基于Role-playing-with-mem项目
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent import RolePlayingAgent
from role import Role, create_default_role_config
from llm.connector import OpenAIConnector, MockLLMConnector
from memory.rag_utils import index_documents_to_chroma

class SimpleHealthAssistant:
    """简化的健康助手类"""
    
    def __init__(self, use_openai=True):
        """
        初始化健康助手
        
        Args:
            use_openai: 是否使用OpenAI API，False则使用模拟器
        """
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.setup_environment()
        self.agent = self.create_agent(use_openai)
        
    def setup_environment(self):
        """设置运行环境"""
        # 切换到项目目录
        os.chdir(self.project_root)
        
        # 创建必要目录
        os.makedirs('data/memory_store', exist_ok=True)
        os.makedirs('data/chroma_db', exist_ok=True)
        os.makedirs('config/roles', exist_ok=True)
        
        # 索引专业知识到向量数据库
        self.index_knowledge()
        
    def index_knowledge(self):
        """索引医疗知识到ChromaDB"""
        knowledge_file = os.path.join(self.project_root, 'data/medical_knowledge.txt')
        
        if os.path.exists(knowledge_file):
            print("📚 正在索引医疗知识库...")
            index_documents_to_chroma(
                file_path=knowledge_file,
                collection_name="medical_knowledge_index_v1",
                db_path=os.path.join(self.project_root, 'data/chroma_db')
            )
            print("✅ 知识库索引完成")
        else:
            print(f"⚠️  知识文件不存在: {knowledge_file}")
    
    def create_agent(self, use_openai=True):
        """创建智能体"""
        # 1. 创建角色配置
        config_path = os.path.join(self.project_root, 'config/roles/health_assistant.json')
        create_default_role_config(config_path)
        
        # 2. 加载角色
        role = Role.from_config(config_path)
        
        # 3. 选择LLM连接器
        if use_openai and os.getenv('OPENAI_API_KEY'):
            try:
                llm_connector = OpenAIConnector(
                    model_name="gpt-4o-mini",
                    base_url="https://hk.uniapi.io/v1"
                )
                print("🤖 使用 OpenAI GPT-4o-mini")
            except Exception as e:
                print(f"⚠️  OpenAI连接失败: {e}")
                llm_connector = MockLLMConnector()
                print("🤖 使用模拟连接器")
        else:
            llm_connector = MockLLMConnector()
            print("🤖 使用模拟连接器")
        
        # 4. 创建智能体
        agent = RolePlayingAgent(
            user_id="demo_user",
            role=role,
            llm_connector=llm_connector
        )
        
        print(f"✅ 智能体初始化完成")
        print(f"   角色: {role.name}")
        print(f"   模型: {llm_connector.model_name}")
        
        return agent
    
    def chat(self, message: str) -> str:
        """
        与智能体对话
        
        Args:
            message: 用户消息
            
        Returns:
            智能体回复
        """
        return self.agent.process_query(message)
    
    def show_memory_status(self):
        """显示记忆状态"""
        print("\n📋 记忆状态:")
        
        # 对话记忆
        dialogue_count = len(self.agent.memory_manager.dialogue_memory.messages)
        print(f"   对话记录: {dialogue_count} 条")
        
        # 专业记忆
        if self.agent.role.professional_knowledge_path:
            print(f"   专业知识库: {self.agent.role.professional_knowledge_path}")
        
        # 记忆文件
        memory_path = f"data/memory_store/{self.agent.role.role_id}"
        if os.path.exists(memory_path):
            files = os.listdir(memory_path)
            print(f"   持久化文件: {len(files)} 个")


def start_chat():
    """直接启动聊天模式"""
    print("🏥 健康助手 - 智能对话")
    print("=" * 40)
    
    # 检查API Key
    use_openai = bool(os.getenv('OPENAI_API_KEY'))
    if not use_openai:
        print("💡 提示: 设置 OPENAI_API_KEY 环境变量可使用真实AI模型")
    
    # 初始化助手
    assistant = SimpleHealthAssistant(use_openai=use_openai)
    
    # 直接进入交互模式
    print("\n💬 开始对话 (输入 'quit' 退出)")
    print("=" * 40)
    
    while True:
        try:
            user_input = input("\n👤 您: ").strip()
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                break
            if not user_input:
                continue
                
            response = assistant.chat(user_input)
            print(f"🤖 助手: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ 错误: {e}")
    
    print("\n👋 感谢使用健康助手！")


if __name__ == '__main__':
    try:
        start_chat()
    except KeyboardInterrupt:
        print("\n\n👋 程序已退出")
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()