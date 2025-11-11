import sys
import os

# 将项目根目录添加到 Python 路径，以便正确解析相对导入
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from agent import run_example, Role, RolePlayingAgent

if __name__ == '__main__':
    # 确保 data 目录存在
    os.makedirs('data/memory_store/default_medical_assistant', exist_ok=True)
    
    # 运行示例
    run_example()
