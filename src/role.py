from typing import Dict, Any, Optional
import json

class Role:
    """
    角色配置类。存储角色的静态信息和专业知识路径。
    """
    def __init__(self, role_id: str, name: str, system_prompt: str, professional_knowledge_path: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        初始化 Role 实例。

        :param role_id: 角色唯一ID。
        :param name: 角色名称。
        :param system_prompt: 角色身份描述，用于LLM的System Prompt。
        :param professional_knowledge_path: 专业知识（如向量数据库索引）的路径或标识符。
        :param metadata: 其他元数据。
        """
        self.role_id = role_id
        self.name = name
        self.system_prompt = system_prompt
        self.professional_knowledge_path = professional_knowledge_path
        self.metadata = metadata if metadata is not None else {}

    @classmethod
    def from_config(cls, config_path: str) -> 'Role':
        """
        从 JSON 配置文件加载 Role 实例。
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return cls(
            role_id=config['role_id'],
            name=config['name'],
            system_prompt=config['system_prompt'],
            professional_knowledge_path=config.get('professional_knowledge_path'),
            metadata=config.get('metadata')
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        将 Role 实例转换为字典。
        """
        return {
            "role_id": self.role_id,
            "name": self.name,
            "system_prompt": self.system_prompt,
            "professional_knowledge_path": self.professional_knowledge_path,
            "metadata": self.metadata
        }

# ----------------------------------------------------------------------
# 辅助函数：创建默认角色配置文件
# ----------------------------------------------------------------------

def create_default_role_config(path: str):
    """
    创建默认的角色配置文件示例。
    """
    default_config = {
        "role_id": "default_medical_assistant",
        "name": "私人健康顾问",
        "system_prompt": "你是一位拥有十年经验的私人健康顾问。你的回答必须专业、严谨，同时充满人文关怀。你擅长根据用户的健康数据和生活习惯提供个性化的建议。请记住，你的核心职责是提供信息和建议，而不是进行诊断或开具处方。",
        "professional_knowledge_path": "medical_knowledge_index_v1",
        "metadata": {
            "version": "1.0",
            "creation_date": "2025-11-11"
        }
    }
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    # 示例：创建默认配置文件
    default_config_path = '/home/ubuntu/Role-playing-with-mem/config/roles/default_role.json'
    create_default_role_config(default_config_path)
    
    # 示例：加载角色
    role = Role.from_config(default_config_path)
    print(f"Loaded Role: {role.name} ({role.role_id})")
    print(f"System Prompt Snippet: {role.system_prompt[:50]}...")
