from abc import ABC, abstractmethod
from typing import List, Dict, Any
import os
from openai import OpenAI

class LLMConnector(ABC):
    """
    大语言模型连接器抽象层。
    """
    def __init__(self, model_name: str, api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key

    @abstractmethod
    def generate_response(self, system_prompt: str, user_prompt: str, history: List[Dict[str, str]] = None) -> str:
        """
        根据 Prompt 和历史记录生成响应。
        
        :param system_prompt: 角色身份和指令。
        :param user_prompt: 用户的当前查询（包含所有记忆融合的上下文）。
        :param history: 格式化的对话历史（可选，用于多轮对话）。
        :return: LLM 生成的文本响应。
        """
        pass

class OpenAIConnector(LLMConnector):
    """
    基于 OpenAI API 的 LLM 连接器实现。
    """
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = None):
        # 优先使用传入的 api_key，否则尝试从环境变量获取
        key = api_key if api_key else os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API Key not found. Please provide it or set the OPENAI_API_KEY environment variable.")
            
        super().__init__(model_name, key)
        
        # 注意：这里使用 client = OpenAI() 即可，因为沙箱环境已配置好 API Key 和 Base URL
        self.client = OpenAI()

    def generate_response(self, system_prompt: str, user_prompt: str, history: List[Dict[str, str]] = None) -> str:
        
        # 构造消息列表
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 实际生产中，可以根据需要将 history 插入到 messages 中
        # 但由于我们的 MemoryManager 已经将历史和 RAG 结果融合到了 user_prompt 中，
        # 这里的 history 参数暂时可以忽略，专注于处理融合后的 user_prompt。

        try:
            print(f"--- OpenAI LLM Call (Model: {self.model_name}) ---")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"OpenAI API Call Error: {e}")
            return f"抱歉，LLM 服务调用失败。错误信息: {e}"


class MockLLMConnector(LLMConnector):
    """
    模拟 LLM 连接器，用于测试和演示。
    """
    def __init__(self, model_name: str = "mock-gpt-4"):
        super().__init__(model_name)

    def generate_response(self, system_prompt: str, user_prompt: str, history: List[Dict[str, str]] = None) -> str:
        print(f"--- Mock LLM Call ---")
        print(f"Model: {self.model_name}")
        print(f"System Prompt Snippet: {system_prompt[:100]}...")
        print(f"User Prompt: {user_prompt}")
        print(f"---------------------")
        
        # 模拟根据 Prompt 内容生成响应
        if "健康" in user_prompt or "医疗" in user_prompt:
            return "作为您的私人健康顾问，我已参考您的历史偏好和专业知识。根据您的问题，我建议您保持积极乐观的心态，并注意均衡饮食。请问您想了解更多关于哪方面的健康建议？"
        elif "偏好" in user_prompt:
            return "我记得您上次提到您喜欢清淡的食物。在为您提供建议时，我会充分考虑您的这一偏好。"
        else:
            return f"您好，我是{self.model_name}模拟的角色扮演智能体。我已接收到您的请求：'{user_prompt}'。我正在努力融合我的专业记忆、对话记忆和激活记忆来为您提供最个性化的回答。"
