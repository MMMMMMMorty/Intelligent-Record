"""
Document Processor for Intelligent Document Generation
Handles text segmentation and LLM-based information extraction
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import httpx
import os


def clean_think_content(text: str) -> str:
    """
    清理 LLM 输出中的 <think>...</think> 思维链内容
    Qwen3 等模型会输出思考过程，需要过滤掉
    """
    if not text:
        return text
    # 移除 <think>...</think> 标签及其内容
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 清理多余的空白
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:8002")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3-1.7b")


@dataclass
class ExtractedInfo:
    """结构化提取信息"""
    person_name: Optional[str] = None
    person_id: Optional[str] = None
    person_address: Optional[str] = None
    event_time: Optional[str] = None
    event_location: Optional[str] = None
    event_description: List[str] = None
    motivation: Optional[str] = None
    legal_points: List[str] = None


class DocumentProcessor:
    """文档处理器：分段处理大文本并提取信息"""
    
    # 最大token数限制（预留空间给输出）
    MAX_TOKENS_PER_CHUNK = 1500
    # 估算：1个token ≈ 0.75个中文字
    CHINESE_CHARS_PER_TOKEN = 0.75
    
    def __init__(self, llm_api_url: str = LLM_API_URL, model_name: str = LLM_MODEL_NAME):
        self.llm_api_url = llm_api_url
        self.model_name = model_name
        
    def estimate_tokens(self, text: str) -> int:
        """估算文本的token数"""
        # 简单估算：中文字符数 / 0.75
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / self.CHINESE_CHARS_PER_TOKEN + other_chars / 2)
    
    def semantic_segmentation(self, text: str) -> List[str]:
        """
        语义分段：按对话轮次和主题切分
        策略：
        1. 首先按时间/主题关键词分段
        2. 然后按token数限制进一步切分
        """
        # 定义主题边界关键词
        topic_markers = [
            "身份确认", "姓名", "叫什么名字", "证件号",
            "事情经过", "发生了什么", "说说看", "讲一下",
            "动机", "为什么", "什么原因", "怎么想",
            "法律", "知道", "规定", "相关条款"
        ]
        
        # 首先按段落切分
        paragraphs = text.split('\n')
        
        # 合并段落形成语义片段
        segments = []
        current_segment = []
        current_tokens = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_tokens = self.estimate_tokens(para)
            
            # 如果当前段落包含主题标记，可能是一个新主题的开始
            is_new_topic = any(marker in para for marker in topic_markers)
            
            # 如果累积token数超过限制，或者遇到新主题且当前片段不为空
            if (current_tokens + para_tokens > self.MAX_TOKENS_PER_CHUNK or 
                (is_new_topic and current_segment and current_tokens > 500)):
                
                if current_segment:
                    segments.append('\n'.join(current_segment))
                    current_segment = []
                    current_tokens = 0
            
            current_segment.append(para)
            current_tokens += para_tokens
        
        # 添加最后一个片段
        if current_segment:
            segments.append('\n'.join(current_segment))
        
        return segments
    
    def chunk_by_sentences(self, text: str, max_tokens: int = 1200) -> List[str]:
        """
        按句子切分，确保每个片段不超过token限制
        """
        # 按句号、问号、感叹号切分
        sentences = re.split(r'([。！？\n])', text)
        sentences = [''.join(i) for i in zip(sentences[::2], sentences[1::2] + [''])]
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence)
            
            if current_tokens + sentence_tokens > max_tokens:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(''.join(current_chunk))
        
        return chunks
    
    async def call_llm(self, messages: List[Dict], max_tokens: int = 2048) -> Dict:
        """调用LLM API"""
        # 每次调用时重新读取环境变量，确保使用正确的模型名
        model_name = os.getenv("LLM_MODEL_NAME", self.model_name)
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.3,  # 降低温度以获得更稳定的输出
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.llm_api_url}/v1/chat/completions",
                    json=payload,
                    timeout=120.0
                )
                response.raise_for_status()
                result = response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    # 过滤 <think> 思维链内容
                    content = clean_think_content(content)
                    return {"success": True, "content": content}
                else:
                    return {"success": False, "error": "Invalid response format"}
                    
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def extract_information(self, text: str) -> ExtractedInfo:
        """
        从对话文本中提取结构化信息
        使用分段处理策略处理长文本
        """
        info = ExtractedInfo(
            event_description=[],
            legal_points=[]
        )
        
        # 第一步：语义分段
        semantic_chunks = self.semantic_segmentation(text)
        
        # 第二步：对每个语义片段提取信息
        for chunk in semantic_chunks:
            chunk_info = await self._extract_from_chunk(chunk)
            self._merge_info(info, chunk_info)
        
        return info
    
    async def _extract_from_chunk(self, chunk: str) -> Dict:
        """从单个文本片段中提取信息"""
        
        prompt = f"""请从以下对话记录中提取关键信息，以JSON格式返回。

对话内容：
{chunk}

请提取以下字段（如果没有则填null）：
- person_name: 当事人姓名
- person_id: 证件号码
- person_address: 住址
- event_time: 事情发生时间
- event_location: 事情发生地点
- event_facts: 事情经过的关键事实（数组）
- motivation: 动机或原因
- legal_issues: 涉及的法律问题（数组）

重要：直接返回JSON结果，不要输出思考过程，不要包含<think>标签。"""

        messages = [{"role": "user", "content": prompt}]
        result = await self.call_llm(messages, max_tokens=1024)
        
        if result["success"]:
            try:
                # 尝试解析JSON
                import json
                # 清理可能的markdown代码块标记
                content = result["content"].strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                extracted = json.loads(content)
                return extracted
            except:
                # JSON解析失败，返回空
                return {}
        
        return {}
    
    def _merge_info(self, target: ExtractedInfo, source: Dict):
        """合并提取的信息"""
        if source.get("person_name") and not target.person_name:
            target.person_name = source["person_name"]
        if source.get("person_id") and not target.person_id:
            target.person_id = source["person_id"]
        if source.get("person_address") and not target.person_address:
            target.person_address = source["person_address"]
        if source.get("event_time") and not target.event_time:
            target.event_time = source["event_time"]
        if source.get("event_location") and not target.event_location:
            target.event_location = source["event_location"]
        if source.get("motivation") and not target.motivation:
            target.motivation = source["motivation"]
        
        if source.get("event_facts"):
            if target.event_description is None:
                target.event_description = []
            target.event_description.extend(source["event_facts"])
        
        if source.get("legal_issues"):
            if target.legal_points is None:
                target.legal_points = []
            target.legal_points.extend(source["legal_issues"])
    
    async def generate_record(self, info: ExtractedInfo, original_text: str) -> str:
        """
        根据提取的信息生成标准文档记录
        """
        # 构建文档头
        header = self._generate_header(info)
        
        # 构建文档主体（一问一答形式）
        body = await self._generate_qa_body(info, original_text)
        
        return header + "\n\n" + body
    
    def _generate_header(self, info: ExtractedInfo) -> str:
        """生成文档头"""
        from datetime import datetime
        
        header = f"""文档记录

时间：{info.event_time or datetime.now().strftime('%Y年%m月%d日 %H时%M分')}
地点：{info.event_location or '___________'}
记录人：_____________  审核人：_____________

当事人：{info.person_name or '___________'}  性别：___  年龄：___
证件号码：{info.person_id or '____________________'}
联系地址：{info.person_address or '____________________'}
现住址：{info.person_address or '____________________'}

（以下为告知内容，略）

问：请说明本次记录的事由？
答："""
        
        return header
    
    async def _generate_qa_body(self, info: ExtractedInfo, original_text: str) -> str:
        """生成一问一答形式的文档主体"""
        
        facts = '\n'.join(info.event_description) if info.event_description else "（未提取到事实经过）"
        motivation = info.motivation or "（未提取到动机）"
        
        prompt = f"""请根据以下信息，生成标准的文档记录主体部分（一问一答形式）。

信息内容：
- 事情经过：{facts}
- 动机：{motivation}

要求：
1. 采用记录人问、当事人答的形式
2. 语言规范，符合文档格式
3. 完整还原事情经过
4. 适当追问细节

重要：直接生成文档内容，不要输出思考过程，不要包含<think>标签。"""

        messages = [{"role": "user", "content": prompt}]
        result = await self.call_llm(messages, max_tokens=4096)
        
        if result["success"]:
            return result["content"]
        else:
            return f"（文档生成失败：{result.get('error', '未知错误')}）"


# 全局处理器实例
document_processor = DocumentProcessor()
