# api_client.py
import os
from volcenginesdkarkruntime import Ark
from dotenv import load_dotenv

load_dotenv()

class DoubaoAPIClient:
    def __init__(self):
        self.api_key = os.getenv("ARK_API_KEY")
        self.base_url = "https://ark.cn-beijing.volces.com/api/v3"

        self.client = Ark(
            base_url=self.base_url,
            api_key=self.api_key
        )

        self.model = "doubao-seed-2-0-pro-260215"
        self.embedding_model = "doubao-embedding-vision-251215"

    def call_llm(self, prompt, model=None, temperature=0.3):
        """调用豆包大语言模型"""
        response = self.client.responses.create(
            model=model or self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt}
                    ]
                }
            ],
            temperature=temperature
        )
        if hasattr(response, 'output_text'):
            return {"content": response.output_text}
        elif hasattr(response, 'choices') and len(response.choices) > 0:
            return {"content": response.choices[0].message.content}
        else:
            return {"content": str(response)}

    def call_llm_with_image(self, prompt, image_url, model=None, temperature=0.3):
        """调用豆包大语言模型（支持图片输入）"""
        response = self.client.responses.create(
            model=model or self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": image_url},
                        {"type": "input_text", "text": prompt}
                    ]
                }
            ],
            temperature=temperature
        )
        if hasattr(response, 'output_text'):
            return {"content": response.output_text}
        elif hasattr(response, 'choices') and len(response.choices) > 0:
            return {"content": response.choices[0].message.content}
        else:
            return {"content": str(response)}

    def get_embedding(self, text, model=None):
        """获取文本Embedding向量"""
        response = self.client.multimodal_embeddings.create(
            model=model or self.embedding_model,
            input=[
                {"type": "text", "text": text}
            ]
        )
        return {"response": response}

    def get_text_embeddings(self, texts, model=None):
        """批量获取文本Embedding向量"""
        inputs = [{"type": "text", "text": t} for t in texts]
        response = self.client.multimodal_embeddings.create(
            model=model or self.embedding_model,
            input=inputs
        )
        return {"response": response}
