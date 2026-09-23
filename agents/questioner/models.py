import os
from typing import Optional, Any
from dotenv import load_dotenv

load_dotenv()

LLAMA_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEEPSEEK_REASONING_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-8B"

def get_hf_llm(
    repo_id: Optional[str] = None,
    api_token: Optional[str] = None,
    temperature: float = 0.6,
    max_new_tokens: int = 512,
) -> Optional[Any]:
    token = api_token or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    model_id = repo_id or os.getenv("HF_MODEL_NAME") or LLAMA_MODEL_ID

    if not token:
        return None

    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        llm = HuggingFaceEndpoint(
            repo_id=model_id,
            huggingfacehub_api_token=token,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            task="text-generation",
        )
        return ChatHuggingFace(llm=llm)
    except Exception:
        return None