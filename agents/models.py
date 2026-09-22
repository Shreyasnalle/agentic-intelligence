import os
from typing import Optional, Any
from dotenv import load_dotenv

load_dotenv()


def get_hf_llm(
    repo_id: str = "meta-llama/Llama-3.3-70B-Instruct",
    api_token: Optional[str] = None,
    temperature: float = 0.7,
    max_new_tokens: int = 1024,
) -> Optional[Any]:
    """
    Instantiates an open-source Hugging Face model using ChatHuggingFace and HuggingFaceEndpoint.
    
    Reads HF_TOKEN or HUGGINGFACEHUB_API_TOKEN from environment if api_token is not provided.
    Returns None if no API token is configured.
    """
    token = api_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not token:
        return None

    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            huggingfacehub_api_token=token,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            task="text-generation",
        )
        chat_model = ChatHuggingFace(llm=llm)
        return chat_model
    except Exception as e:
        print(f"[models] Warning: Could not initialize Hugging Face model '{repo_id}': {e}")
        return None
