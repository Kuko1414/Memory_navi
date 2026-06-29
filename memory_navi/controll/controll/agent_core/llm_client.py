"""vLLM (OpenAI 兼容) 客户端工厂。

proxy-safe 写法：trust_env=False + proxy=None，
避免本机代理环境变量干扰到 localhost 的 vLLM 调用。
"""
import httpx
from openai import OpenAI


def make_vllm_client(base_url: str) -> OpenAI:
    """连到 vLLM 的 OpenAI 客户端（api_key 占位，禁用代理）。"""
    return OpenAI(
        base_url=base_url,
        api_key="no-key",
        http_client=httpx.Client(proxy=None, trust_env=False),
    )


def resolve_model(client: OpenAI, fallback: str) -> str:
    """运行时从 /v1/models 取已加载模型 id，失败则回退。

    避免硬编码模型路径与 vLLM 实际 served-model-name 不一致。
    """
    try:
        models = client.models.list()
        if models.data:
            return models.data[0].id
    except Exception:
        pass
    return fallback
