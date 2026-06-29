"""图像工具：读盘 / 降采样 / 在 base64 与 OpenAI、Anthropic 图像格式间转换。

统一用 ImagePart(b64, mime) 流转。降采样在发给云端模型前做，省 token/带宽。
"""
import base64
import io
from dataclasses import dataclass

from PIL import Image as PILImage


@dataclass
class ImagePart:
    """一张图：base64 字符串 + MIME。"""
    b64: str
    mime: str = "image/jpeg"


def _pil_to_jpeg_b64(img: "PILImage.Image", quality: int) -> str:
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _downscale(img: "PILImage.Image", max_edge: int) -> "PILImage.Image":
    """长边 > max_edge 时等比缩小；否则原样返回。"""
    w, h = img.size
    if max(w, h) <= max_edge:
        return img
    img = img.copy()
    img.thumbnail((max_edge, max_edge), PILImage.Resampling.LANCZOS)
    return img


def downscale_b64(b64_in: str, max_edge: int, quality: int) -> ImagePart:
    """解码 base64 → 降采样 → 重新编码 JPEG。"""
    raw = base64.b64decode(b64_in)
    img = PILImage.open(io.BytesIO(raw))
    img = _downscale(img, max_edge)
    return ImagePart(b64=_pil_to_jpeg_b64(img, quality), mime="image/jpeg")


def read_file_as_imagepart(path: str, max_edge: int, quality: int) -> ImagePart:
    """直接读盘上的图（MCP server 存的 received_image.jpeg）→ 降采样 → ImagePart。"""
    img = PILImage.open(path)
    img = _downscale(img, max_edge)
    return ImagePart(b64=_pil_to_jpeg_b64(img, quality), mime="image/jpeg")


def to_openai_image_url(part: ImagePart) -> dict:
    """OpenAI / vLLM 多模态消息里的 image_url 块（data URL）。"""
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{part.mime};base64,{part.b64}"},
    }


def to_anthropic_image_block(part: ImagePart) -> dict:
    """Anthropic Messages API 的 image content block（base64 source）。"""
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": part.mime, "data": part.b64},
    }
