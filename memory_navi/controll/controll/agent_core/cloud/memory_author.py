"""记忆作者：抓图 → provider 标注 → 校验 → 写文件系统记忆。

由 Supervisor 在触发点（entered_new_area / inspect）调用（下一步）；slice 里直接调。
slice 阶段：强制 abs_pose=null（几何管线未就绪），但保留 VLM 给的 roi 以便日后回填。
"""
import os
from dataclasses import dataclass

import jsonschema

from .. import config
from ..image_utils import downscale_b64, read_file_as_imagepart
from .schema import validate_memory_record


@dataclass
class AuthorResult:
    ok: bool
    record: dict = None
    raw: dict = None
    error: str = None


class MemoryAuthor:
    def __init__(
        self,
        ros_tools,
        provider,
        memory,
        agent_ns: str = config.AGENT_NS,
        camera_topic: str = config.CAMERA_COMPRESSED_TOPIC,
        camera_msg_type: str = config.CAMERA_COMPRESSED_MSG_TYPE,
        image_path: str = config.MCP_IMAGE_PATH,
        max_edge: int = config.IMG_MAX_EDGE,
        quality: int = config.IMG_QUALITY,
    ):
        self._ros = ros_tools
        self._provider = provider
        self._memory = memory
        self._topic = camera_topic
        self._msg_type = camera_msg_type
        self._image_path = image_path
        self._max_edge = max_edge
        self._quality = quality

    def _grab_image(self):
        """触发一次 subscribe_once 刷新盘上帧；优先读盘降采样，否则用返回的 base64。"""
        out = self._ros.call(
            "subscribe_once",
            {
                "topic": self._topic,
                "msg_type": self._msg_type,
                "expects_image": "true",
                "timeout": 5.0,
            },
        )
        if os.path.exists(self._image_path):
            return read_file_as_imagepart(self._image_path, self._max_edge, self._quality)
        if out.images:
            return downscale_b64(out.images[0].b64, self._max_edge, self._quality)
        raise RuntimeError(f"未取到相机图像（topic={self._topic}）：{out.text[:200]}")

    def _finalize(self, raw: dict, area: str, view_pose) -> dict:
        rec = dict(raw)
        rec["area"] = area or rec.get("area", "unknown")
        rec.setdefault("type", "unknown")
        if view_pose is not None:
            rec["view_pose"] = view_pose
        # 规范化：确保 objects/hazards 是 list，各 object 的 array 字段也被修正（本地 VLM 可能返 null/string）
        if not isinstance(rec.get("objects"), list):
            rec["objects"] = []
        if not isinstance(rec.get("hazards"), list):
            rec["hazards"] = []
        _ARR = {"affordance", "verified_by"}
        for obj in rec.get("objects", []) or []:
            if not isinstance(obj, dict):
                continue
            for k in list(obj.keys()):
                if k in _ARR:
                    if obj[k] is None:
                        obj[k] = []
                    elif isinstance(obj[k], str):
                        obj[k] = [obj[k]]
            obj["abs_pose"] = None
            obj["abs_pose_delta_m"] = None
        return rec

    def record(self, area: str, trigger: str = "manual", view_pose=None) -> AuthorResult:
        try:
            image = self._grab_image()
        except Exception as e:  # noqa: BLE001
            return AuthorResult(ok=False, error=f"抓图失败：{e}")

        ctx = {
            "area_hint": area,
            "trigger": trigger,
            "known_areas": self._memory.list_areas(),
            "view_pose": view_pose,
        }
        raw = None
        try:
            raw = self._provider.annotate(image, ctx)
            record = self._finalize(raw, area, view_pose)
            try:
                validate_memory_record(record)
            except jsonschema.ValidationError as e:
                # 修复重试一次
                ctx["schema_error"] = e.message
                raw = self._provider.annotate(image, ctx)
                record = self._finalize(raw, area, view_pose)
                validate_memory_record(record)
        except Exception as e:  # noqa: BLE001
            return AuthorResult(ok=False, raw=raw, error=f"标注/校验失败：{e}")

        self._memory.upsert_area(area, record)
        return AuthorResult(ok=True, record=record, raw=raw)
