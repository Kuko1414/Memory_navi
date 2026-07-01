"""记忆作者：抓图 → provider 标注 → 几何回填(depth_roi→size/abs_pose) → 校验 → 写文件系统记忆。

由 Supervisor 在触发点（entered_new_area / inspect）调用（下一步）；slice 里直接调。
几何回填（方法B）：模型只给 ROI，代码读一帧 depth 算 3D size + abs_pose（见 depth_projection）。
"""
import json
import os
from dataclasses import dataclass

import jsonschema

from .. import config
from ..geometry import depth_projection as dp
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
        if isinstance(view_pose, dict):
            # schema view_pose = {x,y,yaw}；位姿源常给 yaw_deg，这里归一化键名
            yaw = view_pose.get("yaw", view_pose.get("yaw_deg"))
            vp = {}
            if view_pose.get("x") is not None:
                vp["x"] = float(view_pose["x"])
            if view_pose.get("y") is not None:
                vp["y"] = float(view_pose["y"])
            if yaw is not None:
                vp["yaw"] = float(yaw)
            rec["view_pose"] = vp or None
        elif view_pose is not None:
            rec["view_pose"] = view_pose
        # 规范化：确保 objects/hazards 是 list，各 object 的 array 字段也被修正（本地 VLM 可能返 null/string）
        if not isinstance(rec.get("objects"), list):
            rec["objects"] = []
        if not isinstance(rec.get("hazards"), list):
            rec["hazards"] = []
        # hazards 元素规范化：模型常返字符串而非 {type,where,note} 对象 → 包成 {note:str}
        # （否则一条 hazard 格式问题会让整帧记录 schema 校验失败、丢掉标注）
        rec["hazards"] = [
            ({"note": h} if isinstance(h, str) else h)
            for h in rec["hazards"] if isinstance(h, (str, dict))
        ]
        _ARR = {"affordance", "verified_by", "aliases"}
        for obj in rec.get("objects", []) or []:
            if not isinstance(obj, dict):
                continue
            for k in list(obj.keys()):
                if k in _ARR:
                    if obj[k] is None:
                        obj[k] = []
                    elif isinstance(obj[k], str):
                        obj[k] = [obj[k]]
            # 不再强置 abs_pose=null：保留代码 back_project 已回填的值（VLM 自报的数字仍不可信，
            # 但 VLM 一般不会自填 abs_pose；调用方负责用几何管线填）。只规范明显非法的字符串。
            if isinstance(obj.get("abs_pose"), str):
                obj["abs_pose"] = None
            if isinstance(obj.get("size"), str):
                obj["size"] = None
        return rec

    def _backfill_geometry(self, record: dict, view_pose) -> None:
        """方法B 几何回填：对带 roi 的物体，读一帧 depth 算 size + abs_pose（原地写 record）。

        - size：roi_to_size（针孔×median 深度 + 近带厚度）。
        - abs_pose：roi 中心像素 + median 深度 → back_project 到 map（需 view_pose 才填）。
        一次 depth_roi 批量取所有 ROI 的深度统计（只读一帧，省带宽）。失败则静默跳过（保持 null）。
        """
        objs = record.get("objects") or []
        indexed = [(i, o) for i, o in enumerate(objs)
                   if isinstance(o, dict) and isinstance(o.get("roi"), dict)]
        if not indexed:
            return
        rois = [o["roi"] for _, o in indexed]
        try:
            out = self._ros.call("depth_roi", {"rois_json": json.dumps(rois, ensure_ascii=False)})
            data = json.loads((out.text or "").strip())
        except Exception:  # noqa: BLE001
            return
        if not data.get("ok"):
            return
        stats_list = data.get("stats") or []

        pose = None
        if isinstance(view_pose, dict):
            yaw = view_pose.get("yaw_deg", view_pose.get("yaw"))
            x, y = view_pose.get("x"), view_pose.get("y")
            if x is not None and y is not None and yaw is not None:
                pose = {"x": float(x), "y": float(y), "yaw_deg": float(yaw)}

        for (_, o), stats in zip(indexed, stats_list):
            if not stats:
                continue
            o["size"] = dp.roi_to_size(o["roi"], stats)
            if pose and stats.get("median_m"):
                try:
                    uc, vc = dp.roi_center_pixel(o["roi"])
                    o["abs_pose"] = dp.back_project(uc, vc, stats["median_m"], robot_pose=pose)["abs_pose"]
                except Exception:  # noqa: BLE001
                    pass

    def record(self, area: str, trigger: str = "manual", view_pose=None,
               write: bool = True, known_objects=None) -> AuthorResult:
        try:
            image = self._grab_image()
        except Exception as e:  # noqa: BLE001
            return AuthorResult(ok=False, error=f"抓图失败：{e}")

        ctx = {
            "area_hint": area,
            "trigger": trigger,
            "known_areas": self._memory.list_areas() if self._memory is not None else [],
            "view_pose": view_pose,
            "known_objects": known_objects or [],   # 已记录物体名（让记忆作者只补缺口、不重复登记）
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

        # 几何回填（方法B）：模型只给 ROI，代码读 depth 帧算 size + abs_pose
        self._backfill_geometry(record, view_pose)

        # write=False：只回 record（调用方自行逐物体 upsert_object 并集落盘，如 explore_probe），
        # 避免每帧 upsert_area 整条覆盖 area.json。默认 True 保持 slice_demo/geom_backfill 行为不变。
        if write and self._memory is not None:
            self._memory.upsert_area(area, record)
        return AuthorResult(ok=True, record=record, raw=raw)
