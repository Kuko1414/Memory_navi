"""C 分层混合 记录格式的 JSON Schema + 校验。

同一个 schema 同时用于：
  - Anthropic record_memory 工具的 input_schema（强制 Claude 产出合法结构）
  - 本地 jsonschema 校验（写入前把关）
设计要点：abs_pose / abs_pose_delta_m / view.distance_m / roi 等几何字段允许 null，
因为 slice 阶段不算绝对坐标（VLM 不臆造），由日后几何管线回填。
"""
import copy as _copy

try:
    import jsonschema
except ImportError:  # pragma: no cover - offline helpers can import without validation deps
    jsonschema = None

_POSE = {
    "type": ["object", "null"],
    "properties": {
        "x": {"type": "number"},
        "y": {"type": "number"},
        "z": {"type": "number"},
    },
    "additionalProperties": False,
}

_ROI = {
    "type": ["object", "null"],
    "description": "归一化 bbox：x,y 左上角，w,h 宽高，均 0~1。供代码在深度图上取该区域。",
    "properties": {
        "x": {"type": "number"}, "y": {"type": "number"},
        "w": {"type": "number"}, "h": {"type": "number"},
    },
    "additionalProperties": False,
}

_VIEW = {
    "type": ["object", "null"],
    "properties": {
        "angle_deg": {"type": ["number", "null"]},
        "distance_m": {"type": ["number", "null"]},        # 代码用深度回填，VLM 留 null
        "distance_delta_m": {"type": ["number", "null"]},  # ± 余量，代码回填
    },
    "additionalProperties": False,
}

_SIZE = {
    "type": ["object", "null"],
    "description": "3D 包围盒（米）：宽=x向, 高=z向, 深=y向。代码从 bbox 像素比例×depth 估算，VLM 留 null。",
    "properties": {
        "width_m": {"type": ["number", "null"]},
        "height_m": {"type": ["number", "null"]},
        "depth_m": {"type": ["number", "null"]},
    },
    "additionalProperties": False,
}

_OBJECT = {
    "type": "object",
    "required": ["name"],
    "properties": {
        "id": {"type": "string"},                           # 唯一标识，Qwen/Claude 输入
        "name": {"type": "string"},
        "old_name": {"type": ["string", "null"]},           # 整理层 B 纠错前的原名（溯源，便于 eval/nav 读旧标签）
        "aliases": {"type": "array", "items": {"type": "string"}},  # 别名，Claude 日后完善
        "spatial": {"type": ["string", "null"], "description": "定性/相对位置描述，如'桌面右侧'"},
        "roi": _ROI,
        "view": _VIEW,
        "abs_pose": _POSE,                                  # 几何回填（代码 back_project 算）
        "abs_pose_delta_m": {"type": ["number", "null"]},
        "bbox_center": {"type": ["array", "null"], "items": {"type": "number"}},  # 像素中心，几何管线回填
        "distance_m": {"type": ["number", "null"]},         # 深度距离，几何管线回填
        "size": _SIZE,                                      # 3D 尺寸，代码估算
        "size_unreliable": {"type": ["boolean", "null"]},   # 视角受限(高处/越界)→size/abs_pose 不可信，下游过滤
        "state": {"type": ["string", "null"]},
        "affordance": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "verified_by": {"type": "array", "items": {"type": "string"}},
        "last_seen": {"type": "string"},
        # 探索感知管线写入的溯源/诊断字段（代码填，非模型产）——列在此以便存盘校验通过。
        "roi_source": {"type": ["string", "null"]},           # roi 来源：qwen_depth_snap/yolo/...
        "semantic_source": {"type": ["string", "null"]},      # 命名来源：qwen_full/...
        "detector_label": {"type": ["string", "null"]},       # YOLO 检测原始标签
        "detector_source": {"type": ["string", "null"]},      # 检测器来源
        "geometry_status": {"type": ["string", "null"]},      # 几何接地状态：ok/...
        "depth_stats": {"type": ["object", "null"], "additionalProperties": True},   # ROI 深度统计
        "roi_quality": {"type": ["object", "null"], "additionalProperties": True},   # ROI 质量诊断
    },
    "additionalProperties": False,
}

_HAZARD = {
    "type": "object",
    "properties": {
        "type": {"type": "string"},
        "where": {"type": "string"},
        "note": {"type": "string"},
    },
    "additionalProperties": False,
}

_DOOR = {                                                   # 门/开口：探索写拓扑边前落在区域记录里的原料
    "type": "object",
    "properties": {
        "id": {"type": "string"},                          # 门稳定 id（门到门路由用，可选）
        "label": {"type": "string"},                       # 门中文名（Claude 可作 via 目标，可选）
        "pose": {
            "type": ["object", "null"],
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
            "additionalProperties": False,
        },
        "dir": {"type": "string"},                         # left/right/center
        "reason": {"type": "string"},
        "count": {"type": "number"},
    },
    "additionalProperties": False,
}

MEMORY_RECORD_SCHEMA = {
    "type": "object",
    # objects 期望非空，但不设为硬 required：部分模型/网关倾向只回 summary；
    # 切片以"打通 image→Claude→JSON→校验→写盘"为准，objects 充实度属 prompt/模型调优。
    "required": ["area", "type"],
    "properties": {
        "area": {"type": "string"},
        "type": {"type": "string", "description": "区域类型，如 office/corridor/kitchen"},
        "summary": {"type": "string"},
        "observed_at": {"type": "string"},
        "view_pose": {
            "type": ["object", "null"],
            "properties": {
                "x": {"type": "number"}, "y": {"type": "number"}, "yaw": {"type": "number"}
            },
            "additionalProperties": False,
        },
        "objects": {"type": "array", "items": _OBJECT},
        "hazards": {"type": "array", "items": _HAZARD},
        "doors": {"type": "array", "items": _DOOR},         # 门/开口（探索回填，供拓扑边）
        "boundary": {                                       # 房间粗边界（代码从 visited+wall_points 算）
            "type": ["object", "null"],
            "properties": {
                "xmin": {"type": "number"}, "xmax": {"type": "number"},
                "ymin": {"type": "number"}, "ymax": {"type": "number"},
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}


import copy as _copy

# 工具用的【严格】schema：把 objects 设为 required + minItems=1，
# 强制模型把物体填进数组（否则网关模型会只回 summary、留空 objects）。
# 存盘校验仍用宽松的 MEMORY_RECORD_SCHEMA（允许空 objects，不让校验失败）。
TOOL_RECORD_SCHEMA = _copy.deepcopy(MEMORY_RECORD_SCHEMA)
TOOL_RECORD_SCHEMA["required"] = ["area", "type", "objects"]
TOOL_RECORD_SCHEMA["properties"]["objects"]["minItems"] = 1


def validate_memory_record(obj: dict) -> dict:
    """校验记录；不合法抛 jsonschema.ValidationError（调用方做一次修复重试）。"""
    if jsonschema is None:
        raise RuntimeError("jsonschema package is not installed")
    jsonschema.validate(obj, MEMORY_RECORD_SCHEMA)
    return obj


# ---------------------------------------------------------------------------
# 整理层（Claude 整理官）—— 词典式 区域→子区→物体 结构。
# 语义归 Claude / 几何归代码：sub_area.range 由代码从成员 abs_pose 算，Claude 只给成员归属。
# ---------------------------------------------------------------------------

_SUB_AREA = {
    "type": "object",
    "required": ["id", "label", "member_ids"],
    "properties": {
        "id": {"type": "string"},                          # 代码分配，如 "sa_office_0"
        "label": {"type": "string"},                       # 中文标签，如 "办公区"
        "type": {"type": "string"},                        # office/kitchen/lounge/...
        "range": {                                         # 3D 包围盒，代码从成员 abs_pose min/max 算
            "type": ["object", "null"],
            "properties": {
                "xmin": {"type": "number"}, "xmax": {"type": "number"},
                "ymin": {"type": "number"}, "ymax": {"type": "number"},
                "zmin": {"type": "number"}, "zmax": {"type": "number"},
            },
            "additionalProperties": False,
        },
        "member_ids": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "additionalProperties": False,
}

_RELATION = {
    "type": "object",
    "required": ["subject_id", "predicate", "object_id"],
    "properties": {
        "subject_id": {"type": "string"},
        "predicate": {"type": "string", "enum": ["on", "in", "next_to", "under", "above"]},
        "object_id": {"type": "string"},
        "why": {"type": "string"},
    },
    "additionalProperties": False,
}

_CORRECTION = {                                             # 整理层审计日志（B 纠错 + merge/drop）
    "type": "object",
    "required": ["id", "op"],
    "properties": {
        "id": {"type": "string"},
        "op": {"type": "string", "enum": ["rename", "merge", "drop"]},
        "old_name": {"type": ["string", "null"]},
        "new_name": {"type": ["string", "null"]},
        "merged_ids": {"type": "array", "items": {"type": "string"}},
        "why": {"type": "string"},
    },
    "additionalProperties": False,
}

# 写盘校验用的【宽松】整理 schema：在区域记录基础上加整理层字段。
# 不动严格的 TOOL_RECORD_SCHEMA（强制 Qwen 填 objects）。
CURATED_RECORD_SCHEMA = _copy.deepcopy(MEMORY_RECORD_SCHEMA)
CURATED_RECORD_SCHEMA["properties"]["sub_areas"] = {"type": "array", "items": _SUB_AREA}
CURATED_RECORD_SCHEMA["properties"]["relations"] = {"type": "array", "items": _RELATION}
CURATED_RECORD_SCHEMA["properties"]["corrections"] = {"type": "array", "items": _CORRECTION}
CURATED_RECORD_SCHEMA["properties"]["curated_at"] = {"type": "string"}


def validate_curated_record(obj: dict) -> dict:
    """校验整理后记录；不合法抛 jsonschema.ValidationError。"""
    if jsonschema is None:
        raise RuntimeError("jsonschema package is not installed")
    jsonschema.validate(obj, CURATED_RECORD_SCHEMA)
    return obj
