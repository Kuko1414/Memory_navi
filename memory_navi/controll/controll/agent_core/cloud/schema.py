"""C 分层混合 记录格式的 JSON Schema + 校验。

同一个 schema 同时用于：
  - Anthropic record_memory 工具的 input_schema（强制 Claude 产出合法结构）
  - 本地 jsonschema 校验（写入前把关）
设计要点：abs_pose / abs_pose_delta_m / view.distance_m / roi 等几何字段允许 null，
因为 slice 阶段不算绝对坐标（VLM 不臆造），由日后几何管线回填。
"""
import jsonschema

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
        "aliases": {"type": "array", "items": {"type": "string"}},  # 别名，Claude 日后完善
        "spatial": {"type": "string", "description": "定性/相对位置描述，如'桌面右侧'"},
        "roi": _ROI,
        "view": _VIEW,
        "abs_pose": _POSE,                                  # 几何回填（代码 back_project 算）
        "abs_pose_delta_m": {"type": ["number", "null"]},
        "size": _SIZE,                                      # 3D 尺寸，代码估算
        "size_unreliable": {"type": ["boolean", "null"]},   # 视角受限(高处/越界)→size/abs_pose 不可信，下游过滤
        "state": {"type": ["string", "null"]},
        "affordance": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "verified_by": {"type": "array", "items": {"type": "string"}},
        "last_seen": {"type": "string"},
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
    jsonschema.validate(obj, MEMORY_RECORD_SCHEMA)
    return obj
