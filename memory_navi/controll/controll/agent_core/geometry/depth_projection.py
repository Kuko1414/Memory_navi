"""Depth back-projection + TF chain → absolute world coordinates（几何核心）。

把"Qwen ROI(像素) + depth(米) → 物体世界坐标"的纯数学管线独立成模块，
可脱离 ROS/LLM 单测。输入全部显式注入（不隐式读全局配置）。

数学（Webots Camera 约定——相机沿 -Z 看、+X 右、+Y 上）：
  A) pinhole 反投影：像素(u,v) + depth d → Webots 相机系 3D
     相机沿 -Z_w 看；图像面上 (u=0 左, v=0 顶)
     xn = (u - cx)/fx,  yn = (v - cy)/fy
     s = sqrt(xn² + yn² + 1)
     X_w = -xn * d/s,  Y_w = -yn * d/s,  Z_w = -d/s  （Z_w<0=前方）
  B) Webots→OpenCV/ROS 约定转换：+Z_c=前, +X_c=右, +Y_c=下
     X_c = X_w,  Y_c = -Y_w,  Z_c = -Z_w (= d/s)
  C) TF 链：P_map = T_map_base · T_base_camera · (X_c, Y_c, Z_c, 1)^T
     T_base_camera 由实际 TF quat (0, ~0.075, 0, ~0.997) 构造 —— 约
     绕Y上仰 8.6° + 相机Z→robotX 的 90° 隐含旋转。

约定（与 Webots/ROS 一致）：
  - 世界坐标系：+X 东、+Y 北/左、+Z 上，yaw=0 朝 +X，+yaw CCW
  - 深度 d = RangeFinder 实测沿光线的欧式距离（米）
  - Qwen 归一化坐标：(0,0)=左上,(1000,1000)=右下 → 先映射到像素

注意：Webots RangeFinder 与 Camera 共用同一 TF 节点（同位姿/同 FOV），
故深度图像素(u,v)与 RGB 图像素(u,v)一一对应。
"""
import math


# ---- Webots Camera 已知参数（break_room agent0，从 .wbt/URDF + TF）----
DEFAULT_K = {
    "fx": 253.9, "fy": 253.9, "cx": 320.0, "cy": 240.0,
    "width": 640, "height": 480,
}
# camera→base_link 固定 TF（从 /tf_static 实测: quat (0,0.07493,0,0.99719), trans (0.17,0,0.13)）
# 该 quat: y=sin(θ/2)≈0.07493 → θ≈0.15 rad ≈ 8.6° 绕 Y 上仰（base_link Y 轴）
DEFAULT_TF_QUAT = {"x": 0.0, "y": 0.0749297, "z": 0.0, "w": 0.9971888}
DEFAULT_TF_TRANS = {"x": 0.17, "y": 0.0, "z": 0.13}


# ---------------------------------------------------------------------------
# 纯函数（可脱离 ROS/LLM 单测）
# ---------------------------------------------------------------------------

def quat_to_rot_matrix(qx: float, qy: float, qz: float, qw: float) -> tuple:
    """单位四元数 → 3x3 旋转矩阵（行主序 tuple-of-tuples）。"""
    # 标准四元数→旋转矩阵（Hamilton 约定，与 ROS tf2 一致）
    x2, y2, z2 = 2 * qx * qx, 2 * qy * qy, 2 * qz * qz
    xy2, xz2, xw2 = 2 * qx * qy, 2 * qx * qz, 2 * qx * qw
    yz2, yw2, zw2 = 2 * qy * qz, 2 * qy * qw, 2 * qz * qw
    return (
        (1 - y2 - z2, xy2 - zw2, xz2 + yw2),
        (xy2 + zw2, 1 - x2 - z2, yz2 - xw2),
        (xz2 - yw2, yz2 + xw2, 1 - x2 - y2),
    )


def qwen_norm_to_pixel(u_norm: float, v_norm: float, width: int, height: int) -> tuple:
    """Qwen3-VL 归一化坐标(0-1000) → 像素坐标（最近邻）。"""
    u = round(u_norm / 999.0 * (width - 1))
    v = round(v_norm / 999.0 * (height - 1))
    return max(0, min(width - 1, u)), max(0, min(height - 1, v))


def pixel_to_camera(u: float, v: float, depth_m: float, K: dict) -> tuple:
    """pinhole 反投影（Webots 约定：相机沿 -Z 看）→ OpenCV/ROS 相机系(Xc,Yc,Zc)。

    Webots Camera: Z_w < 0 为前。输出 OpenCV/ROS 约定（Z_c > 0 为前）。
    返回 (X_c, Y_c, Z_c)。
    """
    fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
    xn = (u - cx) / fx               # 偏右为正
    yn = (v - cy) / fy               # 偏下为正（v=0 顶→cy 中→yn 负=物体在上方）
    s = math.sqrt(xn * xn + yn * yn + 1.0)
    Z_c = depth_m / s                # +Z 向前
    X_c = xn * Z_c                   # OpenCV: X = (u-cx)/fx * Z
    Y_c = yn * Z_c                   # OpenCV: Y = (v-cy)/fy * Z (+Y下/-Y上)
    return X_c, Y_c, Z_c


def tf_camera_to_base(X_c: float, Y_c: float, Z_c: float,
                      _R=None, trans: dict = None) -> tuple:
    """相机系 3D(OpenCV: +Z前 +X右 +Y下) → base_link 系(+X前 +Y左 +Z上)。

    经实测验证（将已知世界坐标反投验证）的正确映射:
      X_b = Z_c + tx   (相机前→机器人前)
      Y_b = -X_c + ty  (相机右→机器人右=-Y方向,即 -X_c)
      Z_b = -Y_c + tz  (相机下→机器人下=-Z方向,即 -Y_c)
    相机上仰仅 ~8.6°(0.15rad)，对水平方向物体的 X/Y 影响 <5cm，
    此处用简化映射。trans 默认 DEFAULT_TF_TRANS (0.17, 0, 0.13)。
    """
    if trans is None:
        trans = DEFAULT_TF_TRANS
    X_b = Z_c + trans["x"]
    Y_b = -X_c + trans["y"]
    Z_b = -Y_c + trans["z"]
    return X_b, Y_b, Z_b


def tf_base_to_map(X_b: float, Y_b: float, Z_b: float, robot_pose: dict) -> tuple:
    """base_link 系 → map 系 (x, y, z)_world。robot_pose={x,y,yaw_deg}。

    世界系：+X 东、+Y 北/左、+Z 上，yaw=0 朝 +X，+yaw CCW(=左转)。
    """
    yaw = math.radians(robot_pose["yaw_deg"])
    c, s_ = math.cos(yaw), math.sin(yaw)
    x_map = c * X_b - s_ * Y_b + robot_pose["x"]
    y_map = s_ * X_b + c * Y_b + robot_pose["y"]
    z_map = Z_b     # z 暂不补偿 robot 起伏（平地仿真）
    return x_map, y_map, z_map


def back_project(u: float, v: float, depth_m: float,
                 K: dict = None, R_cam_base=None, trans_cam_base: dict = None,
                 robot_pose: dict = None) -> dict:
    """完整管线：像素(u,v) + depth → world (x,y,z)。

    Args:
        u, v: 像素坐标
        depth_m: RangeFinder 深度读数（米，>0）
        K: 相机内参，默认 DEFAULT_K
        R_cam_base: 3x3 旋转矩阵（camera→base），默认按 TF quat 构造
        trans_cam_base: 平移 {"x","y","z"}，默认 DEFAULT_TF_TRANS
        robot_pose: 当前位姿 {"x","y","yaw_deg"}，必填

    Returns:
        {abs_pose:{x,y,z}, pixel_uv:[u,v], depth_used_m, P_cam:{X_c,Y_c,Z_c}}
    """
    if not robot_pose:
        raise ValueError("robot_pose is required for map-frame projection")
    K = K or DEFAULT_K

    # A) 像素+深度 → OpenCV/ROS 相机系 3D（Z_c>0 前）
    X_c, Y_c, Z_c = pixel_to_camera(u, v, depth_m, K)

    # B) 相机系 → base_link（用实际 TF 旋转矩阵+平移）
    X_b, Y_b, Z_b = tf_camera_to_base(X_c, Y_c, Z_c, R_cam_base, trans_cam_base)

    # C) base_link → map
    x_w, y_w, z_w = tf_base_to_map(X_b, Y_b, Z_b, robot_pose)

    return {
        "abs_pose": {"x": round(x_w, 3), "y": round(y_w, 3), "z": round(z_w, 3)},
        "pixel_uv": [round(u), round(v)],
        "depth_used_m": round(depth_m, 3),
        "P_cam": {"X_c": round(X_c, 3), "Y_c": round(Y_c, 3), "Z_c": round(Z_c, 3)},
    }


def bearing_to_depth_column(bearing: str, depths: list) -> float:
    """按 bearing(left/center/right) 取 depth 数组对应列段的中位有效距离。"""
    if not depths:
        return None
    n = len(depths)
    third = max(1, n // 3)
    b = (bearing or "center").lower()
    if "left" in b or "左" in b:
        seg = depths[:third]
    elif "right" in b or "右" in b:
        seg = depths[-third:]
    else:
        seg = depths[third:n - third] or depths
    valid = sorted(d for d in seg if isinstance(d, (int, float)) and d >= 0)
    return float(round(valid[len(valid) // 2], 3)) if valid else None


# ---------------------------------------------------------------------------
# 目标推导
# ---------------------------------------------------------------------------

def derive_observation_point(obj_abs: dict, *,
                             behind: bool = True,
                             heading_deg: float = None,
                             standoff_m: float = 2.0) -> dict:
    """从物体世界位姿推导观察点（"绕到物体后面/前面"）。

    Args:
        obj_abs: back_project 返回的 abs_pose {"x","y","z"}
        behind: True = 绕到后面(x 更大/东侧), False = 前面
        heading_deg: 物体朝向(度)；None 则用默认推断（朝 -x = 门朝机器人始发侧）
        standoff_m: 观察点离物体的距离

    Returns:
        {"x": float, "y": float, "z": float} 建议观察点 world 坐标
    """
    ox, oy, oz = obj_abs["x"], obj_abs["y"], obj_abs.get("z", 0.0)
    if heading_deg is None:
        # 默认：红柜等家具门朝 -x(西/机器人起始侧, heading=180°)→ behind=东(0°)
        # 即：门朝西=heading 180°, "后"=反方向=东=0°
        heading_deg = 180.0   # 门朝西
    # behind → 朝 heading 的反方向退后 standoff
    dir_rad = math.radians(heading_deg + 180.0 if behind else heading_deg)
    return {
        "x": round(ox + standoff_m * math.cos(dir_rad), 3),
        "y": round(oy + standoff_m * math.sin(dir_rad), 3),
        "z": round(oz, 3),
    }
