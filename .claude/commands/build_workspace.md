# /build_workspace — 新建 ROS2 工作区脚手架

> 当用户输入 `/build_workspace <项目路径>` 时，按照以下流程为新项目创建一个标准化的 ROS2 工作区结构。

---

## 阶段 0：输入解析

用户输入格式：`/build_workspace [项目路径] [项目描述]`

- **项目路径**: 新工作区的目录路径（必填，如果未提供则询问用户）
  - 可以是绝对路径，如 `/home/kuko/my_new_robot`
  - 可以是相对路径（相对于当前终端工作目录），如 `my_new_robot`
- **项目描述**: 一句话描述这个项目的目标（可选，未提供则使用默认占位符）

---

## 阶段 1：收集信息

如果用户在 prompt 中未提供完整信息，按以下优先级询问：

1. **项目路径未提供** → 询问："请问新工作区要创建在哪里？（例如：`../my_new_robot` 或绝对路径）"
2. **项目描述未提供** → 使用默认值 `"{{PROJECT_DESCRIPTION}}"`（模板占位符，用户后续手动替换）

---

## 阶段 2：创建工作区结构

### 2.1 创建目录

在指定的项目路径下创建以下目录结构：

```
<项目路径>/
├── src/
├── data/
│   ├── image/
│   └── log/
├── future/
└── scripts/
```

使用 PowerShell 的 `New-Item -ItemType Directory -Force` 递归创建所有目录。

### 2.2 生成主文档文件

从模板目录 `.claude/templates/build_workspace/` 读取模板，替换占位符后写入新工作区。

**模板位置：** `<当前工作区>/.claude/templates/build_workspace/`

**占位符替换规则：**

| 占位符 | 来源 | 说明 |
|--------|------|------|
| `{{DATE}}` | 当前日期 | 格式：`Mon.DD, YYYY`（如 `Jun.01, 2026`） |
| `{{PROJECT_NAME}}` | 用户提供的路径最后一段 | 如路径 `my_new_robot` → `my_new_robot` |
| `{{PROJECT_DESCRIPTION}}` | 用户提供的描述 | 未提供则保留占位符 |
| `{{PROJECT_TYPE}}` | 根据描述推断 | 如 "wheeled robot" / "drone" / "robot arm" |

**生成文件清单：**

| 模板文件 | 生成文件 | 位置 |
|----------|----------|------|
| `ARCHITECTURE_TEMPLATE.md` | `ARCHITECTURE.md` | 项目根目录 |
| `MEMORY_TEMPLATE.md` | `MEMORY.md` | 项目根目录 |
| `PROCESS_TEMPLATE.md` | `PROCESS.md` | 项目根目录 |
| `future_work_TEMPLATE.md` | `future_work.md` | `future/` |

生成步骤：
1. 用 Read 工具读取每个模板文件
2. 执行占位符替换（`{{...}}` → 实际值）
3. 用 Write 工具将替换后的内容写入目标文件

### 2.3 复制 Python 调试脚本

从 `.claude/templates/build_workspace/` 复制 4 个 Python 脚本到 `scripts/`：

| 模板文件 | 目标文件 | 占位符替换 |
|----------|----------|------------|
| `depth_viewer.py` | `scripts/depth_viewer.py` | `{{DEPTH_TOPIC}}` → `/camera/depth/image_raw`，`{{DATA_DIR}}` → `data` |
| `pose_logger.py` | `scripts/pose_logger.py` | `{{ODOM_TOPIC}}` → `/odom`，`{{DATA_DIR}}` → `data` |
| `rgb_viewer.py` | `scripts/rgb_viewer.py` | `{{RGB_TOPIC}}` → `/camera/color/image_raw`，`{{DATA_DIR}}` → `data` |
| `test_odom_yaw.py` | `scripts/test_odom_yaw.py` | `{{ODOM_TOPIC}}` → `/odom` |

> ⚠️ **注意：** 脚本中的话题名和路径使用默认值。用户在新项目中需要根据实际硬件调整这些默认话题名和路径。

### 2.4 创建 .gitignore

在项目根目录创建 `.gitignore` 文件，包含以下 ROS2 标准忽略项：

```
# ROS2 build artifacts
build/
install/
log/

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/

# IDE
.vscode/
.idea/

# Data (runtime generated)
data/image/*
data/log/*
!data/image/.gitkeep
!data/log/.gitkeep

# OS
.DS_Store
Thumbs.db
```

---

## 阶段 3：完成报告

生成完成后，向用户报告：

```
✅ 工作区创建完成！

📁 位置: <绝对路径>
📂 结构:
   ├── ARCHITECTURE.md    # 架构文档（ROS2 Topics / Nodes / 结构）
   ├── MEMORY.md          # 变更记录
   ├── PROCESS.md         # 项目目标与步骤
   ├── .gitignore
   ├── src/               # ROS2 包目录（空）
   ├── data/image/        # 运行时图像
   ├── data/log/          # 调试日志
   ├── future/
   │   └── future_work.md # 待确认功能
   └── scripts/           # 调试脚本
       ├── depth_viewer.py
       ├── pose_logger.py
       ├── rgb_viewer.py
       └── test_odom_yaw.py

💡 下一步:
   1. cd <项目路径>
   2. 编辑 ARCHITECTURE.md 更新 ROS2 Topic 名称
   3. 编辑 PROCESS.md 定义项目步骤
   4. 在 scripts/ 中调整话题订阅后即可用于调试
   5. colcon build 初始化 ROS2 构建
```

---

## 执行流程总览

```
用户输入: /build_workspace <路径> [描述]
    │
    ▼
┌─ 阶段 0-1: 解析输入，收集缺失信息 ────────┐
│  · 解析项目路径                            │
│  · 未提供路径？→ 询问                       │
│  · 未提供描述？→ 使用占位符                  │
└──────────────────────────────────────────┘
    │
    ▼
┌─ 阶段 2: 创建工作区 ──────────────────────┐
│  · 创建目录结构 (src/, data/, future/, scripts/) │
│  · 从 .claude/templates/build_workspace/ 读取 MD 模板       │
│  · 替换 {{占位符}} 并写入目标文件             │
│  · 复制 4 个 Python 脚本（应用默认话题名）    │
│  · 创建 .gitignore                          │
└──────────────────────────────────────────┘
    │
    ▼
┌─ 阶段 3: 完成报告 ────────────────────────┐
│  · 打印工作区结构                           │
│  · 提示下一步操作                            │
└──────────────────────────────────────────┘
```
