# 开发环境配置文档

> RTX 5090 32GB | 30GB RAM | Ubuntu 22.04.5 LTS | Miniconda3

---

## 硬件概览

| 组件 | 型号/规格 |
|------|----------|
| GPU | NVIDIA GeForce RTX 5090 (32 GB) |
| 驱动 | 580.159.03 |
| CUDA | 13.0 |
| CPU | 32 核 |
| RAM | 30 GB |
| OS | Ubuntu 22.04.5 LTS |
| Conda | Miniconda3 26.3.2 (`/home/kuko/miniconda3`) |

---

## Conda 环境一览

| 环境名 | Python | 用途 |
|--------|--------|------|
| `base` | 3.13.13 | 系统默认 |
| `isaaclab` | 3.11.15 | Isaac Lab + Isaac Sim 5.1.0 |
| `vllm` | 3.11.15 | vLLM + Qwen3-VL 推理服务 |

---

## Isaac Lab

### 基本信息

| 项目 | 详情 |
|------|------|
| **位置** | `/home/kuko/IsaacLab/` |
| **Isaac Lab** | 2.3.0 (pip: 0.47.2) |
| **Isaac Sim** | 5.1.0 (App 目录: `apps/isaacsim_4_5`) |
| **Conda 环境** | `isaaclab` |
| **核心包** | `isaaclab_tasks` 0.11.6, `isaaclab_rl` 0.4.4, `isaaclab_mimic` 1.0.15 |

### 启动

```bash
conda activate isaaclab
cd /home/kuko/IsaacLab
./isaaclab.sh                         # GUI 模式
# 或
conda run -n isaaclab python your_script.py
```

---

## vLLM + Qwen3-VL（双模型）

### 架构

```
┌─────────────────────────────────────────┐
│  Qwen3-VL-8B (Orchestrator)             │
│  Port 8000  |  17GB VRAM                │
│  复杂推理、多步骤编排、高质量回答          │
├─────────────────────────────────────────┤
│  Qwen3-VL-4B (Fast)                     │
│  Port 8001  |  ~8GB VRAM                │
│  快速响应、简单任务、预处理               │
├─────────────────────────────────────────┤
│  LibreChat UI (conda env: librechat)    │
│  Port 3080  |  Agents + ROS MCP 工具     │
└─────────────────────────────────────────┘
```

### 模型信息

| 模型 | 大小 | 路径 |
|------|------|------|
| Qwen3-VL-8B-Instruct | ~17 GB FP16 | `/home/kuko/.cache/huggingface/hub/qwen/Qwen3-VL-8B-Instruct/` |
| Qwen3-VL-4B-Instruct | ~8 GB FP16 | `/home/kuko/.cache/huggingface/hub/Qwen/Qwen3-VL-4B-Instruct/` |

### 版本

| 组件 | 版本 |
|------|------|
| vLLM | 0.22.1 |
| Qwen3-VL 系列 | Instruct (FP16) |
| Conda 环境 | `vllm` |

### 显存分配策略

| 模式 | 8B | 4B | 合计 |
|------|-----|-----|------|
| 单独启动 | 85% (~27GB) | 85% (~27GB) | — |
| 双模型并行 | 65% (~21GB) | 28% (~9GB) | ~30GB / 32GB |

> 双模型时 `max-model-len` 降至 8192 以控制 KV cache 开销

---

## 启动方式

### 一键启动（推荐）

由 `/home/kuko/Kuko1414/start_all.sh` 管理：

```bash
bash /home/kuko/Kuko1414/start_all.sh           # 8B + LibreChat
bash /home/kuko/Kuko1414/start_all.sh 8b        # 仅 8B (port 8000)
bash /home/kuko/Kuko1414/start_all.sh chat      # 仅 LibreChat (port 3080)
bash /home/kuko/Kuko1414/start_all.sh stop      # 停止全部
bash /home/kuko/Kuko1414/start_all.sh status    # 查看状态
# (4B 暂不用，脚本内注释保留；需要时取消注释)
```

### 手动启动

```bash
# 8B 模型 (solo)
CUDA_HOME=/home/kuko/miniconda3/envs/vllm/lib/python3.11/site-packages/nvidia/cu13/ \
VLLM_USE_FLASHINFER_SAMPLER=0 \
conda run -n vllm vllm serve \
  /home/kuko/.cache/huggingface/hub/qwen/Qwen3-VL-8B-Instruct \
  --trust-remote-code --gpu-memory-utilization 0.85 \
  --max-model-len 32768 --enforce-eager \
  --host 0.0.0.0 --port 8000

# LibreChat（conda env: librechat；前端需先 `npm run frontend` 构建过一次）
conda run -n librechat mongod --dbpath ~/librechat-data/db --port 27017 &
conda run -n librechat bash -lc "cd ~/LibreChat && npm run backend"
```

### 关键配置说明

| 参数 | 值 | 原因 |
|------|----|------|
| `VLLM_USE_FLASHINFER_SAMPLER=0` | 禁用 FlashInfer 采样 | CUDA 13 + FlashInfer CCCL 不兼容 ([Issue #2195](https://github.com/flashinfer-ai/flashinfer/issues/2195)) |
| `--enforce-eager` | 禁用 CUDA Graph | 同上 |
| `--trust-remote-code` | 允许远程代码 | Qwen3-VL 必需 |

### LibreChat UI

浏览器打开 **`http://localhost:3080`**（首次需注册一个本地账号）

- 端点选 **Agents**，底层模型选 `Qwen-vLLM / qwen3-vl-8b`
- 在 Agent Builder 里加入 `ros` MCP server 并勾选工具（connect_to_robot / get_topics /
  get_topic_type / subscribe_once / publish_once / view_saved_image）
- 让 Qwen 自己 function-call 操控 Webots 机器人；配置见 `~/LibreChat/librechat.yaml`

---

## API 调用示例

```python
from openai import OpenAI
import httpx, base64

# 8B Orchestrator
client_8b = OpenAI(
    base_url="http://localhost:8000/v1", api_key="no-key",
    http_client=httpx.Client(proxy=None, trust_env=False),
)
# 4B Fast
client_4b = OpenAI(
    base_url="http://localhost:8001/v1", api_key="no-key",
    http_client=httpx.Client(proxy=None, trust_env=False),
)

# 图片理解
with open("image.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

resp = client_8b.chat.completions.create(
    model="qwen3-vl-8b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "描述这张图片"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ],
    }],
    max_tokens=200,
)
print(resp.choices[0].message.content)
```

---

## 文件索引

| 文件 | 用途 |
|------|------|
| `/home/kuko/Kuko1414/start_all.sh` | 8B + LibreChat 一键启动/停止脚本 |
| `~/LibreChat/librechat.yaml` | LibreChat 配置（vLLM endpoint + ros MCP） |
| `/home/kuko/Kuko1414/SETUP.md` | 本文档 |
| `/home/kuko/IsaacLab/isaaclab.sh` | Isaac Lab 启动脚本 |
| `/tmp/vllm_8b.log` | 8B 模型日志 |
| `/tmp/librechat.log` | LibreChat 后端日志 |
| `/tmp/mongod.log` | MongoDB 日志 |
