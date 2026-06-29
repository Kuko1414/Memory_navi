#!/bin/bash
# ============================================
# Qwen3-VL + LibreChat 一键启动脚本
#   8B (Orchestrator) → port 8000
#   LibreChat UI       → port 3080
#   (4B 暂不用：方案书已说明，相关代码已注释保留；需要时取消注释即可启用)
# ============================================

# 模型路径
MODEL_8B="/home/kuko/.cache/huggingface/hub/qwen/Qwen3-VL-8B-Instruct"
# MODEL_4B="/home/kuko/.cache/huggingface/hub/Qwen/Qwen3-VL-4B-Instruct"   # 4B 暂不用

# 环境
VENV_PYTHON="/home/kuko/miniconda3/envs/vllm/bin/python3"
CUDA_BASE="/home/kuko/miniconda3/envs/vllm/lib/python3.11/site-packages/nvidia/cu13"

# LibreChat（conda env: librechat）
LIBRECHAT_DIR="/home/kuko/LibreChat"
LIBRECHAT_DB="/home/kuko/librechat-data/db"

# 端口
PORT_8B=8000
# PORT_4B=8001        # 4B 暂不用
PORT_UI=3080          # LibreChat

# 显存分配
# 0.80：给同机其他 GPU 程序(Webots/游戏等)留 ~1GB+ 余量；GPU 独占时可调回 0.85
GPU_MEM_SOLO=0.80
# GPU_MEM_4B_DUAL=0.42   # 4B 暂不用
# GPU_MEM_8B_DUAL=0.48   # 双模型时给 8B 收紧的显存；现 8B 单卡 solo 用 GPU_MEM_SOLO

# max-model-len: solo 宽松（LibreChat agent 需容纳工具定义+大传感器消息+图像+多轮历史）
# 注意：这只决定单序列可用的 KV 长度上限，不额外占显存（KV 池已由 gpu-memory-utilization 预留）。
# 8192 实测在 agent 多轮+LaserScan(数百 ranges) 下会 400 超长；16384 更稳。
MAX_LEN_SOLO=16384
# MAX_LEN_4B_DUAL=2048   # 4B 暂不用
# MAX_LEN_8B_DUAL=4096   # 双模型时收紧；现 8B solo 用 MAX_LEN_SOLO

# served-model-name: 给 LibreChat 一个干净的 model id（而非 HF 长路径）
SERVED_8B="qwen3-vl-8b"

# 去 conda（ROS2 Humble 需 py3.10，conda base 是 py3.13 会冲突）
conda deactivate 2>/dev/null || true

# 清理代理
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy

# vLLM 基础参数 (max-model-len 按模式传参)
# --enable-auto-tool-choice + --tool-call-parser hermes: 开启 Qwen3 function calling
#   (Qwen3 用 Hermes 风格工具模板; agent_core 的 tool_loop 用非流式, 避开 hermes 流式 raw-text bug)
VLLM_BASE="--trust-remote-code --enforce-eager --enable-auto-tool-choice --tool-call-parser hermes"

# ============================================
_start_8b() {
    local mem=${1:-$GPU_MEM_SOLO}
    local max_len=${2:-$MAX_LEN_SOLO}
    echo "🚀 启动 Qwen3-VL-8B (Orchestrator) port $PORT_8B (max_len=$max_len, mem=$mem, served=$SERVED_8B)..."
    CUDA_HOME="$CUDA_BASE" VLLM_USE_FLASHINFER_SAMPLER=0 \
    nohup $VENV_PYTHON -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_8B" \
      --served-model-name "$SERVED_8B" \
      $VLLM_BASE \
      --max-model-len $max_len \
      --gpu-memory-utilization $mem \
      --host 0.0.0.0 \
      --port $PORT_8B \
      > /tmp/vllm_8b.log 2>&1 &
    echo "   PID: $! → 日志: /tmp/vllm_8b.log"
}

# # ----- 4B 暂不用（保留备用，取消下面整段注释即可启用）-----
# _start_4b() {
#     local mem=${1:-$GPU_MEM_SOLO}
#     local max_len=${2:-$MAX_LEN_SOLO}
#     echo "🚀 启动 Qwen3-VL-4B (Fast) port $PORT_4B (max_len=$max_len, mem=$mem)..."
#     CUDA_HOME="$CUDA_BASE" VLLM_USE_FLASHINFER_SAMPLER=0 \
#     nohup $VENV_PYTHON -m vllm.entrypoints.openai.api_server \
#       --model "$MODEL_4B" \
#       --served-model-name qwen3-vl-4b \
#       $VLLM_BASE \
#       --max-model-len $max_len \
#       --gpu-memory-utilization $mem \
#       --host 0.0.0.0 \
#       --port $PORT_4B \
#       > /tmp/vllm_4b.log 2>&1 &
#     echo "   PID: $! → 日志: /tmp/vllm_4b.log"
# }

_wait_ready() {
    local port=$1
    local label=$2
    echo "   等待 $label 加载..."
    for i in $(seq 1 90); do
      if curl -s http://localhost:$port/health > /dev/null 2>&1; then
        echo "   ✅ $label 就绪"
        return 0
      fi
      sleep 5
    done
    echo "   ❌ $label 超时"
    return 1
}

_start_librechat() {
    echo "💬 启动 LibreChat (http://localhost:$PORT_UI)..."
    # 本地 MongoDB（dbpath 不存在则建）
    mkdir -p "$LIBRECHAT_DB"
    nohup conda run -n librechat mongod --dbpath "$LIBRECHAT_DB" --port 27017 \
      > /tmp/mongod.log 2>&1 &
    echo "   MongoDB PID: $! → 日志: /tmp/mongod.log"
    # LibreChat 后端（托管已构建好的前端）
    nohup conda run -n librechat bash -lc "cd $LIBRECHAT_DIR && npm run backend" \
      > /tmp/librechat.log 2>&1 &
    echo "   LibreChat PID: $! → 日志: /tmp/librechat.log → http://localhost:$PORT_UI"
}

_start_relays() {
    echo "📷 启动相机中继（JPEG + depth summary）..."
    source /opt/ros/humble/setup.bash 2>/dev/null
    nohup python3 /home/kuko/Kuko1414/memory_navi/image_jpeg_relay.py \
      /agent0/camera/image_color /agent0/camera/image_color/compressed \
      > /tmp/jpeg_relay.log 2>&1 &
    echo "   JPEG relay PID: $! → /tmp/jpeg_relay.log"
    nohup python3 /home/kuko/Kuko1414/memory_navi/depth_summary_relay.py \
      /agent0/camera/depth/image /agent0/camera/depth/columns \
      > /tmp/depth_relay.log 2>&1 &
    echo "   depth relay PID: $! → /tmp/depth_relay.log"
}

# ============================================
case "${1:-all}" in
  relays)
    _start_relays
    ;;

  # ----- 8B only -----
  8b)
    _start_8b $GPU_MEM_SOLO $MAX_LEN_SOLO
    _wait_ready $PORT_8B "8B"
    ;;

  # # ----- 4B only（暂不用，取消注释启用）-----
  # 4b)
  #   _start_4b $GPU_MEM_SOLO $MAX_LEN_SOLO
  #   _wait_ready $PORT_4B "4B"
  #   ;;

  # ----- LibreChat UI only -----
  chat|ui)
    _start_librechat
    ;;

  # ----- 全部启动 (8B + LibreChat) -----
  all)
    echo "========================================"
    echo "  Qwen3-VL-8B + LibreChat 部署"
    echo "  8B (Orchestrator) :$PORT_8B  |  LibreChat :$PORT_UI"
    echo "========================================"

    echo ""
    _start_8b $GPU_MEM_SOLO $MAX_LEN_SOLO
    _wait_ready $PORT_8B "8B"

    echo ""
    _start_librechat

    echo ""
    echo "========================================"
    echo "  ✅ 全部就绪"
    echo "  8B API:     http://localhost:$PORT_8B/v1  (model: $SERVED_8B)"
    echo "  LibreChat:  http://localhost:$PORT_UI"
    echo "  停止:       bash $0 stop"
    echo "========================================"
    ;;

  # ----- 停止 -----
  stop)
    echo "🛑 停止所有服务..."
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null && echo "   vLLM 实例已停止"
    pkill -f "npm run backend" 2>/dev/null && echo "   LibreChat 后端已停止"
    pkill -f "mongod --dbpath $LIBRECHAT_DB" 2>/dev/null && echo "   MongoDB 已停止"
    ;;

  # ----- 状态 -----
  status)
    echo "📊 服务状态:"
    curl -s http://localhost:$PORT_8B/health > /dev/null 2>&1 \
      && echo "   Qwen3-VL-8B  ($PORT_8B): ✅" || echo "   Qwen3-VL-8B  ($PORT_8B): ❌"
    curl -s http://localhost:$PORT_UI > /dev/null 2>&1 \
      && echo "   LibreChat    ($PORT_UI): ✅" || echo "   LibreChat    ($PORT_UI): ❌"
    echo ""
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null
    ;;

  *)
    echo "用法: bash $0 {all|8b|chat|stop|status}"
    echo ""
    echo "  all    启动 8B Orchestrator + LibreChat (默认)"
    echo "  8b     只启动 8B Orchestrator (port $PORT_8B)"
    echo "  chat   只启动 LibreChat UI (port $PORT_UI)"
    echo "  stop   停止所有服务"
    echo "  status 查看服务状态"
    echo ""
    echo "  (4B 暂不用，相关代码在脚本内已注释保留)"
    ;;
esac
