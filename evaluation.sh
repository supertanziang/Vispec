#!/bin/bash
# ==============================================================================
# ViSpec 一键评测脚本 evaluation.sh
# ------------------------------------------------------------------------------
# 功能:一运行就输出「不同模型 × 不同温度 × 不同数据集」的
#        加速比 (ratio / speedup) 和 接收率 (τ / acceptance length)。
#
# 用法:
#     bash evaluation.sh                          # 用下方默认配置直接跑
#     bash evaluation.sh --only-summary           # 跳过推理,只读已有结果出表
#     bash evaluation.sh --models qwen,qwen_3b     # 临时覆盖模型
#     MODELS="qwen" BENCHMARKS="mmvet" bash evaluation.sh   # 用环境变量覆盖
#
# 本质:对 vispec/evaluation/run_eval.py 的封装。所有真正的执行逻辑在 run_eval.py,
#       这里只负责设置配置 + 调用,方便“一个 bash 跑完拿表格”。
# ==============================================================================

set -e
cd "$(dirname "$0")"            # 切到脚本所在目录(项目根)

# ------------------------------------------------------------------------------
# 配置区 —— 改这里即可(也可用同名环境变量覆盖)
# ------------------------------------------------------------------------------

# 要评测的模型(逗号分隔)。可选: qwen, qwen_3b, llava, llava_13b, llava_1.5
#   —— 具体路径在 run_eval.py 顶部的 MODELS 注册表里配置
MODELS="${MODELS:-qwen}"

# 要评测的数据集(逗号分隔)。
#   立即可用(无需下数据): mmvet, sqa
#   HF 自动下载(数据较大): coco_caption(~19GB,较慢), hr_bench(高分辨率,需多卡)
#   需手动下数据到 data/<name>: mme, gqa, textvqa, vqav2, seed_bench, vizwiz
BENCHMARKS="${BENCHMARKS:-mmvet,sqa,mme}"

# 采样温度(逗号分隔)。0.0=贪心(快、可复现);1.0=随机采样(对齐论文双档)
TEMPERATURES="${TEMPERATURES:-0.0}"

# 用哪块 GPU(单卡评测)
GPU="${GPU:-0}"

# ViSpec 投机解码超参(spec 推理用)
#   DEPTH       : 草稿 token 树深度(猜多少步)
#   TOP_K       : 树宽度(每步保留几个候选)
#   TOTAL_TOKEN : 从树里最终选多少 token 交目标模型验证
#   NUM_Q       : ImgAdaptor query 向量数,须与训练一致(2q 档=2)
DEPTH="${DEPTH:-3}"
TOP_K="${TOP_K:-8}"
TOTAL_TOKEN="${TOTAL_TOKEN:-30}"
NUM_Q="${NUM_Q:-2}"

# 单条样本最多生成多少 token
MAX_NEW_TOKEN="${MAX_NEW_TOKEN:-1024}"

# HuggingFace token(数据/模型自动下载时若被限流则需要;没有可留空)
export HF_TOKEN="${HF_TOKEN:-}"

# 避免大量并发文件句柄报错
ulimit -n 1048576 2>/dev/null || true

# ------------------------------------------------------------------------------
# 执行 —— 透传给 run_eval.py;脚本额外的命令行参数($@)原样转发
#   注意:DEPTH/TOP_K/TOTAL_TOKEN/NUM_Q/MAX_NEW_TOKEN 是改 run_eval.py 顶部 CONFIG 的,
#   这里通过环境变量传入,run_eval.py 已支持读取(见下方 Python 端)。
# ------------------------------------------------------------------------------
echo "=============================================================="
echo " ViSpec 评测"
echo "   models       = $MODELS"
echo "   benchmarks   = $BENCHMARKS"
echo "   temperatures = $TEMPERATURES"
echo "   gpu          = $GPU"
echo "   spec 超参    = depth=$DEPTH top_k=$TOP_K total_token=$TOTAL_TOKEN num_q=$NUM_Q"
echo "=============================================================="

VISPEC_DEPTH="$DEPTH" \
VISPEC_TOP_K="$TOP_K" \
VISPEC_TOTAL_TOKEN="$TOTAL_TOKEN" \
VISPEC_NUM_Q="$NUM_Q" \
VISPEC_MAX_NEW_TOKEN="$MAX_NEW_TOKEN" \
python -m vispec.evaluation.run_eval \
    --models "$MODELS" \
    --benchmarks "$BENCHMARKS" \
    --temperatures "$TEMPERATURES" \
    --gpu "$GPU" \
    "$@"
