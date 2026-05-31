#!/bin/bash
# ==============================================================================
# ViSpec Stage 2.1 — Initial Training (text-only)
# ------------------------------------------------------------------------------
# 严格按照 README 2.1 节给出的命令离线训练,不改超参。
#   - basepath        : 本地 Qwen2.5-VL-7B-Instruct
#   - configpath      : vispec/train/qwen2.5_vl_7B_config.json
#   - tmpdir          : Stage 1 用的纯文本 ShareGPT 数据(已生成)
#   - cpdir           : 输出 checkpoint 目录
#   - lr / bs / max-len / num-workers / begin-epoch 全部按 README 默认
# 用法:
#     bash train_stage1.sh
# ==============================================================================

set -e
cd "$(dirname "$0")"            # 切到项目根

# 避免大量并发文件句柄报错
ulimit -n 1048576 2>/dev/null || true

# 把 TMPDIR 切到短路径,避免 PyTorch multiprocessing 的
# "AF_UNIX path too long" (Linux unix socket 路径上限 108 字节)
export TMPDIR=/tmp

BASEPATH="${BASEPATH:-./model/Qwen2.5-VL-7B-Instruct}"
CONFIGPATH="${CONFIGPATH:-vispec/train/qwen2.5_vl_7B_config.json}"
# 默认指向 200 条数据目录;如用其他规模,用 TMPDIR_DATA=... bash train_stage1.sh 覆盖
# 例: TMPDIR_DATA=./data/train/gen_text_test/qwen2.5vl_shargpt_0_1000_mubf16 bash train_stage1.sh
TMPDIR_DATA="${TMPDIR_DATA:-./data/train/gen_text_test/qwen2.5vl_shargpt_0_200_mubf16}"
CPDIR="${CPDIR:-./checkpoints/stage1_qwen7b}"

mkdir -p "$CPDIR"

echo "=============================================================="
echo " ViSpec Stage 2.1 训练"
echo "   basepath    = $BASEPATH"
echo "   configpath  = $CONFIGPATH"
echo "   tmpdir      = $TMPDIR_DATA"
echo "   cpdir       = $CPDIR"
echo "=============================================================="

accelerate launch --multi_gpu \
  -m --mixed_precision=bf16 \
  vispec.train.main \
  --cpdir="$CPDIR" \
  --basepath="$BASEPATH" \
  --begin-epoch=0 \
  --bs=1 \
  --configpath="$CONFIGPATH" \
  --lr=3e-5 \
  --max-len=4096 \
  --num-workers=8 \
  --tmpdir="$TMPDIR_DATA"
