#!/bin/bash
# ==============================================================================
# ViSpec Stage 2.2 — Training with ViSpec (multimodal)
# ------------------------------------------------------------------------------
# 严格按照 README 2.2 节给出的命令离线训练,不改超参。
#   - basepath        : 本地 Qwen2.5-VL-7B-Instruct
#   - configpath      : vispec/train/qwen2.5_vl_7B_config.json
#   - tmpdir          : Stage 2 用的多模态数据 (qwen_pretrain_gen)
#   - cpdir           : 输出 checkpoint 目录
#   - loadpath        : Stage 1 训完的 state_20/model.safetensors
#   - lr=3e-6 / bs=1 / max-len=4096 / num-workers=8 / mtp-steps=1 / num-q=2
#   - use-ours=True   : 启用 ViSpec
# 用法:
#     bash train_stage2.sh                                # 用默认 stage1 ckpt
#     LOADPATH=path/to/model.safetensors bash train_stage2.sh
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
# 默认指向 200 条数据目录;如用其他规模,用 TMPDIR_DATA=... bash train_stage2.sh 覆盖
# 例: TMPDIR_DATA=./data/train/gen_mm_test/qwen_pretrain_gen_0_1000_mufp16 bash train_stage2.sh
TMPDIR_DATA="${TMPDIR_DATA:-./data/train/gen_mm_test/qwen_pretrain_gen_0_200_mufp16}"
CPDIR="${CPDIR:-./checkpoints/stage2_qwen7b}"
LOADPATH="${LOADPATH:-./checkpoints/stage1_qwen7b/state_20/model.safetensors}"

mkdir -p "$CPDIR"

if [[ ! -f "$LOADPATH" ]]; then
  echo "Error: Stage 1 checkpoint not found at $LOADPATH"
  echo "       请先跑完 train_stage1.sh,或用 LOADPATH=... 指定别的 ckpt"
  exit 1
fi

echo "=============================================================="
echo " ViSpec Stage 2.2 训练 (multimodal, ViSpec)"
echo "   basepath    = $BASEPATH"
echo "   configpath  = $CONFIGPATH"
echo "   tmpdir      = $TMPDIR_DATA"
echo "   cpdir       = $CPDIR"
echo "   loadpath    = $LOADPATH"
echo "=============================================================="

accelerate launch --multi_gpu \
  -m --mixed_precision=bf16 \
  vispec.train.main_mtp \
  --cpdir="$CPDIR" \
  --basepath="$BASEPATH" \
  --begin-epoch=0 \
  --bs=1 \
  --configpath="$CONFIGPATH" \
  --loadpath="$LOADPATH" \
  --lr=3e-6 \
  --max-len=4096 \
  --mtp-steps=1 \
  --num-q=2 \
  --num-workers=8 \
  --tmpdir="$TMPDIR_DATA" \
  --use-ours=True
