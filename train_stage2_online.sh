#!/bin/bash
# ==============================================================================
# ViSpec Stage 2 在线训练(main_mtp_online.py)
# ------------------------------------------------------------------------------
# 与离线 train_stage2.sh 区别:目标模型常驻显存,训练循环里对(图片+预生成长回复)
# 实时 forward 算 hidden state。需先用 gen_stage2_parallel.sh 预生成长回复 token。
#
# 超参严格对齐 README 2.2(lr=3e-6 / bs=1 / max-len=4096 / mtp-steps=1 / num-q=2)。
#
# 完整流程:
#   1) 先并行预生成长回复(只存 token,几 KB/条):
#        bash gen_stage2_parallel.sh
#   2) 训练(默认加载 Stage 1 在线 ckpt 的 state_20):
#        bash train_stage2_online.sh
#
# 用法:
#     bash train_stage2_online.sh
#     GPUS="0,1,2,3" DATAPATH=data/train/gen_mm_online_full bash train_stage2_online.sh
# ==============================================================================

set -e
cd "$(dirname "$0")"
ulimit -n 1048576 2>/dev/null || true
export TMPDIR=/tmp
export HF_DATASETS_CACHE=/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/cache/dataset/

BASEPATH="${BASEPATH:-./model/Qwen2.5-VL-7B-Instruct}"
CONFIGPATH="${CONFIGPATH:-vispec/train/qwen2.5_vl_7B_config.json}"
DATAPATH="${DATAPATH:-data/train/gen_mm_online_full}"
CPDIR="${CPDIR:-./checkpoints/stage2_qwen7b_online}"
LOADPATH="${LOADPATH:-./checkpoints/stage1_qwen7b_online/state_20/model.safetensors}"
GPUS="${GPUS:-0,1,2,3,4}"
PORT="${PORT:-29500}"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}

mkdir -p "$CPDIR"

if [[ ! -f "$LOADPATH" ]]; then
  echo "Error: Stage 1 在线 checkpoint 不存在: $LOADPATH"
  echo "       请先跑完 train_stage1_online.sh,或用 LOADPATH=... 指定"
  exit 1
fi

NPT=$(find "$DATAPATH" -name '*.pt' 2>/dev/null | wc -l)
if [[ "$NPT" -eq 0 ]]; then
  echo "Error: 预生成长回复数据为空: $DATAPATH"
  echo "       请先跑 gen_stage2_parallel.sh 预生成"
  exit 1
fi

echo "=============================================================="
echo " ViSpec Stage 2 在线训练 (ViSpec + MTP)"
echo "   basepath   = $BASEPATH"
echo "   datapath   = $DATAPATH   (有效数据 $NPT 条)"
echo "   cpdir      = $CPDIR"
echo "   loadpath   = $LOADPATH"
echo "   GPU        = $GPUS  (NGPU=$NGPU)   port=$PORT"
echo "=============================================================="

# 单卡不能加 --multi_gpu;多卡才加
MULTI=""
if [[ "$NGPU" -gt 1 ]]; then MULTI="--multi_gpu"; fi

CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch \
  $MULTI --num_processes="$NGPU" --main_process_port="$PORT" \
  -m --mixed_precision=bf16 \
  vispec.train.main_mtp_online \
  --cpdir="$CPDIR" \
  --basepath="$BASEPATH" \
  --configpath="$CONFIGPATH" \
  --datapath="$DATAPATH" \
  --loadpath="$LOADPATH" \
  --begin-epoch=0 \
  --bs=1 \
  --lr=3e-6 \
  --max-len=4096 \
  --mtp-steps=1 \
  --num-q=2 \
  --num-workers=8 \
  --use-ours=True
