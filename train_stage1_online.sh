#!/bin/bash
# ==============================================================================
# ViSpec Stage 1 在线训练(main_online.py)
# ------------------------------------------------------------------------------
# 与离线 train_stage1.sh 区别:目标模型常驻显存,训练循环里实时 forward 算
# hidden state,不读预存 ckpt。输入直接是 ShareGPT 原始 json,无需预生成。
#
# 超参严格对齐 README 2.1(lr=3e-5 / bs=1 / max-len=4096)。
#
# 多卡:每张卡常驻一份目标模型(7B,bf16 约 16G)+ 草稿模型,数据并行。
#       main_online.py 里已用 torch.cuda.set_device(local_process_index)
#       把每个进程绑到自己的物理卡,避免 NCCL Duplicate GPU。
#
# 用法:
#     bash train_stage1_online.sh                        # 默认全量 [0,68623) / 5 卡
#     END=20000 bash train_stage1_online.sh              # 只用前 20000 条
#     GPUS="0,1,2,3" bash train_stage1_online.sh         # 指定 4 卡
# ==============================================================================

set -e
cd "$(dirname "$0")"
ulimit -n 1048576 2>/dev/null || true
export TMPDIR=/tmp
export HF_DATASETS_CACHE=/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/cache/dataset/

BASEPATH="${BASEPATH:-./model/Qwen2.5-VL-7B-Instruct}"
CONFIGPATH="${CONFIGPATH:-vispec/train/qwen2.5_vl_7B_config.json}"
DATA_JSON="${DATA_JSON:-data/train/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json}"
CPDIR="${CPDIR:-./checkpoints/stage1_qwen7b_online}"
START="${START:-0}"
END="${END:-68623}"            # ShareGPT 全量 68623 条(有效率约 30%)
GPUS="${GPUS:-0,1,2,3,4}"
PORT="${PORT:-29500}"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}

mkdir -p "$CPDIR"

echo "=============================================================="
echo " ViSpec Stage 1 在线训练"
echo "   basepath   = $BASEPATH"
echo "   data_json  = $DATA_JSON  [$START, $END)"
echo "   cpdir      = $CPDIR"
echo "   GPU        = $GPUS  (NGPU=$NGPU)   port=$PORT"
echo "=============================================================="

# 单卡不能加 --multi_gpu(会报错);多卡才加
MULTI=""
if [[ "$NGPU" -gt 1 ]]; then MULTI="--multi_gpu"; fi

CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch \
  $MULTI --num_processes="$NGPU" --main_process_port="$PORT" \
  -m --mixed_precision=bf16 \
  vispec.train.main_online \
  --cpdir="$CPDIR" \
  --basepath="$BASEPATH" \
  --configpath="$CONFIGPATH" \
  --data-json="$DATA_JSON" \
  --start="$START" \
  --end="$END" \
  --begin-epoch=0 \
  --bs=1 \
  --lr=3e-5 \
  --max-len=4096 \
  --num-workers=8
