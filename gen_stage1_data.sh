#!/bin/bash
# ==============================================================================
# ViSpec Stage 1 数据生成脚本
# ------------------------------------------------------------------------------
# 用法:
#   bash gen_stage1_data.sh                    # 默认生成 200 条
#   bash gen_stage1_data.sh --num-samples 5000 # 生成 5000 条
#   NUM_SAMPLES=1000 bash gen_stage1_data.sh   # 环境变量覆盖
#
# 输出目录: data/train/gen_text_test/qwen2.5vl_shargpt_0_<N>_mubf16/
# 上限: 68623 条 (ShareGPT 数据集总量)
# ==============================================================================

set -e
cd "$(dirname "$0")"
ulimit -n 1048576 2>/dev/null || true
export TMPDIR=/tmp

# ---------- 超参数 ----------
NUM_SAMPLES="${NUM_SAMPLES:-200}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

MODEL="${MODEL:-./model/Qwen2.5-VL-7B-Instruct}"
OUTDIR="${OUTDIR:-data/train/gen_text_test}"
OUTDIR_FULL="${OUTDIR}/qwen2.5vl_shargpt_0_${NUM_SAMPLES}_mubf16"

echo "=============================================================="
echo " ViSpec Stage 1 数据生成(纯文本)"
echo "   num_samples  = $NUM_SAMPLES  (上限 68623)"
echo "   model        = $MODEL"
echo "   output_dir   = $OUTDIR_FULL"
echo "=============================================================="

# Stage 1 是纯文本,不需要解压图片,直接生成
TMPDIR=/tmp python -m vispec.ge_data.allocation_qwen_shargpt \
  --outdir="$OUTDIR" \
  --start=0 \
  --end="${NUM_SAMPLES}" \
  --model="$MODEL"

echo ""
echo "=========================================================="
GENERATED=$(find "${OUTDIR_FULL}" -name '*.ckpt' 2>/dev/null | wc -l)
echo " 完成!生成 ckpt: ${GENERATED} 条"
echo " 输出目录: ${OUTDIR_FULL}"
echo ""
echo " 用以下命令训练:"
echo "   TMPDIR_DATA=${OUTDIR_FULL} bash train_stage1.sh"
echo "=========================================================="
