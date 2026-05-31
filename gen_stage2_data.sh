#!/bin/bash
# ==============================================================================
# ViSpec Stage 2 数据生成脚本
# ------------------------------------------------------------------------------
# 用法:
#   bash gen_stage2_data.sh                    # 默认生成 200 条
#   bash gen_stage2_data.sh --num-samples 1000 # 生成 1000 条
#   NUM_SAMPLES=500 bash gen_stage2_data.sh    # 环境变量覆盖
#
# 流程:
#   1. 按 shuffle(seed=42) 顺序,精确解压前 N 条对应的图片(按需解压,不解压整包)
#   2. 调用 allocation_qwen_pretrain_gen.py 在 8 张 GPU 上并行生成 .ckpt 数据
#
# 输出目录: data/train/gen_mm_test/qwen_pretrain_gen_0_<N>_mufp16/
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
IMAGES_ZIP="${IMAGES_ZIP:-data/train/LLaVA-Pretrain/images.zip}"
JSON_PATH="${JSON_PATH:-data/train/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json}"
IMAGE_DIR="${IMAGE_DIR:-data/train/LLaVA-Pretrain}"
OUTDIR="${OUTDIR:-data/train/gen_mm_test}"
OUTDIR_FULL="${OUTDIR}/qwen_pretrain_gen_0_${NUM_SAMPLES}_mufp16"

echo "=============================================================="
echo " ViSpec Stage 2 数据生成"
echo "   num_samples  = $NUM_SAMPLES"
echo "   model        = $MODEL"
echo "   images_zip   = $IMAGES_ZIP"
echo "   output_dir   = $OUTDIR_FULL"
echo "=============================================================="

# ---------- Step 1: 按需解压前 N 条对应的图片 ----------
echo ""
echo "[Step 1] 解压前 ${NUM_SAMPLES} 条样本所需图片..."

TMPDIR=/tmp python - <<PYEOF
import json, os, subprocess, sys
from datasets import Dataset

n = int("${NUM_SAMPLES}")
zip_path = "${IMAGES_ZIP}"
image_dir = "${IMAGE_DIR}"
json_path = "${JSON_PATH}"

ds = Dataset.from_list(json.load(open(json_path))).shuffle(seed=42)
sub = ds.select(range(n))

need = []
for row in sub:
    p = os.path.join(image_dir, row["image"])
    if not os.path.exists(p):
        need.append(row["image"])

if not need:
    print(f"  全部 {n} 张图已解压,跳过。")
    sys.exit(0)

print(f"  需要解压 {len(need)} 张图(共 {n} 张,{n-len(need)} 张已存在)...")

# 写临时文件列表给 unzip -@
import tempfile
with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    f.write("\n".join(need))
    tmpfile = f.name

ret = subprocess.run(
    f"unzip -n -q {zip_path} -d {image_dir} < {tmpfile}",
    shell=True
)
os.unlink(tmpfile)

# 验证
missing_after = [r for r in need if not os.path.exists(os.path.join(image_dir, r))]
if missing_after:
    print(f"  警告:仍有 {len(missing_after)} 张图未找到,可能不在 zip 内,数据生成时会跳过。")
else:
    print(f"  解压完成,全部 {len(need)} 张图就绪。")
PYEOF

# ---------- Step 2: 生成 .ckpt 数据 ----------
echo ""
echo "[Step 2] 调用 allocation 脚本生成 ${NUM_SAMPLES} 条 ckpt..."

TMPDIR=/tmp python -m vispec.ge_data.allocation_qwen_pretrain_gen \
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
echo "   TMPDIR_DATA=${OUTDIR_FULL} bash train_stage2.sh"
echo "=========================================================="
