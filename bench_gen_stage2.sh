#!/bin/bash
# Stage2 generate batch_size 拐点 benchmark
# 单卡 A100-80G,从 bs=24 起步,逐步加大,直到 OOM 或显存饱和
# 每组用相同样本范围,记录:耗时、有效产出、显存峰值

set -u
cd "$(dirname "$0")"
ulimit -n 1048576 2>/dev/null || true
export TMPDIR=/tmp
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_ds_cache}"

GPU="${GPU:-0}"
MODEL="${MODEL:-./model/Qwen2.5-VL-7B-Instruct}"
DATA_PATH="${DATA_PATH:-data/train/LLaVA-Pretrain/}"
MAXTOK="${MAXTOK:-1024}"
TEMP="${TEMP:-1.0}"
# 范围放大一点,确保任意 bs 都能 flush ≥2 个 batch(本地图约 1/4 有效)
START="${START:-0}"
END="${END:-600}"

BSLIST=(${BSLIST:-24 32 48 64})

OUTROOT="data/_bench_stage2"
LOGDIR="$OUTROOT/_logs"
mkdir -p "$LOGDIR"

echo "========== Stage2 generate batch_size benchmark =========="
echo " GPU=$GPU   range=[$START,$END)   max_new_tok=$MAXTOK"
echo " bs list = ${BSLIST[*]}"
echo "==========================================================="

for BS in "${BSLIST[@]}"; do
  OUT="$OUTROOT/bs${BS}"
  rm -rf "$OUT"
  mkdir -p "$OUT"
  LOG="$LOGDIR/bs${BS}.log"
  SMI="$LOGDIR/bs${BS}.smi"
  : > "$SMI"

  echo ""
  echo ">> bs=$BS  out=$OUT  log=$LOG"

  # 后台轮询显存
  ( while true; do
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$GPU" >> "$SMI"
      sleep 2
    done ) &
  SMIPID=$!

  T0=$(date +%s)
  python -m vispec.ge_data.gen_stage2_responses \
      --outdir "$OUT" \
      --index 0 \
      --start "$START" --end "$END" \
      --model "$MODEL" \
      --data-path "$DATA_PATH" \
      --gpu_index "$GPU" \
      --max_new_tokens "$MAXTOK" \
      --temperature "$TEMP" \
      --batch-size "$BS" \
      > "$LOG" 2>&1
  RC=$?
  T1=$(date +%s)

  kill $SMIPID 2>/dev/null
  wait $SMIPID 2>/dev/null

  DUR=$((T1 - T0))
  NPT=$(find "$OUT" -name '*.pt' 2>/dev/null | wc -l)
  PEAK=$(awk -F',' '{gsub(/ /,"",$2); if($2+0>m) m=$2+0} END{print m}' "$SMI")
  OOM_HIT=$(grep -ci "OutOfMemory\|CUDA out of memory" "$LOG")

  if [[ $RC -ne 0 || $OOM_HIT -gt 0 ]]; then
    STATE="FAIL(rc=$RC,oom=$OOM_HIT)"
  else
    STATE="OK"
  fi

  if [[ $NPT -gt 0 && $DUR -gt 0 ]]; then
    THR=$(awk -v n="$NPT" -v d="$DUR" 'BEGIN{printf "%.2f", n*60.0/d}')
  else
    THR="n/a"
  fi

  printf "    [bs=%-3d] %-25s dur=%4ds  produced=%4d  thr=%6s/min  peak=%5s MiB\n" \
         "$BS" "$STATE" "$DUR" "$NPT" "$THR" "${PEAK:-?}"

  # OOM 后没必要再加大
  if [[ $OOM_HIT -gt 0 ]]; then
    echo "    -> 检测到 OOM,停止递增"
    break
  fi
done

echo ""
echo "==========================================================="
echo " 详细日志:$LOGDIR/"
echo "==========================================================="
