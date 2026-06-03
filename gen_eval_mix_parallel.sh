#!/bin/bash
# ==============================================================================
# ViSpec Stage 2 评测集采样多卡并行预生成
# ------------------------------------------------------------------------------
# 从 5 个评测集 (MME / MM-Vet / SQA / TextVQA / VQAv2) 抽样共 8000 条,用每个
# 评测集自身的 prompt 模板生成长回复,落到 data/train/gen_mm_eval_mix/<bench>/<gpu_id>/。
#
# 数据分配 (合计 8000):
#   mmvet   218   (全用)
#   mme    1800   (池 2374)
#   sqa    1598   (池 2017,test 含图)
#   textvqa 1400  (池 1667)
#   vqav2  2984   (池 2984,全用)
#
# 用法:
#   bash gen_eval_mix_parallel.sh                                # 默认 GPU=0,1,2  bs=48
#   GPUS="0,1,2,3" BATCH_SIZE=48 bash gen_eval_mix_parallel.sh    # 自定义
#   BENCHES="mmvet mme" bash gen_eval_mix_parallel.sh              # 只跑部分 bench
#
# 时间预算: bs=48 ≈ 28 条/min/卡 → 8000 / 28 / 3 卡 ≈ 95 min 全部跑完。
# ==============================================================================

set -e
cd "$(dirname "$0")"
ulimit -n 1048576 2>/dev/null || true
export TMPDIR=/tmp

MODEL="${MODEL:-./model/Qwen2.5-VL-7B-Instruct}"
OUTDIR="${OUTDIR:-data/train/gen_mm_eval_mix}"
GPUS="${GPUS:-0,1,2}"
BATCH_SIZE="${BATCH_SIZE:-48}"
MAXTOK="${MAXTOK:-1024}"
TEMP="${TEMP:-1.0}"
BENCHES="${BENCHES:-mmvet mme sqa textvqa vqav2}"

# 每个 bench 的目标抽样数 (与计划严格对应)
declare -A BENCH_N
BENCH_N[mmvet]=218
BENCH_N[mme]=1800
BENCH_N[sqa]=1598
BENCH_N[textvqa]=1400
BENCH_N[vqav2]=2984

IFS=',' read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}

LOGDIR="$OUTDIR/_logs"
mkdir -p "$OUTDIR" "$LOGDIR"

echo "=============================================================="
echo " ViSpec Stage 2 评测集采样并行预生成"
echo "   model     = $MODEL"
echo "   outdir    = $OUTDIR"
echo "   GPU       = $GPUS  (NGPU=$NGPU)"
echo "   bs=$BATCH_SIZE  max_new_tokens=$MAXTOK  temperature=$TEMP"
echo "   benches   = $BENCHES"
echo "=============================================================="

for BENCH in $BENCHES; do
  N=${BENCH_N[$BENCH]:-0}
  if [[ "$N" -le 0 ]]; then
    echo "[skip] $BENCH: 未在 BENCH_N 配置或目标=0,跳过"
    continue
  fi
  CHUNK=$(( (N + NGPU - 1) / NGPU ))   # 向上取整

  echo ""
  echo ">>>>>> bench=$BENCH  目标=$N  分片大小=$CHUNK"
  echo "------"

  PIDS=()
  for idx in "${!GPU_ARR[@]}"; do
    gpu="${GPU_ARR[$idx]}"
    s=$(( idx * CHUNK ))
    e=$(( s + CHUNK ))
    if [[ $e -gt $N ]]; then e=$N; fi
    if [[ $s -ge $N ]]; then break; fi

    LOG="$LOGDIR/${BENCH}_gpu_${gpu}.log"
    echo ">> GPU $gpu  bench=$BENCH  范围 [$s, $e)  日志 $LOG"
    python -m vispec.ge_data.gen_eval_mix_responses \
      --bench "$BENCH" \
      --outdir "$OUTDIR" \
      --index "$gpu" \
      --start "$s" --end "$e" \
      --model "$MODEL" \
      --gpu_index "$gpu" \
      --max_new_tokens "$MAXTOK" \
      --temperature "$TEMP" \
      --batch-size "$BATCH_SIZE" \
      > "$LOG" 2>&1 &
    PIDS+=($!)
  done

  echo "已启动 ${#PIDS[@]} 个进程,PID: ${PIDS[*]}"
  echo "等待 $BENCH 全部完成..."
  wait
  CUR=$(find "$OUTDIR/$BENCH" -name 'data_*.pt' 2>/dev/null | wc -l)
  echo "[$BENCH] 完成  累计产出: $CUR 条 (目标 $N)"
done

TOTAL=$(find "$OUTDIR" -name 'data_*.pt' 2>/dev/null | wc -l)
echo ""
echo "=============================================================="
echo " 全部完成!evaluation-mix 数据 $TOTAL 条"
echo " 输出根目录: $OUTDIR"
echo ""
echo " 训练时合并主数据 + 评测 mix 的最简方法:"
echo "   mkdir -p data/train/gen_mm_combined"
echo "   ln -sf \$(realpath $OUTDIR) data/train/gen_mm_combined/eval_mix"
echo "   ln -sf \$(realpath data/train/gen_mm_online)/0 data/train/gen_mm_combined/online_0"
echo "   ln -sf \$(realpath data/train/gen_mm_online)/1 data/train/gen_mm_combined/online_1"
echo "   ln -sf \$(realpath data/train/gen_mm_online)/2 data/train/gen_mm_combined/online_2"
echo "   DATAPATH=data/train/gen_mm_combined bash train_stage2_online.sh"
echo "=============================================================="
