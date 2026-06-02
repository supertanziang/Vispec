#!/bin/bash
# ==============================================================================
# ViSpec Stage 2 长回复多卡并行预生成
# ------------------------------------------------------------------------------
# 把 [START, END) 的索引范围平均切成 NGPU 份,每张卡一个进程并行 generate。
# 每个分片写到 OUTDIR/<gpu_id>/ 子目录(--index = gpu_id),互不覆盖。
# 单卡内通过 --batch-size N 同时跑多张图,A100-80G 实测最优 bs=48
# (bs48: 38GB / 28.5 有效条/min;bs64: 49GB / 29.3 有效条/min,边际收益仅 2.7%)。
#
# 索引语义说明:
#   gen_stage2_responses.py 内部维护一个 present-index 缓存
#   (data/train/LLaVA-Pretrain/_present_indices.shuffled42.json),
#   把 shuffle(seed=42) 后所有「本地确实有图」的样本下标一次性筛出来。
#   之后 [START, END) 落在 present 空间上 ——
#   END - START 就是有效产出条数,不需要再开 4× 缓冲。
#   首次运行会扫一遍 558k 条文件存在性(约 1-2 分钟),之后秒级读 cache。
#   本地总有效池 ~13.3 万条,够 60k 训练数据有充足余量。
#
# 用法:
#     bash gen_stage2_parallel.sh                                       # 默认 [0,60000) / 2 卡 / bs=48
#     START=0 END=100000 NGPU=8 BATCH_SIZE=48 bash gen_stage2_parallel.sh
#     GPUS="0,1,2,3" BATCH_SIZE=48 bash gen_stage2_parallel.sh          # 指定用哪几张卡
#     BATCH_SIZE=64 bash gen_stage2_parallel.sh                         # 显存裕量足时可上 bs=64
# ==============================================================================

set -e
cd "$(dirname "$0")"
ulimit -n 1048576 2>/dev/null || true
export TMPDIR=/tmp

MODEL="${MODEL:-./model/Qwen2.5-VL-7B-Instruct}"
DATA_PATH="${DATA_PATH:-data/train/LLaVA-Pretrain/}"
OUTDIR="${OUTDIR:-data/train/gen_mm_online}"
START="${START:-0}"
END="${END:-60000}"
MAXTOK="${MAXTOK:-1024}"
TEMP="${TEMP:-1.0}"
GPUS="${GPUS:-0,1,2}"
BATCH_SIZE="${BATCH_SIZE:-64}"  # A100-80G 甜点:bs=48 显存 ~38GB / 单条 2.10s,bs=64 仅快 2.7% 显存却到 49GB

# 解析 GPU 列表
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}

TOTAL=$((END - START))
CHUNK=$(( (TOTAL + NGPU - 1) / NGPU ))   # 向上取整

mkdir -p "$OUTDIR"
LOGDIR="$OUTDIR/_logs"
mkdir -p "$LOGDIR"

echo "=============================================================="
echo " ViSpec Stage 2 长回复并行预生成"
echo "   model        = $MODEL"
echo "   data_path    = $DATA_PATH"
echo "   outdir       = $OUTDIR"
echo "   present 索引 = [$START, $END)  共 $TOTAL  (=有效产出条数,已预过滤缺图样本)"
echo "   GPU          = $GPUS  (NGPU=$NGPU)"
echo "   每卡分片大小 = $CHUNK"
echo "   max_new_tok  = $MAXTOK   temperature = $TEMP   batch_size = $BATCH_SIZE"
echo "=============================================================="
echo " 首次运行会建立 present-index 缓存(约 1-2 分钟扫文件),之后秒级"
echo "=============================================================="

PIDS=()
for idx in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$idx]}"
  s=$(( START + idx * CHUNK ))
  e=$(( s + CHUNK ))
  if [[ $e -gt $END ]]; then e=$END; fi
  if [[ $s -ge $END ]]; then break; fi

  echo ">> GPU $gpu  ->  分片 index=$gpu  范围 [$s, $e)  日志 $LOGDIR/gpu_$gpu.log"
  python -m vispec.ge_data.gen_stage2_responses \
    --outdir "$OUTDIR" \
    --index "$gpu" \
    --start "$s" --end "$e" \
    --model "$MODEL" \
    --data-path "$DATA_PATH" \
    --gpu_index "$gpu" \
    --max_new_tokens "$MAXTOK" \
    --temperature "$TEMP" \
    --batch-size "$BATCH_SIZE" \
    > "$LOGDIR/gpu_$gpu.log" 2>&1 &
  PIDS+=($!)
done

echo ""
echo "已启动 ${#PIDS[@]} 个预生成进程,PID: ${PIDS[*]}"
echo "实时进度: tail -f $LOGDIR/gpu_*.log"
echo "当前有效产出数: find $OUTDIR -name '*.pt' | wc -l"
echo ""
echo "等待全部完成..."
wait
echo ""
echo "=============================================================="
echo " 预生成完成!有效数据条数: $(find "$OUTDIR" -name '*.pt' | wc -l)"
echo "=============================================================="
