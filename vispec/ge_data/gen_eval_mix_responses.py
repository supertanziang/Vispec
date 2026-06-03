"""
ViSpec Stage 2 评测集采样长回复预生成
==============================================================================
从 5 个评测集 (MME / MM-Vet / SQA / TextVQA / VQAv2) 抽样,用每个评测集
自身的 prompt 模板 (vispec/evaluation/<bench>_prompt.py) 让目标模型
generate 长回复,产出与 gen_stage2_responses.py 完全一致的轻量 .pt:
  {full_ids, prompt_len, image_file}
供 OnlineMMDataset 直接读取。

输出: <outdir>/<bench>/<index>/data_<i>.pt
图片: HF arrow 内嵌的 PIL 会先落盘到 <image-out-dir>/<bench>/<id>.jpg

用法 (单卡单 bench):
  python -m vispec.ge_data.gen_eval_mix_responses \
    --bench mmvet --start 0 --end 218 --index 0 --gpu_index 0 \
    --batch-size 4
==============================================================================
"""

import argparse
import json
import os

parser = argparse.ArgumentParser(description="Stage 2 eval-mix long-response pre-generation")
parser.add_argument(
    "--bench", type=str, required=True,
    choices=["mme", "mmvet", "sqa", "textvqa", "vqav2"],
    help="选哪个评测集"
)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=100)
parser.add_argument("--index", type=int, default=0,
                    help="分片号 (与 gpu_index 通常对应),决定输出子目录名")
parser.add_argument("--gpu_index", type=int, nargs="+", default=[0])
parser.add_argument("--outdir", type=str, default="data/train/gen_mm_eval_mix")
parser.add_argument("--image-out-dir", type=str, default=None,
                    help="HF arrow 内嵌图片落盘根目录,默认 <outdir>/_images")
parser.add_argument("--max_new_tokens", type=int, default=1024)
parser.add_argument("--model", type=str, default="./model/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--batch-size", type=int, default=48)
parser.add_argument("--seed", type=int, default=42, help="数据集 shuffle 种子,与 60K 主数据保持一致")
# 路径(各评测集的根目录)
parser.add_argument("--mme-root",     type=str, default="data/eval/MME")
parser.add_argument("--mmvet-root",   type=str, default="data/eval/mmvet")
parser.add_argument("--sqa-dataset",  type=str, default="data/eval/sqa/dataset")
parser.add_argument("--sqa-metadata", type=str, default="data/eval/sqa/metadata")
parser.add_argument("--textvqa-root", type=str, default="data/eval/textvqa")
parser.add_argument("--vqav2-root",   type=str, default="data/eval/vqav2")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]

import random

import torch
from datasets import Dataset, load_from_disk
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

# ---------- ScienceQA: 复用现成的工具函数构造 prompt 文本 ----------
from vispec.evaluation.scienceqa_prompt import (
    create_one_example,
    get_question_text,
    get_context_text,
    get_choice_text,
    get_answer,
    get_lecture_text,
    get_solution_text,
)


SYSTEM_MSG = {
    "role": "system",
    "content": [{
        "type": "text",
        "text": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
    }],
}

# ============================================================================
# Pool 加载: 5 个评测集各一个 loader,统一返回
#   list[ {"id": str, "image_path": str (磁盘绝对路径), "_meta": ...} ]
# image_path 总是磁盘绝对路径; _meta 字段供 make_messages 使用。
# ============================================================================
def _save_pil_to_disk(img, out_path):
    """HF arrow 内嵌 PIL → 磁盘 jpg。已存在则跳过。
    多进程并发安全:tmp 文件名带 PID,避免不同进程争用同一 .tmp。
    """
    if os.path.exists(out_path):
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if img.mode != "RGB":
        img = img.convert("RGB")
    tmp = f"{out_path}.tmp.{os.getpid()}"
    img.save(tmp, format="JPEG", quality=92)
    try:
        os.replace(tmp, out_path)
    except FileNotFoundError:
        # 已被另一进程 rename 走;若目标已存在则 OK,否则继续报错
        if not os.path.exists(out_path):
            raise


def load_pool_mme(image_out_root):
    """MME: 读 jsonl + 用 os.walk 建文件名→绝对路径索引。
    去重: 同 (image_path, question) 只保留一条。
    图片在磁盘上,直接引用绝对路径,不导出。
    """
    img_root = os.path.join(args.mme_root, "MME_Benchmark_release_version/MME_Benchmark")
    name2path = {}
    for dp, _, fs in os.walk(img_root):
        for fn in fs:
            if fn.lower().endswith((".jpg", ".png", ".jpeg")):
                name2path.setdefault(fn, os.path.abspath(os.path.join(dp, fn)))

    pool = []
    seen = set()
    with open(os.path.join(args.mme_root, "llava_mme.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            if d["image"] not in name2path:
                continue
            text = d["text"].partition("\n")[0]
            img_abs = name2path[d["image"]]
            key = (img_abs, text)
            if key in seen:
                continue
            seen.add(key)
            pool.append({
                "id": str(d["question_id"]),
                "image_path": img_abs,
                "_meta": {"text": text},
            })
    return pool


def load_pool_mmvet(image_out_root):
    """MM-Vet: HF arrow,字段 id / image (PIL) / question / answer / capability。"""
    raw = load_from_disk(args.mmvet_root)
    out_dir = os.path.join(image_out_root, "mmvet")
    pool = []
    for ex in raw:
        rid = ex["id"]
        img_abs = os.path.abspath(os.path.join(out_dir, f"{rid}.jpg"))
        _save_pil_to_disk(ex["image"], img_abs)
        pool.append({
            "id": rid,
            "image_path": img_abs,
            "_meta": {"question": ex["question"]},
        })
    return pool


def load_pool_textvqa(image_out_root):
    """TextVQA: HF arrow,字段 question_id / image / question / ..."""
    raw = load_from_disk(args.textvqa_root)
    out_dir = os.path.join(image_out_root, "textvqa")
    pool = []
    for ex in raw:
        rid = str(ex["question_id"])
        img_abs = os.path.abspath(os.path.join(out_dir, f"{rid}.jpg"))
        _save_pil_to_disk(ex["image"], img_abs)
        pool.append({
            "id": rid,
            "image_path": img_abs,
            "_meta": {"question": ex["question"]},
        })
    return pool


def load_pool_vqav2(image_out_root):
    """VQAv2: HF arrow,字段 question_id / image / question(去掉 \\n 后段)。"""
    raw = load_from_disk(args.vqav2_root)
    out_dir = os.path.join(image_out_root, "vqav2")
    pool = []
    for ex in raw:
        rid = str(ex["question_id"])
        img_abs = os.path.abspath(os.path.join(out_dir, f"{rid}.jpg"))
        _save_pil_to_disk(ex["image"], img_abs)
        pool.append({
            "id": rid,
            "image_path": img_abs,
            "_meta": {"text": ex["question"].partition("\n")[0]},
        })
    return pool


def load_pool_sqa(image_out_root):
    """ScienceQA: 合并 metadata + HF arrow 中的图片,只保留 test split 含图样本。
    与 gen_baseline_answer_sqa.py:load_data 完全一致的 stitching 逻辑。
    """
    problems = json.load(open(os.path.join(args.sqa_metadata, "problems.json")))
    pid_splits = json.load(open(os.path.join(args.sqa_metadata, "pid_splits.json")))
    captions = json.load(open(os.path.join(args.sqa_metadata, "captions.json")))["captions"]
    data = load_from_disk(args.sqa_dataset)

    for qid in problems:
        problems[qid]["caption"] = captions[qid] if qid in captions else ""

    for split_name, qids in pid_splits.items():
        if split_name not in ["train", "val", "test"]:
            continue
        sn = "validation" if split_name == "val" else split_name
        split = data[sn]
        for i, qid in enumerate(qids):
            problems[qid]["image"] = split[i]["image"]

    out_dir = os.path.join(image_out_root, "sqa")
    pool = []
    for qid in pid_splits["test"]:
        prob = problems[qid]
        if prob["image"] is None:
            continue
        img_abs = os.path.abspath(os.path.join(out_dir, f"{qid}.jpg"))
        _save_pil_to_disk(prob["image"], img_abs)
        pool.append({
            "id": qid,
            "image_path": img_abs,
            "_meta": {"problem": {k: v for k, v in prob.items() if k != "image"}},
        })
    return pool


# ============================================================================
# make_messages_<bench>: 构造 (messages, images_path_list) 给 processor。
# 内容严格复制对应 vispec/evaluation/<bench>_prompt.py 中的 examples 结构。
# ============================================================================
def make_messages_mme(rec):
    msgs = [SYSTEM_MSG, {
        "role": "user",
        "content": [
            {"type": "text", "text": rec["_meta"]["text"]},
            {"type": "text", "text": "Please answer with an explanation."},
            {"type": "image"},
        ],
    }]
    return msgs, [rec["image_path"]]


def make_messages_mmvet(rec):
    msgs = [SYSTEM_MSG, {
        "role": "user",
        "content": [
            {"type": "text", "text": rec["_meta"]["question"]},
            {"type": "text", "text": "Please answer with an explanation."},
            {"type": "image"},
        ],
    }]
    return msgs, [rec["image_path"]]


def make_messages_textvqa(rec):
    msgs = [SYSTEM_MSG, {
        "role": "user",
        "content": [
            {"type": "text", "text": rec["_meta"]["question"]},
            {"type": "text", "text": (
                "Perform an OCR task on the provided image. Please extract the text accurately "
                "and provide a detailed explanation of the process. Ensure the response is "
                "comprehensive and well-structured."
            )},
            {"type": "image"},
        ],
    }]
    return msgs, [rec["image_path"]]


def make_messages_vqav2(rec):
    """vqav2_prompt.py 的字段是 data["text"],与 gen_baseline_answer_vqav2.py:36 保持一致。"""
    msgs = [SYSTEM_MSG, {
        "role": "user",
        "content": [
            {"type": "text", "text": rec["_meta"]["text"]},
            {"type": "text", "text": "Please answer with an explanation."},
            {"type": "image"},
        ],
    }]
    return msgs, [rec["image_path"]]


# ScienceQA 评测脚本里默认 args(对齐 evaluation.sh 的 BENCHMARKS["sqa"]["extra_args"]):
#   prompt_format=QCM-ALE / use_caption=False / options=A..E
class _SqaCfg:
    prompt_format = "QCM-ALE"
    use_caption = False
    options = ["A", "B", "C", "D", "E"]


def make_messages_sqa(rec):
    """对照 scienceqa_prompt.build_prompt 的 0-shot test 分支构造 messages。"""
    prob = rec["_meta"]["problem"]
    cfg = _SqaCfg()
    question = get_question_text(prob)
    context = get_context_text(prob, cfg.use_caption)
    choice = get_choice_text(prob, cfg.options)
    answer = get_answer(prob, cfg.options)
    lecture = get_lecture_text(prob)
    solution = get_solution_text(prob)
    test_example = create_one_example(
        cfg.prompt_format, question, context, choice, answer, lecture, solution,
        test_example=True,
    )
    test_example = test_example.replace(
        "Answer:",
        'Your answer should begin with "The answer is". Please answer with an explanation. Answer:',
    )
    msgs = [SYSTEM_MSG, {
        "role": "user",
        "content": [
            {"type": "text", "text": test_example},
            {"type": "image"},
        ],
    }]
    return msgs, [rec["image_path"]]


BENCH_LOADERS = {
    "mme":     load_pool_mme,
    "mmvet":   load_pool_mmvet,
    "sqa":     load_pool_sqa,
    "textvqa": load_pool_textvqa,
    "vqav2":   load_pool_vqav2,
}
BENCH_MAKERS = {
    "mme":     make_messages_mme,
    "mmvet":   make_messages_mmvet,
    "sqa":     make_messages_sqa,
    "textvqa": make_messages_textvqa,
    "vqav2":   make_messages_vqav2,
}


# ============================================================================
# 主流程
# ============================================================================
image_out_root = args.image_out_dir or os.path.join(args.outdir, "_images")
os.makedirs(image_out_root, exist_ok=True)

print(f"[load_pool] bench={args.bench} ...")
pool = BENCH_LOADERS[args.bench](image_out_root)
print(f"[load_pool] bench={args.bench} 池容量 = {len(pool)}")

# shuffle 一次,然后切片(把 [args.start, args.end) 作用在 shuffle 后的序列上,
# 与 60K 主数据 shuffle(seed=42) 的语义一致)
rng = random.Random(args.seed)
indices = list(range(len(pool)))
rng.shuffle(indices)
sliced_idx = indices[args.start: args.end]
sub = [pool[i] for i in sliced_idx]
if len(sub) == 0:
    raise ValueError(f"[start,end)=[{args.start},{args.end}) 为空 (池容量 {len(pool)})")
print(f"[slice] {args.bench} 切片 [{args.start},{args.end}) → 实际 {len(sub)} 条")

# Processor & 模型
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    args.model, use_fast=True, min_pixels=min_pixels, max_pixels=max_pixels
)
processor.tokenizer.padding_side = "left"

bigmodel = AutoModelForImageTextToText.from_pretrained(
    args.model, device_map="auto", torch_dtype="auto"
)
bigmodel.eval()


@torch.no_grad()
def gen_batch(batch_records):
    """对 batch 做 generate,返回 list[{full_ids, prompt_len, image_file}]。
    逻辑与 gen_stage2_responses.py:172-221 保持一致(左 padding / 剥 pad / EOS 截断)。
    """
    make = BENCH_MAKERS[args.bench]
    images_per_sample = []
    texts = []
    image_files = []
    for rec in batch_records:
        msgs, img_list = make(rec)
        text = processor.apply_chat_template(msgs, add_generation_prompt=True)
        # img_list 长度永远 1(每条 1 张图)
        images_per_sample.append(Image.open(img_list[0]))
        texts.append(text)
        image_files.append(img_list[0])

    inputs = processor(
        images=images_per_sample,
        text=texts,
        padding=True,
        return_tensors="pt",
    ).to(bigmodel.device)

    prompt_lens = inputs["attention_mask"].sum(dim=-1).tolist()
    pad_lens = (inputs["input_ids"].shape[-1] - inputs["attention_mask"].sum(dim=-1)).tolist()

    outs = bigmodel.generate(
        **inputs,
        do_sample=args.temperature != 0,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    eos_id = processor.tokenizer.eos_token_id
    results = []
    for i, out_ids in enumerate(outs):
        ids = out_ids[pad_lens[i]:].cpu()
        prompt_end = prompt_lens[i]
        gen_part = ids[prompt_end:]
        eos_pos = (gen_part == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            ids = torch.cat([ids[:prompt_end], gen_part[: eos_pos[0].item() + 1]])
        results.append({
            "full_ids": ids,
            "prompt_len": int(prompt_lens[i]),
            "image_file": image_files[i],
        })
    return results


# ---------- 输出目录 ----------
outdir = os.path.join(args.outdir, args.bench, str(args.index))
os.makedirs(outdir, exist_ok=True)


def writedata(name, data_point, idx):
    final = f"{name}/data_{idx}.pt"
    tmp = final + ".tmp"
    torch.save(data_point, tmp)
    os.replace(tmp, final)


def scan_done(name):
    done = set()
    if not os.path.isdir(name):
        return done
    for fn in os.listdir(name):
        if fn.startswith("data_") and fn.endswith(".pt"):
            try:
                done.add(int(fn[len("data_"):-len(".pt")]))
            except ValueError:
                pass
        elif fn.endswith(".pt.tmp"):
            try:
                os.remove(os.path.join(name, fn))
            except OSError:
                pass
    return done


def open_or_none(path):
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except (FileNotFoundError, OSError):
        return False


bs = args.batch_size
buffer = []        # list[(local_idx, record)]
done = scan_done(outdir)
if done:
    print(f"[resume] outdir 已有 {len(done)} 条产出,本次将自动跳过")


def flush_buffer():
    if not buffer:
        return
    items = [b[1] for b in buffer]
    indices_local = [b[0] for b in buffer]
    try:
        recs = gen_batch(items)
        idx_for_recs = indices_local
    except Exception as e:
        print(f"[err] batch generate 失败,回退到逐条重试: {e}")
        recs, idx_for_recs = [], []
        for ii, d in zip(indices_local, items):
            try:
                recs.extend(gen_batch([d]))
                idx_for_recs.append(ii)
            except Exception as ee:
                print(f"[skip] 单条失败 {d['image_path']}: {ee}")
    for ii, rec in zip(idx_for_recs, recs):
        writedata(outdir, rec, ii)
    buffer.clear()


for i, rec in enumerate(tqdm(sub)):
    if i in done:
        continue
    if not open_or_none(rec["image_path"]):
        print(f"[skip] 图片缺失或损坏: {rec['image_path']}")
        continue
    buffer.append((i, rec))
    if len(buffer) >= bs:
        flush_buffer()

flush_buffer()

print(f"[done] {args.bench} index={args.index}: 累计 {len(scan_done(outdir))} 条 / 切片 {len(sub)} 条")
