"""
ViSpec Stage 2 长回复预生成(在线训练用)
==============================================================================
为什么需要这一步:
  ViSpec Stage 2 的训练标签来自「目标模型对图片 generate 的长回复」。generate 是
  自回归生成 1000+ token,极慢,无法放进训练循环。因此离线预生成长回复。

与离线 ge_data_all_qwen_pretrain_gen.py 的区别:
  - 离线脚本:generate 后立刻提取 hidden_state/inputs_embeds 一起存盘(每条十几 MB)。
  - 本脚本:  只存「完整 token 序列 full_ids + prompt_len + 图片路径」(每条几 KB),
             hidden state 留到在线训练循环里现场 forward 算。

输出:每条一个 .pt 文件,字段 {full_ids, prompt_len, image_file},
      供 vispec/train/online_dataset.py 的 OnlineMMDataset 读取。

用法:
  python -m vispec.ge_data.gen_stage2_responses \
    --outdir data/train/gen_mm_online --start 0 --end 20 \
    --model ./model/Qwen2.5-VL-7B-Instruct --gpu_index 0
==============================================================================
"""

import argparse
import json
import os

parser = argparse.ArgumentParser(description="Stage 2 long-response pre-generation")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=100)
parser.add_argument("--index", type=int, default=0)
parser.add_argument("--gpu_index", type=int, nargs="+", default=[0])
parser.add_argument("--outdir", type=str, default="data/train/gen_mm_online")
parser.add_argument("--max_new_tokens", type=int, default=1024)
parser.add_argument("--model", type=str, default="./model/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument(
    "--data-path", type=str, default="data/train/LLaVA-Pretrain/"
)
parser.add_argument(
    "--batch-size", type=int, default=1,
    help="单次 generate 同时处理多少张图。bs>1 走 left padding 批量推理,显著提速;"
         "显存占用约 batch_size 倍,Qwen2.5-VL-7B 在 80G 卡上建议 batch=4(max_pixels 默认),"
         "batch=8(max_pixels 砍半)。",
)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]

import torch
from datasets import Dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

bigname = args.model


# ---------------------------------------------------------------------------
# 数据集构造:复用离线脚本完全相同的 prompt 拼装逻辑(保证训练分布一致)
# ---------------------------------------------------------------------------
def _load_or_build_present_index(ds_shuffled, data_path):
    """LLaVA-Pretrain 本地图片只有约 27%,默认会让 [start,end) 里 ~3/4 样本被 skip。
    这里先扫一遍 shuffle 后的全表,把「本地真实存在图片」的索引列出并 cache,
    后续 args.start/end 直接落在「有效索引空间」上,不用再开 4× 缓冲。
    cache 落盘到 data_path/_present_indices.shuffled42.json,首次约 1-2 分钟,后续秒级。
    """
    cache = os.path.join(data_path, "_present_indices.shuffled42.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)

    print(f"[present-index] 首次扫描本地图片(shuffle seed=42),写到 {cache} ...")
    present = []
    images = ds_shuffled["image"]
    for i, rel in enumerate(tqdm(images, desc="stat images")):
        if os.path.exists(os.path.join(data_path, rel)):
            present.append(i)
    with open(cache, "w") as f:
        json.dump(present, f)
    print(f"[present-index] 完成:本地有效样本 {len(present)} / 总 {len(images)} "
          f"({100.0*len(present)/max(1,len(images)):.1f}%)")
    return present


def build_dataset_rank(processor, path):
    with open(os.path.join(path, "blip_laion_cc_sbu_558k.json")) as f:
        ds = json.load(f)
    ds = Dataset.from_list(ds)
    ds = ds.shuffle(seed=42)
    # 把 [start,end) 限制在「本地有图」的索引空间上,使 END-START 直接等于有效产出条数
    present = _load_or_build_present_index(ds, path)
    sub = present[args.start: args.end]
    if len(sub) == 0:
        raise ValueError(
            f"[start,end)=[{args.start},{args.end}) 落在 present 索引空间(共 {len(present)} 条)外"
        )
    ds1 = ds.select(sub)
    original_columns1 = ds1.column_names

    def preprocess_function(examples):
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
                    },
                ],
            }
        ]
        for conv in examples["conversations"]:
            if conv["from"] == "human":
                assert conv["value"].endswith("\n<image>") or conv[
                    "value"
                ].startswith("<image>\n")
                conversation.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": conv["value"].strip().strip("<image>").strip(),
                            },
                            {"type": "image"},
                            {
                                "type": "text",
                                "text": "Please answer with at least 1000 words.",
                            },
                        ],
                    }
                )
            elif conv["from"] == "gpt":
                pass
            else:
                raise ValueError("Unknown role")

        prompt_input = processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        return {
            "image_files": [os.path.join(path, examples["image"])],
            "text": prompt_input,
        }

    ds1 = ds1.map(
        preprocess_function,
        batched=False,
        num_proc=1,
        remove_columns=original_columns1,
        load_from_cache_file=False,
    )
    return ds1


min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    bigname, use_fast=True, min_pixels=min_pixels, max_pixels=max_pixels
)
# batch generate 必须 left padding(右 padding 会让短样本生成时把 pad 当真 token,输出乱)
processor.tokenizer.padding_side = "left"
ds = build_dataset_rank(processor, args.data_path)
print(ds)

bigmodel = AutoModelForImageTextToText.from_pretrained(
    bigname, device_map="auto", torch_dtype="auto"
)
bigmodel.eval()


@torch.no_grad()
def gen_batch(batch_data):
    """对一个 batch 的样本(每个 dict 含 image_files / text)做 generate。
    返回 list[{full_ids, prompt_len, image_file}],与单条 gen_one 同结构,顺序与输入一致。
    """
    images = [Image.open(d["image_files"][0]) for d in batch_data]
    texts = [d["text"] for d in batch_data]
    image_files = [d["image_files"][0] for d in batch_data]

    # processor 支持 batch:images 是 list[List[PIL]] 时按图片 list 聚合,文本 list 同步聚合
    # Qwen2.5-VL 一张图变成多个 <|image_pad|> token,batch 时不同图 grid 大小不同,但
    # processor 内部会处理:image_grid_thw 是 [batch, 3] 形状,文本里的 image_pad 数也对齐展开
    inputs = processor(
        images=images,
        text=texts,
        padding=True,             # 左 padding 到 batch 内最长
        return_tensors="pt",
    ).to(bigmodel.device)

    # 每条样本去掉 pad 后的真实 prompt 长度(用 attention_mask 的 1 数)
    prompt_lens = inputs["attention_mask"].sum(dim=-1).tolist()  # list[int],每条
    pad_lens = (inputs["input_ids"].shape[-1] - inputs["attention_mask"].sum(dim=-1)).tolist()

    outs = bigmodel.generate(
        **inputs,
        do_sample=args.temperature != 0,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    # outs: [batch, max_prompt_len + gen_len_max] 含左侧 pad
    # 每条要剥离前面的 pad,保留 [真实 prompt + 真实回复] 直到 EOS
    eos_id = processor.tokenizer.eos_token_id
    pad_id = processor.tokenizer.pad_token_id
    results = []
    for i, out_ids in enumerate(outs):
        # 1. 剥离左侧 pad(用原始 input_ids 的 pad 数定位)
        ids = out_ids[pad_lens[i]:].cpu()
        # 2. EOS 之后的内容裁掉(EOS 自身保留——与原单条版一致,原版 generate 不显式裁)
        # 找第一个 prompt 之后的 EOS
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


outdir = os.path.join(args.outdir, str(args.index))
os.makedirs(outdir, exist_ok=True)


def writedata(name, data_point, idx):
    """原子写:先写 .tmp 再 rename,避免中断留下损坏 .pt。"""
    os.makedirs(name, exist_ok=True)
    final = f"{name}/data_{idx}.pt"
    tmp = final + ".tmp"
    torch.save(data_point, tmp)
    os.replace(tmp, final)


def scan_done(name):
    """扫描 outdir 里已存在的 data_<i>.pt,返回 i 集合,用于断点续跑。"""
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
            # 上次中断留下的半成品,清掉
            try:
                os.remove(os.path.join(name, fn))
            except OSError:
                pass
    return done


# ---------------------------------------------------------------------------
# 主循环:按 batch_size 攒一批后调用 gen_batch
#   - 缺图样本(FileNotFoundError)逐条尝试,从 batch 里剔除继续,不影响其他样本
#   - 文件名 = present 切片内的位置 i,可断点续跑(已存在的 i 直接 skip)
# ---------------------------------------------------------------------------
def open_or_none(path):
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except (FileNotFoundError, OSError) as e:
        return False


bs = args.batch_size
buffer = []          # 待 generate 的 batch,元素为 (i, data) 元组
done = scan_done(outdir)
if done:
    print(f"[resume] 检测到 outdir 已有 {len(done)} 条产出,本次将自动跳过")
pending_iter = iter(tqdm(ds))


def flush_buffer():
    if not buffer:
        return
    items = [b[1] for b in buffer]
    indices = [b[0] for b in buffer]
    try:
        recs = gen_batch(items)
        idx_for_recs = indices
    except Exception as e:
        print(f"[err] batch generate 失败,回退到逐条重试: {e}")
        recs = []
        idx_for_recs = []
        for ii, d in zip(indices, items):
            try:
                recs.extend(gen_batch([d]))
                idx_for_recs.append(ii)
            except Exception as ee:
                print(f"[skip] 单条失败 {d['image_files'][0]}: {ee}")
    for ii, rec in zip(idx_for_recs, recs):
        writedata(outdir, rec, ii)
    buffer.clear()


for i, data in enumerate(pending_iter):
    if i in done:
        continue
    # 提前剔除缺图样本,避免拖累整 batch
    if not open_or_none(data["image_files"][0]):
        print(f"[skip] 图片缺失或损坏,跳过 dataset_index={i}: {data['image_files'][0]}")
        continue
    buffer.append((i, data))
    if len(buffer) >= bs:
        flush_buffer()

flush_buffer()  # 处理收尾余量
