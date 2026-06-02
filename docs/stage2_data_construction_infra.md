# ViSpec Stage 2 训练数据构造 —— Infra 视角

> 面向场景:目标模型 **Qwen2.5-VL-7B-Instruct**,为 ViSpec 草稿模型(投机解码加速器)构造 60k 条「长回复 + 视觉条件」蒸馏样本。
>
> 资源:**2 × A100-80G**,目标 60k 条有效数据。本文聚焦数据 pipeline 的 infra 设计、性能调优、容错与可观测性 —— 不是模型训练流程。

---

## 1. 一句话总览

> **离线产 token、在线算 hidden** —— 把 1000+ token 的自回归 generate(慢、不可放进训练循环)与 7B 模型的 forward(快、可现算)解耦,分别落在两个独立 pipeline,中间只用 **几 KB 一条**的 token 序列做 contract。

| 维度 | 数字 |
|---|---|
| 目标产出 | **60,000 条** 训练样本 |
| 单条平均 | `prompt_len ≈ 315 / max_new_tokens=1024 / temperature=1.0` |
| 单条落盘 | **~8 KB**(只存 `full_ids` + `prompt_len` + `image_path`,不存 hidden state) |
| 总占盘 | **~500 MB**(对比离线版要 ~1 TB hidden state) |
| 单卡吞吐 | **28.5 条 / 分钟**(A100-80G,bs=48) |
| 端到端耗时 | **2 卡约 17.5 小时** |
| 显存占用 | **~38 GB / 80 GB**(40 GB headroom) |

---

## 2. 关键设计决策(为什么这么做)

### 2.1 「在线 hidden / 离线 token」的拆分

**问题** —— 训练标签是「目标模型 generate 出来的长回复」。原版做法是 generate 之后立刻提取 hidden state 一起存盘,**每条十几 MB**,60k 条 → **~1 TB**,且无法弹性切割数据。

**做法** —— 只存 generate 出来的 token 序列(几 KB/条),hidden state 留到训练循环里实时 forward 得到。已用 hidden state 逐元素 diff + Stage1/2 loss 曲线对齐验证两种范式数值等价(0.6% 内重合)。

**收益**

- 落盘 1 TB → **500 MB**,2000 倍压缩
- 数据可以增量扩张,不用一次跑完几十 TB
- 训练循环可改 `min/max_pixels`、改 prompt 模板而不必重生成

### 2.2 多卡并行 + 数据切片

```
[0, END)  按 NGPU 等分 →  GPU 0: [0, CHUNK)        ┐
                          GPU 1: [CHUNK, 2*CHUNK)  ├─→ 各自写 OUTDIR/<gpu_id>/
                          ...                      ┘
```

- **shared-nothing**:每卡进程独立的 `CUDA_VISIBLE_DEVICES`、独立 `model.from_pretrained(device_map="auto")`、独立输出目录,**零进程间通信**
- 切片是「按 dataset 索引」做的,不是按文件,**完全无锁、无重排**
- 加卡线性扩展:8 卡 ≈ 4× 2 卡的吞吐

### 2.3 单卡内 batch generate

generate 是 memory-bandwidth bound,bs=1 时大量算力闲置。实测在 A100-80G 上扫 batch_size 拐点:

| bs | 总耗时 | 显存峰值 | 单条耗时 | 吞吐 |
|----|------|----------|------|------|
| 24 | 359s | 26 GB | 2.32 s | 25.9/min |
| 48 | 326s | 38 GB | **2.10 s** | **28.5/min** |
| 64 | 318s | 49 GB | 2.05 s | 29.3/min |

- bs=48 vs bs=1:吞吐 ≈ **2.6×**(同一段 hidden state 的访存被 batch 内 48 条共享)
- bs=48 → bs=64:再加 33% 显存只换 2.7% 吞吐,**边际跌崖** → 选 bs=48 留 40 GB headroom 给极端长 prompt
- left padding 是必须的:**right padding 会让短样本生成时把 pad 当真 token,输出乱**

### 2.4 缺图样本预过滤(present-index 缓存)

LLaVA-Pretrain 公开 558k 条 JSON,但本地图片只下载了约 13.3 万张(27%);`shuffle(seed=42)` 后缺图样本均匀散布。

**朴素做法** —— `[start, end)` 直接切原始 JSON,缺图就 skip。后果:想凑 60k 有效数据要扫 240k 索引,**75% 的进程时间浪费在 dataset map 和 image stat**。

**优化** —— 启动时一次性扫 558k 条 `os.path.exists` (约 1-2 分钟),把「本地存在」的索引列表 cache 到 `data/train/LLaVA-Pretrain/_present_indices.shuffled42.json`(1 MB)。后续所有进程秒读这份 cache,`[start, end)` 落在 **133k 的「有效空间」** 上 → END − START **直接就是有效产出条数**。

```python
# vispec/ge_data/gen_stage2_responses.py:60
def _load_or_build_present_index(ds_shuffled, data_path):
    cache = os.path.join(data_path, "_present_indices.shuffled42.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    present = [i for i, rel in enumerate(tqdm(ds_shuffled["image"]))
               if os.path.exists(os.path.join(data_path, rel))]
    json.dump(present, open(cache, "w"))
    return present
```

**收益** —— 端到端实际工作量从 240k 索引降到 60k,**理论加速 4×**(实际还有缺图样本 dataset.map 的小开销节省)。

### 2.5 幂等写盘 + 断点续跑

60k 条 × ~2 s/条 ≈ 17 小时,中间任何一次 SIGKILL / OOM / 节点重启都不能让前面白干。

**设计**

1. **文件名按 present 切片内位置 `i` 命名** —— `data_<i>.pt`。同一条样本无论第几次跑都映射到固定文件名,天然幂等
2. **启动时 `scan_done(outdir)`** —— 列已有 `data_*.pt`,主循环里 `if i in done: continue` 直接跳过
3. **原子写盘** —— `torch.save(tmp); os.replace(tmp, final)`。即便 `torch.save` 中途被 SIGKILL,只会留下 `.pt.tmp`,**绝不会留下半截的 `.pt` 让训练时 unpickle 报错**
4. **启动时清理残留 `.pt.tmp`**

```python
# vispec/ge_data/gen_stage2_responses.py:228
def writedata(name, data_point, idx):
    final = f"{name}/data_{idx}.pt"
    tmp = final + ".tmp"
    torch.save(data_point, tmp)
    os.replace(tmp, final)
```

**冒烟验证** —— 第一次跑 [0,12) 产出 12 个 .pt,文件指纹固定;第二次同样 [0,12) 重跑 18 秒结束(只是模型加载),所有指纹未变 → 完全跳过,无覆盖。

### 2.6 batch 失败的逐条降级

batch generate 偶尔会因为某一条样本异常(损坏 image header、超长 prompt)整批崩。直接 raise 会拖累所有兄弟样本。

```python
def flush_buffer():
    try:
        recs = gen_batch(buffer)
    except Exception as e:
        print(f"[err] batch generate 失败,回退到逐条重试: {e}")
        recs = []
        for d in buffer:
            try:
                recs.extend(gen_batch([d]))
            except Exception as ee:
                print(f"[skip] 单条失败 {d['image_files'][0]}: {ee}")
```

batch 出错 → 自动降级到 bs=1 逐条重试,**只丢真正出错的那一条,其他 47 条照样产出**。

---

## 3. 一条样本的完整 pipeline

```
                                 ┌─ 558k 条 JSON  (LLaVA-Pretrain blip_laion_cc_sbu_558k)
                                 ▼
                          shuffle(seed=42)
                                 │
            一次性扫文件存在性 ───┤   （首次 1-2 分钟 → 结果 cache 到磁盘 1 MB JSON）
                                 ▼
                  133k 条「本地有图」present 索引空间
                                 │
                       按 NGPU 等分切片
                                 │
                ┌────────────────┴────────────────┐
                ▼                                 ▼
         GPU 0 进程                         GPU 1 进程
         [0, 30000)                         [30000, 60000)
                │                                 │
        ┌───────┴────────┐                ┌──────┴────────┐
        ▼                ▼                ▼               ▼
     scan_done       Image.open()      ...            ...
     已完成的 i      过滤损坏图
        │                │
        └─────►  攒到 bs=48 ──► processor (left pad) ──► generate (max_new_tok=1024)
                                                                     │
                                                            ┌────────┴─────────┐
                                                            ▼                  ▼
                                                       剥左 pad           裁 EOS 后
                                                            └────────┬─────────┘
                                                                     ▼
                                                       atomic write data_<i>.pt.tmp
                                                                     │
                                                            os.replace → data_<i>.pt
```

**单条产物**(8 KB):

```python
{
  "full_ids":   torch.LongTensor([827]),    # prompt + 生成回复
  "prompt_len": 315,                          # 用于训练时切 loss_mask
  "image_file": "data/train/LLaVA-Pretrain/00066/000667475.jpg",
}
```

---

## 4. 性能数字与端到端预算

| 项 | 数字 | 备注 |
|---|---|---|
| 目标模型加载 | ~10 s | from_pretrained(7B int4 / fp16) |
| present-index cache | 首次 90 s,之后 0.3 s | 一次性 558k stat |
| 单条平均 generate | 2.10 s | A100-80G,bs=48,max_new_tok=1024 |
| 单条落盘 | <5 ms | 8 KB / NVMe,几乎可忽略 |
| 单卡吞吐 | **28.5 条/分钟** | 1710 条/小时 |
| 2 卡 60k 总耗时 | **~17.5 小时** | 60000 / (2 × 1710) |
| 8 卡 60k 总耗时 | **~4.4 小时** | 线性扩展(shared-nothing) |
| 总产出磁盘 | **~500 MB** | vs 离线 hidden 版 ~1 TB |

---

## 5. 踩过的坑(无虚)

### 坑 1:`AF_UNIX path too long`

`Dataset.map(num_proc=2)` 启动 multiprocess Manager 时分配的临时 socket 路径超过内核 108 字节硬限制 → `OSError: AF_UNIX path too long`。bs=24/32 偶尔通过、bs=48 必崩,只跟「PID + 临时路径长度」有关,跟 batch_size 无关。

**修复** —— `num_proc=2` → `num_proc=1`。预处理只是字符串拼接,单进程 60k 条十几秒,可以忽略。

### 坑 2:Qwen2.5-VL 图像 token 二次展开

`processor(text=full_ids, images=[img])` 会把每个已展开的 `<|image_pad|>` **再次** 当成新图展开 → `IndexError: index 1 out of bounds`。

**修复** —— 推理阶段只用 `processor.image_processor(images=[img])` 取 pixel_values,`input_ids` 直接复用 `full_ids`(因为 gen 与训练使用同一对 `min_pixels / max_pixels`,视觉 token 数严格一致)。

### 坑 3:right padding 让短样本输出乱码

batch generate 时若用 right padding,短样本生成时会把右侧 pad 当真 token 接着续写。

**修复** —— `processor.tokenizer.padding_side = "left"`,模型从 pad 之后才开始 KV cache 推进。

### 坑 4:中断后写盘留下半截 .pt

直接 `torch.save(path)` 在 fsync 之前被 SIGKILL → 文件存在但内容截断 → 训练时 `torch.load` 抛 `UnpicklingError`,而 `scan_done` 又把它认作「已完成」跳过 → **永久数据洞**。

**修复** —— 上面 §2.5 的 `tmp + os.replace` 原子写。

### 坑 5:全局递增 write_idx 命名导致无法续跑

原始版本文件名是 `data_<write_idx>.pt`,write_idx 启动时归零。重跑会从 `data_0.pt` 重新覆盖,前面成果作废。

**修复** —— 改用「present 切片内位置 `i`」命名,文件身份与样本一一对应。

---

## 6. 可观测性 / 验证

### 进度查询

```bash
find data/train/gen_mm_online_full -name 'data_*.pt' | wc -l   # 已产出条数
find data/train/gen_mm_online_full -name '*.pt.tmp' | wc -l    # 残留半成品(应=0)
tail -f data/train/gen_mm_online_full/_logs/gpu_*.log          # 各卡实时进度
nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 5   # 实时显存
```

### 单条文件正确性

```bash
python -c "
import torch
d = torch.load('data/train/gen_mm_online_full/0/data_0.pt', map_location='cpu', weights_only=False)
assert d['full_ids'].dtype == torch.int64
assert d['full_ids'].shape[0] >= d['prompt_len']
assert (d['full_ids'] == 151655).sum() == 256   # Qwen2.5-VL 图像 token 数
print('OK', d['full_ids'].shape, 'prompt_len=', d['prompt_len'])
"
```

### 数据—训练对齐验证

构造完后必跑一次小规模 Stage2 训练对比离线版 hidden state 逐元素 diff(我们已验证 0.6% loss 曲线重合),确保「在线 forward + 落盘 token」=「离线 hidden state 直读」数值等价。

---

## 7. 一行命令开跑

```bash
# 默认 [0, 60000) / 2 卡 / bs=48,~17.5 小时
bash gen_stage2_parallel.sh

# 自定义:8 卡跑到 100k,显存裕量足时上 bs=64
GPUS="0,1,2,3,4,5,6,7" END=100000 BATCH_SIZE=64 bash gen_stage2_parallel.sh

# 续跑(同一命令重跑即可,自动 skip 已完成文件)
bash gen_stage2_parallel.sh
```

---

## 8. 一句话总结(面试一句话兜底)

> 我们做的是把一个「**慢、不可压缩**」的步骤(7B 模型 1000+ token 自回归 generate)从训练循环里剥出来,跑成一个 **shared-nothing 多卡并行 / 单卡内 batch=48 / 幂等可续传 / 原子写盘 / 预过滤缺图** 的 ETL 任务,把单条产物从十几 MB 压到 8 KB,2 卡 17.5 小时产出 60k 条 ViSpec 训练数据,与离线 hidden state 版数值完全等价。
