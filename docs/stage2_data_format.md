# ViSpec Stage 2 数据格式说明

> 本文聚焦 **Stage 2(多模态)训练数据的格式**:在线 `.pt` 与离线 `.ckpt` 两种,
> 各字段含义、张量形状,以及当前各目录的可用数量。
> 训练流程见 [`stage2_pipeline.md`](./stage2_pipeline.md)。

---

## 0. 为什么 Stage 2 要重新生成数据,而不用原始数据集的标签?

这是 Stage 2 数据准备最核心的设计决定。**原始 LLaVA-Pretrain 的 `gpt` 答案被完全丢弃,
改用目标模型 Qwen2.5-VL-7B 现场 generate 的长回复作为训练标签。** 原因有四:

### ① 训练目标根本不同:不是「学答案」,而是「学目标模型的隐状态分布」

投机解码(speculative decoding)的草稿模型,任务是**模仿目标模型自己**:在「看图 + 已生成上文」
的条件下,预测目标模型下一步会输出什么。所以监督信号必须来自**目标模型本身的 forward 输出**
(末层 hidden → lm_head → 概率分布),做的是**知识蒸馏**,而不是去拟合人类/数据集给的标准答案。

- 用原始标签:草稿模型学的是「数据集认为的正确答案」——但推理时目标模型未必这么说,草稿一旦
  和目标模型不一致,投机解码的 token 就会被拒绝,加速失效。
- 用目标模型生成:草稿模型学的是「目标模型实际会怎么说」——这才是投机解码命中率的来源。

> 一句话:**草稿模型要拟合的是目标模型的行为,而非真实世界的答案。** 标签只能由目标模型产出。

### ② 原始标签太短,撑不起「长序列」训练

LLaVA-Pretrain 的 `gpt` 字段本质是**一句图片 caption**。实测前 2000 条统计:

| 指标 | 原始 `gpt` 答案词数 |
|---|---|
| 均值 | **9.9 词** |
| 中位数 | 9 词 |
| 最大 | 21 词 |

示例(原汁原味):
```
[gpt] select luxury furniture 3 - inch gel memory foam mattress topper
[gpt] a grey watch with an army style strap
[gpt] a dragon kite flying in the blue sky stock images
```

而 Stage 2 的 prompt **刻意追加** `"Please answer with at least 1000 words."`,目标模型会生成
**几百~上千 token** 的长回复(本项目示例样本回复段 512 token)。原因:

- 投机解码加速只在**较长的自回归生成**里才有意义,需要足够长的「回复段」喂给草稿模型学习;
- 只有 9 词的标签,loss_mask 里能算 loss 的位置寥寥无几,训练信号严重不足。

#### 这句 "at least 1000 words" 加在哪、为什么这么做

**位置**:`vispec/ge_data/gen_stage2_responses.py:91`(在线),拼成 user 消息的第三段,
紧跟在「人类问题 + 图片」之后:
```python
{
  "role": "user",
  "content": [
    {"type": "text",  "text": conv["value"]...},                   # ① 原始人类问题
    {"type": "image"},                                              # ② 图片
    {"type": "text",  "text": "Please answer with at least 1000 words."},  # ③ ← 强制长回复
  ],
}
```
> 离线生成脚本 `ge_data_all_qwen_pretrain_gen.py:83`、`ge_data_all_llava_pretrain_gen.py:83`
> 完全相同,三处一致。

**为什么这么做**(它和 `--max_new_tokens=1024` 是一对配合):

1. **原始问题本身诱导出的回复太短**。LLaVA-Pretrain 的人类问题都是
   "Render a clear and concise summary" / "Write a terse but informative summary" 这类**要求简洁**的指令,
   目标模型不加约束就只吐一句 caption 即停(原始 `gpt` 标签均值仅 9.9 词)。
   这句话把任务**从「概括」改写成「长篇展开」**,强行拉长输出。

2. **"1000 words" 是手段,不是硬指标**。它只是一个强提示让模型尽量长写,**真正的长度上限由
   `max_new_tokens=1024`(`MAXTOK`)截断封顶**,模型并不需要真凑满 1000 词。
   本项目示例样本回复段只有 512 token,也没到 1000 词 —— 提示词负责「把短答案倾向掰成长答案」,
   `max_new_tokens` 负责「封顶 + 控制单条生成耗时/显存」。

3. **长度直接决定训练信号密度**。草稿模型只在回复段(`loss_mask=1`)上算 loss,
   回复越长 → 每条样本可学习位置越多 → 信号越密,且能覆盖「长序列自回归」这一推理时的真实场景。

### ③ 分布一致性:训练数据必须来自目标模型自己的采样分布

草稿模型部署时面对的输入,是目标模型**自己逐 token 生成**出来的序列。若用原始标签训练,会产生
**exposure bias / 分布漂移**:训练时见的是数据集文本,推理时见的是目标模型自回归输出,两者分布不同。
用目标模型 `generate`(`temperature=1.0` 采样)产出标签,保证训练分布 = 推理分布。

### ④ 还需要原始标签里没有的「中间状态」

蒸馏需要每个位置的 `hidden_state` / `inputs_embeds`,这些只有让目标模型对**完整序列**跑一遍
forward 才有(在线实时算,离线提前存)。原始数据集只有纯文本答案,根本不含这些张量。

### 小结

| | 用原始 `gpt` 标签 | 用目标模型 generate(本项目) |
|---|---|---|
| 监督信号 | 数据集标准答案(文本) | 目标模型隐状态分布(蒸馏) |
| 序列长度 | ~10 词,太短 | 几百~上千 token |
| 分布 | 与推理时不一致(漂移) | 与推理时一致(同一采样分布) |
| 含中间张量 | 否 | 是(hidden_state/embeds) |
| 投机解码命中 | 低 | 高 |

> 所以 Stage 2 必须先跑 `gen_stage2_parallel.sh`,用目标模型重新生成数据。
> **原始数据集只贡献「图片 + 那句人类问题」,`gpt` 答案一律弃用。**

具体被丢弃的位置见 `gen_stage2_responses.py` 的 `preprocess_function`:
```python
for conv in examples["conversations"]:
    if conv["from"] == "human":
        ...  # 用问题 + <image> + "Please answer with at least 1000 words."
    elif conv["from"] == "gpt":
        pass   # ← 原始答案直接 pass 丢弃
```

---

## 1. 两种数据格式对比

Stage 2 有「在线 / 离线」两套训练路径,落盘数据格式完全不同:

| | **在线 `.pt`** | **离线 `.ckpt`** |
|---|---|---|
| 产出脚本 | `gen_stage2_parallel.sh` → `gen_stage2_responses.py` | `gen_stage2_data.sh` → `allocation_qwen_pretrain_gen.py` |
| 训练脚本 | `train_stage2_online.sh` → `main_mtp_online.py` | `train_stage2.sh` → `main_mtp.py` |
| 存的内容 | **只存 token**(几个轻量字段) | **预存目标模型算好的 hidden_state / inputs_embeds** |
| 单条大小 | **~8 KB** | **~11 MB**(约 1400 倍) |
| hidden state | 训练循环里**实时 forward** 算 | 提前算好,训练时直接 load |
| 目标模型 | 训练时常驻显存(冻结) | 训练时不加载 |
| 目录 | `data/train/gen_mm_online[_full]/` | `data/train/gen_mm_test/` |

---

## 2. 在线格式 `.pt`(推荐)

### 路径布局
```
data/train/gen_mm_online_full/   # 正式数据(多卡并行,每卡一个子目录)
├── 0/                           # GPU 0 分片
│   ├── data_0.pt
│   ├── data_1.pt
│   └── ...
├── 1/                           # GPU 1 分片
└── ...
```
> 训练时 `list_response_files()` 递归收集所有子目录下的 `*.pt`,无需关心分片结构。

### 单条 `.pt` 字段(`gen_stage2_responses.py` 产出)

```python
{
  "full_ids":   torch.Size([seq]),  # int64,完整 token 序列 = prompt + 目标模型生成的长回复
  "prompt_len": int,                 # prompt 段长度(含已展开的图像 token),用于切分 loss_mask
  "image_file": str,                 # 图片路径,训练时再 Image.open 取视觉特征
}
```

### 真实样本举例(`gen_mm_online/0/data_0.pt`)

```python
{
  "full_ids":   torch.Size([827]),   # int64
  "prompt_len": 315,
  "image_file": "data/train/LLaVA-Pretrain/00066/000667475.jpg",
}
```

序列结构:
```
位置:  0 ........................... 314 | 315 ........................ 826
       └────────── prompt(315) ──────────┘└─────── 长回复(512)────────┘
       ┌─────────────┬───────────────────┐
       │ 文本 prompt  │  图像 token(256 个) │   ← <image> 被 processor 展开成 256 个
       │  (~59 个)    │  id 全是 151655     │      <|image_pad|>(grid 32×32÷merge² = 256)
       └─────────────┴───────────────────┘
```

### 训练时现场派生(`OnlineMMDataset.__getitem__`)
`.pt` 本身不存 mask,Dataset 读入后实时算出:
```python
loss_mask  = [0]*prompt_len + [1]*(seq - prompt_len)   # prompt 段不算 loss,回复段算
image_mask = (full_ids == 151655)                       # 标记图像 token 位置(供 ImgAdaptor 定位)
```

> ⚠️ **为什么不存 hidden state**:hidden state 只是目标模型一次 forward,很快;而 generate 长回复极慢,
> 所以只把 generate 的结果(token)离线存下,hidden state 放训练循环实时算 —— 省下离线几十 GB 的存储。

---

## 3. 离线格式 `.ckpt`(对照)

### 路径布局
```
data/train/gen_mm_test/qwen_pretrain_gen_0_<N>_mufp16/
├── data_0.ckpt
├── data_1.ckpt
└── ...
```

### 单条 `.ckpt` 字段
离线在 generate 之后**立刻提取并存盘**目标模型的中间结果,训练时直接 load、不碰目标模型。核心字段:

| 字段 | 含义 |
|---|---|
| `hidden_state` | 目标模型末层隐状态(已算好) |
| `inputs_embeds` | embedding 层输出(已算好) |
| `input_ids` | 完整 token 序列 |
| `loss_mask` | prompt=0 / 回复=1 |
| `image_mask` | 图像 token 位置 |
| (视觉相关 pixel/grid 等) | |

> 因为把 `hidden_state` / `inputs_embeds`(每个 `[seq, 3584]` float)一起存,单条才高达 ~11 MB。

---

## 4. 当前各目录可用数量(截至 2026-06-01)

| 目录 | 用途 | 格式 | 可用条数 | 占用 |
|---|---|---|---|---|
| `data/train/gen_mm_online_full/` | Stage2 在线-**正式** | `.pt` | **0 条** ❌(待生成) | 空 |
| `data/train/gen_mm_online/` | Stage2 在线-小验证 | `.pt` | **20 条** | 244 K |
| `data/train/gen_mm_test/qwen_pretrain_gen_0_20_mufp16/` | Stage2 离线-小验证 | `.ckpt` | **20 条** | 303 M |

### 上游原始数据(生成 Stage2 数据的素材)
| 数据 | 规模 |
|---|---|
| `LLaVA-Pretrain/blip_laion_cc_sbu_558k.json` | 558128 条标注 |
| `LLaVA-Pretrain/` 本地已解压图片 | **133267 张(约 24%)** |

> ⚠️ 本地图片只有约 24%,缺图样本生成时自动 skip。
> 因此**有效产出 ≈ 扫描索引范围 × 24%**(如默认扫 `[0,160000)` → 约 3.8 万条有效)。

---

## 5. 如何生成正式在线数据

```bash
# 默认:5 卡并行,扫描原始索引 [0,160000),约产 3.8 万有效条 → gen_mm_online_full/
bash gen_stage2_parallel.sh

# 自定义范围 / GPU
START=0 END=160000 GPUS="0,1,2,3,4" MAXTOK=1024 bash gen_stage2_parallel.sh
```
生成后即可训练:
```bash
bash train_stage2_online.sh
# 或 DATAPATH=data/train/gen_mm_online_full bash train_stage2_online.sh
```
</content>
