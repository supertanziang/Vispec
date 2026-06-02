# ViSpec Stage 2 在线训练流程详解(含完整示例)

> 目标:训练一个**轻量草稿模型**(1 层 decoder),让它在「看图 + 已生成上文」的条件下,
> 模仿目标模型 **Qwen2.5-VL-7B** 下一个 token 的隐状态分布,从而在推理时做投机解码(speculative decoding)加速。
>
> 本文用一条**真实样本**贯穿始终,逐步标注每一处的**输入 / 输出张量形状**,并重点拆解**损失函数**。

---

## 0. 关键维度速查(本项目实际值)

| 名称 | 值 | 来源 |
|---|---|---|
| 目标模型 | Qwen2.5-VL-7B-Instruct | `model/Qwen2.5-VL-7B-Instruct` |
| `hidden_size` | **3584** | 目标模型 & 草稿模型一致 |
| `vocab_size` | **152064** | lm_head 输出维度 |
| `num_attention_heads` | 28 | |
| 草稿模型层数 `num_hidden_layers` | **1** | `qwen2.5_vl_7B_config.json`,刻意做小 |
| `image_token_id` (`<|image_pad|>`) | **151655** | config.json |
| `num_q`(ImgAdaptor 视觉压缩后的 query 数) | **2** | `--num-q=2` |
| `mtp_steps`(Multi-Token Prediction 步数) | **1** | `--mtp-steps=1` |
| 损失系数 | `10 * ploss + 0.1 * rloss` | `compute_loss()` |

---

## 1. 整体两步流程

在线训练 = **先离线预生成长回复 token,再在线实时算 hidden state 训练**。

```
┌─────────────────────────────────────────────────────────────────────┐
│ 第 1 步:预生成长回复(gen_stage2_parallel.sh)                          │
│   目标模型对 [图片+问题] generate 长回复 → 只存 token(几 KB/条)         │
│   产物:每条一个 .pt = { full_ids, prompt_len, image_file }            │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 第 2 步:在线训练(train_stage2_online.sh → main_mtp_online.py)         │
│   循环里:                                                              │
│     ① 目标模型对 (图片 + full_ids) 实时 forward → 末层 hidden + embedding │
│     ② 右移构造 target(teacher 信号)                                    │
│     ③ 草稿模型(ViSpec: ImgAdaptor + 1 层 decoder)forward → predict      │
│     ④ MTP 多步展开 → compute_loss(蒸馏 loss)→ 只更新草稿模型             │
└─────────────────────────────────────────────────────────────────────┘
```

**为什么分两步**:训练标签是「目标模型 generate 的长回复」。generate 是自回归生成 1000+ token,极慢,不能放进训练循环;而 hidden state 只是一次 forward,很快,所以放在循环里实时算(省下离线存几十 GB 的 hidden state)。

---

## 2. 贯穿全文的示例样本

我们取 `data/train/gen_mm_online/0/data_0.pt`(真实文件):

```python
{
  "full_ids":   torch.Size([827]),   # int64,完整 token 序列
  "prompt_len": 315,                  # prompt 部分长度(含图像 token)
  "image_file": "data/train/LLaVA-Pretrain/00066/000667475.jpg",
}
```

这条样本的结构(`full_ids` 长度 827):

```
位置:  0 ........................... 314 | 315 ........................ 826
       └────────── prompt(315) ──────────┘└─────── 长回复(512)────────┘
       ┌─────────────┬───────────────────┐
       │ 文本 prompt  │  图像 token(256 个) │   ← prompt 内部:文本 + 已展开的 <|image_pad|>
       │  (~59 个)    │  id 全是 151655     │
       └─────────────┴───────────────────┘
```

- prompt = system prompt + 用户问题 + `<image>` + `"Please answer with at least 1000 words."` 经 chat template 拼接。
- 其中 `<image>` 被 processor **展开成 256 个** `<|image_pad|>`(grid 32×32 ÷ merge_size² = 1024÷4 = 256)。
- 长回复 512 个 token 是目标模型 `generate` 出来的(`max_new_tokens=1024`,本例实际生成 512)。

---

## 3. 第 1 步:预生成(`gen_stage2_responses.py`)

### 输入
- 原始数据集 `blip_laion_cc_sbu_558k.json`(shuffle seed=42 后取 `[start, end)`)
- 每条:一张图 + 一句人类问题

### 处理
```python
inputs = processor(images=[img], text=prompt, return_tensors="pt")
prompt_len = inputs["input_ids"].shape[-1]            # 315(图像已展开)
outs = bigmodel.generate(**inputs, max_new_tokens=1024, temperature=1.0)
full_ids = outs[0].cpu()                              # [827]
```

### 输出(落盘,每条一个 `.pt`)
| 字段 | 形状/类型 | 含义 |
|---|---|---|
| `full_ids` | `[827]` int64 | prompt + 生成回复的完整序列 |
| `prompt_len` | `315` int | 用于切分 loss_mask(prompt 不算 loss) |
| `image_file` | str | 图片路径(训练时再 `Image.open`) |

> ⚠️ **不存 hidden state**,只存 token,每条几 KB。这是「在线」相对「离线」省空间的核心。
> ⚠️ 本地图片只有约 25%,缺图样本自动 skip,所以原始索引范围要开到约 4×目标条数。

---

## 4. 第 2 步:数据集层(`OnlineMMDataset.__getitem__`)

读入一条 `.pt`,**现场派生两个 mask**:

```python
full_ids   = rec["full_ids"][:max_len]                # [827]
prompt_len = 315

# loss_mask:prompt 段=0(不算 loss),回复段=1(算 loss)
loss_mask  = [0]*315 + [1]*512                        # len 827

# image_mask:标记图像 token 位置(供 ImgAdaptor 定位视觉特征)
image_mask = (full_ids == 151655)                     # 256 个 True,都落在 prompt 段内
```

**collator(`OnlineMMCollator`,强制 bs=1)** 打包为:

| 字段 | 形状 |
|---|---|
| `full_ids` | `[1, 827]` |
| `loss_mask` | `list`,内含一个长度 827 的 list |
| `image_mask` | `list`,内含一个长度 827 的 list |
| `image_file` | str |

---

## 5. 第 2 步:目标模型实时 forward(`build_online_mm_tensors`)

这是「在线」的核心:把轻量 token 还原成与离线等价的训练张量。

### 输入
```python
full_ids = batch["full_ids"]                          # [1, 827]
img_proc = processor.image_processor(images=[image])  # 只取视觉特征!
# → pixel_values, image_grid_thw
```

> 🔴 **坑点 1**:只能用 `image_processor`,**不能**用整个 `processor(text=..., images=...)`。
> 因为 `full_ids` 里图像 token 已被展开成 256 个,若再喂 `processor`,它会把每个已展开的
> `<|image_pad|>` 当成一处新图再次展开 → `IndexError: index 1 out of bounds`。
> `input_ids` 直接复用 `full_ids` 即可(gen 与训练用同一 min/max_pixels,视觉 token 数严格一致)。

### 目标模型 forward(冻结,no_grad)
```python
outs = target_model(input_ids=full_ids, pixel_values=..., image_grid_thw=...,
                    output_hidden_states=True)
hidden_full = outs.hidden_states[-1].float()   # [1, 827, 3584]  末层隐状态
embeds_full = outs.hidden_states[0].float()    # [1, 827, 3584]  embedding 层(已含图像 embedding)
```

### 右移构造监督信号(EAGLE 范式:用「位置 t 的 hidden」预测「位置 t+1 的 hidden」)
```python
# 草稿模型的输入 token-embedding(右移一位,末尾补 0)
inputs_embeds = cat([embeds_full[:, 1:, :], zeros[:, :1, :]], dim=1)   # [1, 827, 3584]

# teacher 目标:下一位置的末层 hidden(右移一位)
target        = cat([hidden_full[:, 1:, :], zeros[:, :1, :]], dim=1)   # [1, 827, 3584]

hidden_states = hidden_full + noise        # [1, 827, 3584] 训练加均匀噪声做正则
loss_mask[:, -1] = 0                        # 最后一位无下一 token,置 0
```

### 输出(交给草稿模型)
| 张量 | 形状 | 角色 |
|---|---|---|
| `hidden_states` | `[1, 827, 3584]` | 草稿模型主输入(当前位置末层 hidden + 噪声) |
| `inputs_embeds` | `[1, 827, 3584]` | 草稿模型 token-embedding 输入(右移) |
| `target` | `[1, 827, 3584]` | **teacher 隐状态**(右移),loss 用 |
| `loss_mask` | `[1, 827]` bool | 只有 512 个回复位置为 True |
| `image_mask` | `[1, 827]` bool | 256 个图像位置为 True |

---

## 6. 第 2 步:草稿模型 forward(ViSpec 核心 `cnets_ours.Model`)

> 🔴 **坑点 2**:草稿模型 forward 必须 `input_ids=None` + 只传 `inputs_embeds`
> (其断言 `(input_ids is None) ^ (inputs_embeds is not None)`,两者只能其一)。

```python
predict = model(
    hidden_states,            # [1, 827, 3584]
    input_ids=None,
    inputs_embeds=inputs_embeds,   # [1, 827, 3584]
    attention_mask=attn,
    image_mask=image_mask_t,  # [1, 827] bool
)
# predict: [1, 827, 3584]
```

### 6.1 ViSpec 创新点:视觉 token 压缩(`ImgAdaptor`)

草稿模型不会原样处理 256 个图像 token(太重),而是用一个**注意力池化层** `ImgAdaptor`
把它们压缩成 `num_q=2` 个 query:

```
256 个图像 token embedding [1, 256, 3584]
        │  ImgAdaptor:2 个可学习 query 对 256 个 key/value 做交叉注意力
        ▼
[1, 2, 3584]  →  1 个进入序列, 1 个存为 last_img_hidden(作为后续文本的视觉上下文)
```

于是序列在草稿模型内部被压缩:`827 - 256 + 1 = 572`,算 attention 时只跑 572,**省计算**。
算完后再用 `trans_mat`(置换/还原矩阵)einsum 映射回 827,与 `target` / `loss_mask` 对齐:

```python
hidden_states = torch.einsum("bn...,bnm->bm...", hidden_states, trans_mat)  # 572 → 827
```

### 6.2 文本位置的融合(EAGLE 风格)

对非图像位置,草稿模型把 **token-embedding** 与 **末层 hidden** 拼接后过一个 `fc` 融合,
并把视觉上下文 `last_img_hidden` 通过 `img_fc` 注入:

```python
hidden = img_fc(cat(txt_hidden, last_img_hidden))     # 注入视觉上下文
h_s    = fc(cat(txt_emd, hidden))                     # 融合 embedding + hidden → 进 decoder layer
```

### 输出
| 张量 | 形状 | 含义 |
|---|---|---|
| `predict` | `[1, 827, 3584]` | 草稿模型预测的「下一位置末层 hidden」 |

---

## 7. 第 2 步:MTP 多步展开(Multi-Token Prediction)

`mtp_steps=1`,即在「预测下一个 token」之外,再多预测 1 步(预测下下个),提升投机解码命中长度。

```python
mtp_predicts = [predict]                 # 第 0 步:直接预测
mtp_predict  = predict
for m in range(mtp_steps):               # 1 次
    # 把上一步预测当作输入,再 forward 一次(模拟"已接受上一个草稿 token"后的状态)
    mtp_predict = cat([hidden_states[:, :1], mtp_predict[:, :-1]], dim=1)
    mtp_predict = model(mtp_predict, input_ids=None, inputs_embeds=inputs_embeds,
                        attention_mask=attn, image_mask=image_mask_t)
    mtp_predicts.append(mtp_predict)

mtp_predicts = torch.cat(mtp_predicts, dim=0)   # [2, 827, 3584]   (mtp_steps+1 = 2)
```

| 张量 | 形状 |
|---|---|
| `mtp_predicts` | `[2, 827, 3584]`(2 = mtp_steps+1) |

---

## 8. 🎯 损失函数(`compute_loss`,本文重点)

### 8.1 直觉

草稿模型不是直接学 token,而是学**目标模型在每个位置的输出概率分布**(知识蒸馏)。
具体把草稿 hidden 和 teacher hidden 各自过**同一个冻结的 lm_head**,得到词表上的分布,再让两者接近。

### 8.2 准备 teacher 概率

```python
target_head = head(target)                          # [1, 827, 152064]  teacher logits
# 复制 mtp_steps+1 份,与 mtp_predicts 对齐
target_head = target_head.expand([2, 1, 827, 152064]).flatten(0,1)  # [2, 827, 152064]
target_p    = softmax(target_head, dim=-1).detach() # [2, 827, 152064]  teacher 概率(不回传)

loss_mask = loss_mask[:, :, None].expand([2,1,827,1]).flatten(0,1)   # [2, 827, 1]
```

### 8.3 `compute_loss` 内部(两部分损失)

```python
def compute_loss(target_p, predict, loss_mask, topk=10):
    out_head = head(predict)                         # [2, 827, 152064]  student logits

    # ---- 只保留 loss_mask=True 的位置(本例:512 回复位置 × 2 mtp ≈ 1024 行)----
    masked_logits = out_head[loss_mask[..., 0]]      # [N, 152064]
    target_p      = target_p[loss_mask[..., 0]]      # [N, 152064]
    predict_p     = softmax(masked_logits, dim=-1)   # student 概率

    # ===== ① 分布对齐损失 ploss(L1 / 全词表)=====
    l1_distance = torch.abs(predict_p - target_p)    # |student - teacher|
    ploss = torch.mean(l1_distance.sum(dim=-1))      # 每位置对全词表求和,再对所有位置平均

    # ===== ② Top-k 排序蒸馏损失 rloss(ListMLE 风格)=====
    _, topk_indices = torch.topk(target_p, k=10, dim=-1)        # teacher 概率最高的 10 个 token
    student_topk_logits = out_head[loss_mask[...,0]].gather(-1, topk_indices)  # student 在这 10 个上的 logit
    # 让 student 在这 10 个 token 上的"排序"与 teacher 一致(plackett-luce / listMLE)
    reversed_logits = torch.flip(student_topk_logits, dims=[-1])
    log_cumsum_exp  = torch.logcumsumexp(reversed_logits, dim=-1)
    log_denominator = torch.flip(log_cumsum_exp, dims=[-1])
    log_likelihood  = student_topk_logits - log_denominator
    rloss = -torch.mean(log_likelihood.sum(-1))

    return 10 * ploss + 0.1 * rloss, out_head
```

### 8.4 两部分的作用对比

| 损失 | 公式核心 | 作用 | 权重 |
|---|---|---|---|
| **ploss** | `mean( Σ_vocab |softmax(student) − softmax(teacher)| )` | 让 student **整个词表分布**逼近 teacher(L1 距离) | **× 10** |
| **rloss** | ListMLE:`−mean( Σ logP(排序) )` over teacher top-10 | 让 student 在 teacher **最可能的 10 个 token 上排序一致**(投机解码只在乎 top 命中) | **× 0.1** |

> **最终 loss** = `10 * ploss + 0.1 * rloss`,只反传更新**草稿模型**(目标模型、lm_head 全程冻结)。

### 8.5 监控指标
```python
predicted = out_head.argmax(-1)
target_ids = target_head.argmax(-1)
acc = (predicted == target_ids)[loss_mask].mean()    # top-1 命中率,越高投机解码越快
# 另记 top-1/2/3 accuracy
```

---

## 9. 一次 step 的完整数据流(本例汇总)

```
.pt 文件
  full_ids [827] / prompt_len 315 / image_file
        │
        ▼  OnlineMMDataset + Collator
  full_ids [1,827] / loss_mask(512 个1) / image_mask(256 个1) / image_file
        │
        ▼  build_online_mm_tensors  (目标模型 forward, no_grad)
  ┌─ hidden_states [1,827,3584]  (末层 hidden + noise)
  ├─ inputs_embeds [1,827,3584]  (embedding 层, 右移)
  ├─ target        [1,827,3584]  (末层 hidden, 右移 = teacher 信号)
  ├─ loss_mask     [1,827] bool
  └─ image_mask    [1,827] bool
        │
        ▼  草稿模型 forward (ImgAdaptor 压缩 256→2, 内部 827→572→trans_mat→827)
  predict [1,827,3584]
        │
        ▼  MTP 展开 (mtp_steps=1)
  mtp_predicts [2,827,3584]
        │
        ▼  head(冻结 lm_head)  +  compute_loss
  ┌─ student: softmax(head(mtp_predicts))   [N,152064]
  ├─ teacher: softmax(head(target))         [N,152064]   (detach)
  │     N ≈ 512(回复位置) × 2(mtp) = 1024 行
  ▼
  loss = 10 * ploss(全词表 L1) + 0.1 * rloss(top-10 排序)
        │
        ▼  accelerator.backward → 只更新草稿模型
```

---

## 10. 如何运行

```bash
# 前置:① Stage1 在线 ckpt 存在(checkpoints/stage1_qwen7b_online/state_20/model.safetensors)
#       ② 预生成数据已就绪

# 第 1 步:预生成长回复(默认 5 卡并行,扫描 [0,160000),约产 4 万有效条)
bash gen_stage2_parallel.sh
# 或:START=0 END=160000 GPUS="0,1,2,3,4" MAXTOK=1024 bash gen_stage2_parallel.sh

# 第 2 步:在线训练(超参对齐 README 2.2:lr=3e-6 bs=1 max-len=4096 mtp-steps=1 num-q=2)
bash train_stage2_online.sh
# 或:GPUS="0,1,2,3" DATAPATH=data/train/gen_mm_online_full bash train_stage2_online.sh
```

checkpoint 输出到 `checkpoints/stage2_qwen7b_online/state_<epoch>`,每 5 个 epoch 存一次,支持断点续训。

---

## 11. 与离线版(`train_stage2.sh` / `main_mtp.py`)的唯一区别

| | 离线 | 在线(本文) |
|---|---|---|
| hidden state | 提前算好存盘(`.ckpt`,十几 MB/条) | 训练循环里实时 forward |
| 预存数据 | `hidden_state` + `inputs_embeds` | 只存 `full_ids`(几 KB/条) |
| 目标模型 | 训练时不加载 | 常驻显存,冻结 eval |
| **草稿模型 / 损失 / MTP / ImgAdaptor** | **完全相同** | **完全相同** |

> 在线与离线已验证数值对齐:hidden_state 逐元素一致,Stage1/2 loss 曲线在 0.6% 内重合。
</content>
</invoke>
