# ViSpec 在线训练 I/O 详解（Stage 1 / Stage 2 / DM）

> 本文只针对 **在线训练**（`vispec/train/main_online.py`、`vispec/train/main_mtp_online.py`），
> 不再覆盖离线 `main.py` / `main_mtp.py` 的预存 ckpt 流程。
> 阅读前默认你已经看过 `README.md` 的 Stage 1 / Stage 2 章节。

---

## 0. 在线 vs 离线一句话区别

| 维度 | 离线（`main.py` / `main_mtp.py`） | 在线（`main_online.py` / `main_mtp_online.py`） |
|---|---|---|
| 目标模型何时算 hidden | 提前用 `ge_data/...` 跑一遍并落盘成 `.ckpt` | **训练循环里实时 forward**，目标模型常驻显存 |
| 数据集返回什么 | 已算好的 `hidden_state` / `inputs_embeds` / `loss_mask` …… | 仅原始输入：`input_ids` / `loss_mask`（Stage 2 多一个 `image_file` + `image_mask`） |
| 是否需要预生成 | 必须先跑 `ge_data_all_*` 落盘 | **Stage 1 不需要**（直接读 ShareGPT json）；**Stage 2 需要**预生成「长回复 token」（轻量 .pt，几 KB/条），见 `vispec/ge_data/gen_stage2_responses.py` |
| 单条样本占盘 | hidden 落盘十几 MB | Stage 1：0；Stage 2：仅 token，几 KB |
| 数值一致性 | — | 与离线版 hidden_state 逐元素一致，loss 曲线 0.6 % 内重合（见 memory `online-offline-loss-alignment-verified.md`） |

**核心**：在线版把"目标模型 forward"挪进了训练循环，其余（草稿模型 forward、右移构造 target、蒸馏 loss、保存）与离线版逐行对齐。

---

## 1. 术语对齐

| 名称 | 含义 |
|---|---|
| **Target / Big Model** | 被加速的大型 VLM（如 `Qwen2.5-VL-7B-Instruct`）。`eval` + 全部参数 `requires_grad=False`，**仅做 teacher**。 |
| **DM（Draft Model，草稿模型）** | 真正被训练的小模型。Stage 1 用 `vispec/model/cnets.py:Model`（无视觉模块）；Stage 2 用 `vispec/model/cnets_ours.py:Model`（含 `ImgAdaptor` + `img_fc`）。 |
| **head** | 从 Target 拷贝出来的 `lm_head` 权重，`eval()` + freeze。所有蒸馏分布都通过它计算（见 `main_online.py:154-157`、`main_mtp_online.py:148-151`）。 |
| **hidden_state** | Target 最后一层 transformer 的输出（`outs.hidden_states[-1]`）。是 DM 的核心输入信号、也是 DM 的回归目标（右移一位）。 |
| **inputs_embeds** | Target embedding 层输出（`outs.hidden_states[0]`）。Stage 2 直接保留多模态序列里图像 token 的 embedding 形态，避免重新 tokenize。 |
| **loss_mask** | 1 表示该 token 计入 loss，0 表示 prompt / padding 不计入。所有版本最后一位强制 0（最后一位没有"下一位"）。 |
| **image_mask**（仅 Stage 2） | 1 表示该位置是图像 token，DM 会对其走 `ImgAdaptor` 分支。 |

---

## 2. Stage 1 在线训练（`vispec/train/main_online.py`）

**入口脚本**：`train_stage1_online.sh`
**草稿模型**：`vispec/model/cnets.py:Model`（无视觉模块）
**数据来源**：`data/train/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json`（纯文本对话）
**关键超参**（与 README 2.1 严格一致）：`lr=3e-5`、`bs=1`、`max-len=4096`、`num_epochs=20`、`p_w=0.1`、`v_w=1.0`、`std=0.2`、`save_freq=5`

### 2.1 数据流（按 batch）

```
ShareGPT json
   ↓ build_text_dataset (复刻 ge_data_all_qwen_shargpt.preprocess_function)
input_ids [seq] (long) + loss_mask [seq]      ← OnlineTextDataset
   ↓ OnlineCollator: pad 到 max_length
batch = { input_ids [B,L], loss_mask [B,L], attention_mask [B,L] }
   ↓ build_online_tensors  ← 在线版的核心：实时调用 target_model
hidden_states  = outs.hidden_states[-1].float()   # [B, L, H]   末层
inputs_embeds  = outs.hidden_states[0].float()    # [B, L, H]   embed 层
   ↓ 右移一位（与离线版完全一致）
inputs_embeds  ← cat(embeds[:,1:],     0_pad)
target         ← cat(hidden[:,1:],     0_pad)
hidden_states  ← embeds_full           （+uniform noise，仅训练时）
   ↓ DM forward（cnets.Model）
predict [B, L, H]
   ↓ compute_loss（vloss + ploss + 0.1·rloss）
loss
```

### 2.2 DM 的输入字段

调用见 `main_online.py:473-477`、`main_online.py:560-564`：

```python
predict = model(
    hidden_states,                # ← Target 末层 hidden（训练时加 uniform noise）
    inputs_embeds=inputs_embeds,  # ← Target embed 层 hidden 的"右移一位"版本
    attention_mask=attention_mask,
)
```

| key | shape | 说明 |
|---|---|---|
| `hidden_states` | (B, L, H) | Target 末层 hidden；训练时叠加 uniform noise（`(rand-0.5)*std*512/L`）；Stage 1 噪声只加到 hidden 上，不动 embed |
| `inputs_embeds` | (B, L, H) | `embeds_full` 左移 1 位（末位 0 pad）—— "下一位的 embedding" |
| `attention_mask` | (B, L) | 有效长度的 1 |

> **不传 `input_ids`**：因为 `inputs_embeds` 已经覆盖；`cnets.Model.forward` 强制要求二者**恰好二选一**。

### 2.3 DM 的输出 / 损失目标

DM 输出 `predict ∈ [B, L, H]`，每个位置 `predict[t]` 是 DM 对"位置 t+1 处 Target 末层 hidden"的预测。

监督来自三个量（`compute_loss` @ `main_online.py:286-305`）：

1. **vloss = SmoothL1(predict, target)** —— 隐藏态级回归，要求 DM 的 hidden 与 Target 的 hidden 数值贴近。
2. **ploss = -E[ target_p · log softmax(head(predict)) ]** —— 通过 frozen `head` 把 hidden 投到词表，做 Teacher 分布的 KD（Soft cross entropy）。
3. **rloss = listwise top-10 ranking loss** —— 对 top-10 token 之间的相对顺序额外加约束。

最终：`loss = v_w*vloss + p_w*(ploss + 0.1*rloss)`，默认 `v_w=1.0, p_w=0.1`。

### 2.4 一句话总结 Stage 1

> 在线 Stage 1 的"输入"是 ShareGPT 原始对话；目标模型在循环里现场算出 `(hidden, embed)`，构造"用 hidden\[t] + embed\[t+1] 预测 hidden\[t+1]"的回归 + 蒸馏任务，让 DM 在**纯文本**上把 `fc` + N 层 LlamaDecoderLayer 训到能贴住老师。Stage 2 加视觉模块前的"暖身"。

---

## 3. Stage 2 在线训练（`vispec/train/main_mtp_online.py`）

**入口脚本**：`train_stage2_online.sh`
**草稿模型**：`vispec/model/cnets_ours.py:Model`（**含 `ImgAdaptor` + `img_fc`**）
**数据来源**：`vispec/ge_data/gen_stage2_responses.py` 预生成的轻量 `.pt`（每条几 KB），位于 `data/train/gen_mm_online*`
- `.pt` 字段：`{ full_ids, prompt_len, image_file }`
- `full_ids` = "prompt token + Target generate 出的长回复 token"，**已包含展开后的图像 token**
**关键超参**（与 README 2.2 严格一致）：`lr=3e-6`、`bs=1`、`max-len=4096`、`mtp-steps=1`、`num-q=2`、`use-ours=True`、`num_epochs=20`、`save_freq=5`

> 训练前必须存在 Stage 1 在线 ckpt（默认 `./checkpoints/stage1_qwen7b_online/state_20/model.safetensors`），用 `--loadpath` 加载（`ImgAdaptor` / `img_fc` 是新加的，stage1 没有，会作为 missing key 用默认初始化；`img_fc` 被初始化为"恒等 + 0"，即开训时不动 hidden，慢慢学起来）。

### 3.1 数据流（按 batch，bs 强制 = 1）

```
预生成 .pt   (full_ids, prompt_len, image_file)
   ↓ OnlineMMDataset.__getitem__
{full_ids [seq], loss_mask [seq], image_mask [seq], image_file}
   ↓ OnlineMMCollator (bs=1)
batch
   ↓ build_online_mm_tensors  ← 核心：image + full_ids 实时给 target_model
target_model 输入:
   input_ids       = full_ids                         [1, L]
   pixel_values    = processor.image_processor(image) [1, 3, H, W] 等
   image_grid_thw  = ...
   attention_mask  = ones_like(full_ids)
target_model 输出:
   hidden_full = outs.hidden_states[-1].float()       # [1, L, H]
   embeds_full = outs.hidden_states[0].float()        # [1, L, H]
   ↓ 右移一位（与离线 main_mtp 完全一致）
inputs_embeds = cat(embeds[:,1:], 0_pad)
target        = cat(hidden[:,1:], 0_pad)
hidden_states = hidden_full   （+uniform noise，仅训练时）
loss_mask     = pad/截到 L，最后一位强制 0
image_mask    = pad/截到 L
   ↓ DM forward (cnets_ours.Model) + MTP 展开
predict_0 = model(hidden_states, ...)
for m in range(mtp_steps):
    feed       = cat(hidden_states[:,:1], predict_{m}[:,:-1], dim=1)
    predict_m+1 = model(feed, inputs_embeds, image_mask, ...)
mtp_predicts = cat(predict_0..predict_mtp_steps)   # [(mtp+1)*1, L, H]
   ↓ compute_loss (10·L1分布距离 + 0.1·rloss)
loss
```

### 3.2 DM 的输入字段（在线 Stage 2）

调用见 `main_mtp_online.py:407-413`：

```python
predict = model(
    hidden_states,
    input_ids=None,                # ← 关键：Stage 2 在线一律传 None
    inputs_embeds=inputs_embeds,
    attention_mask=attn,
    image_mask=image_mask_t,
)
```

| key | shape | 说明 |
|---|---|---|
| `hidden_states` | (1, L, H) | Target 末层 hidden（含图像 token 处的 hidden）；训练时加 uniform noise |
| `input_ids` | **None** | 在线 Stage 2 一律传 None，避免 `cnets_ours.forward` 里 input_ids 与 inputs_embeds 二选一约束被破坏（参考 memory `stage2-online-mm-forward-pitfalls.md`） |
| `inputs_embeds` | (1, L, H) | Target embedding 层输出，右移 1 位；含图像 token 的 embedding |
| `attention_mask` | (1, L) | 全 1（bs=1 + max-len 已截，无 padding） |
| `image_mask` | (1, L) | 1 = 图像 token 位置（来自 `full_ids == image_token_id`） |

> ⚠️ **图像不能再过一次 processor**：`full_ids` 已经是 generate 阶段 processor 把 `<|image_pad|>` 展开后的完整序列，再调一次 `processor(text=..., images=...)` 会再次展开图像 token，触发 `image_grid_thw[index]` 越界（IndexError）。实现里只用 `processor.image_processor` 取 pixel_values / image_grid_thw，`input_ids` 直接复用 `full_ids`。详见 `main_mtp_online.py:222-237` 注释 + memory `stage2-online-mm-forward-pitfalls.md`。

### 3.3 DM 内部行为（cnets_ours.Model.forward 多模态分支）

输入 `(hidden_states, inputs_embeds, image_mask)` 后，DM 在 prefill 路径上做：

1. 用 `image_mask` 找出每段图像最后一个图像 token 位置 `last_img_ids`。
2. 对每段连续区间 `[img_id_start, img_id_end)`：
   - **文本 token 部分**：
     ```
     txt_hidden_fused = img_fc( cat[txt_hidden, last_img_hidden_broadcast] )   # 把上一张图的全局视觉特征叠到每个文本 token
     out_txt          = fc( cat[txt_emb, txt_hidden_fused] )                   # embed + hidden 融合
     ```
   - **图像 token 部分**：
     ```
     img_adapted = ImgAdaptor(img_emb)                                         # [1, num_q, H]
     ```
     `num_q` 个可学习 query 通过 cross-attention 把 N 个图像 token 压缩成 `num_q` 个紧凑 hidden；最后一个保留为 `last_img_hidden`，喂给后续文本段。
3. 拼出"压缩坐标系"序列，过若干 LlamaDecoderLayer。
4. 用转移矩阵 `trans_mat`（0/1）把"压缩坐标系" hidden 反映射回原序列长度，与 Target 的 `(1, L, H)` 对齐。
5. 输出 `predict ∈ (1, L, H)`。

### 3.4 MTP 多步展开（`mtp_steps=1` 默认）

`main_mtp_online.py:414-428`：

```python
mtp_predicts = [predict]
mtp_predict  = predict
for m in range(args.mtp_steps):
    mtp_predict = cat(hidden_states[:, :1], mtp_predict[:, :-1], dim=1)
    # 把上一步 DM 自己的预测当作"老师 hidden"再喂一遍
    mtp_predict = model(mtp_predict, input_ids=None, inputs_embeds=..., image_mask=..., ...)
    mtp_predicts.append(mtp_predict)
mtp_predicts = cat(mtp_predicts, dim=0)   # shape ((mtp_steps+1)*1, L, H)
```

- 第 0 次：`hidden_states` 是真实 Target hidden（teacher 直接喂）。
- 第 m≥1 次：用**自己上一步预测**顶替 Target hidden —— 模拟推理时"没人喂 teacher hidden"的真实场景。
- README 中称这一手段是为了"避免 DM 在训练时通过 Target hidden 走 shortcut"。

### 3.5 Stage 2 损失（`compute_loss` @ `main_mtp_online.py:167-182`）

```
out_head      = head(mtp_predicts)             # ((mtp+1)*1, L, V)
target_head   = head(target).expand((mtp+1)…).flatten(0,1)
target_p      = softmax(target_head)
mask          = loss_mask 复制 (mtp+1) 份
masked_logits = out_head[mask]
predict_p     = softmax(masked_logits)
ploss         = mean( sum( |predict_p - target_p|, -1 ) )    # 分布级 L1
rloss         = listwise top-10 ranking
loss          = 10 * ploss + 0.1 * rloss
```

> 注意：Stage 2 **没有** SmoothL1 hidden 回归（vloss）。理由：Stage 2 已经从 Stage 1 ckpt 起步，hidden 数值范围已学好；这里只用分布级 L1 + ranking 让 DM 的输出分布对齐 Target 的下一 token 分布。

### 3.6 一句话总结 Stage 2

> 在线 Stage 2 的"输入"是"图片 + Stage 2 预生成的长回复 token"；目标模型在循环里现场对它 forward 出 `(hidden, embed)`，DM 走 `ImgAdaptor + img_fc + fc + N 层 LlamaDecoderLayer` 路径，并通过 MTP 多步展开模拟"没有 teacher hidden"的推理场景；监督是跨 `mtp_steps+1` 步的分布级 L1 + listwise ranking。这一阶段真正训出 ViSpec 的视觉适配模块。

---

## 4. DM forward 细化（在线训练时给它的输入 / 它的输出）

### 4.1 训练时（`past_key_values is None`，prefill 分支）

| 输入 | Stage 1 在线 | Stage 2 在线 |
|---|---|---|
| 第一个位置参数 `hidden_states` | (1, L, H) Target 末层 hidden + uniform noise | 同左，Stage 2 多了图像 token 处的 hidden |
| `input_ids` | **None** | **None** |
| `inputs_embeds` | (1, L, H) Target embed 右移 1 位 | 同左，含图像 embedding |
| `attention_mask` | (1, L) | (1, L) 全 1 |
| `image_mask` | 不传 | (1, L) bool |
| 走分支 | `cnets.Model`：`fc(cat[embed, hidden])` → N 层 decoder | `cnets_ours.Model`：图像分支（ImgAdaptor + img_fc + fc）+ 反映射 trans_mat → N 层 decoder |
| 输出 | `predict ∈ (1, L, H)` | `predict ∈ (1, L, H)`（已经过 trans_mat 映射回原 L） |

### 4.2 推理时（speculative decoding，`past_key_values is not None`，仅一位 forward）

DM 的对外接口不变，但跳过图像分支（`past_key_values is None` 守卫，见 `cnets_ours.py:879-988`）：

- 第一次（prefill）：传入 Target prefill 后的 `(hidden, embed, image_mask)`，DM 初始化 `last_img_hidden` + KV cache。
- 后续 step：仅传 `last_hidden + token_id + past_key_values`，单位置 forward。
- 用冻结 `head` 取 argmax 即下一草稿 token。`topK_genrate` 在此基础上构造 draft tree，由 Target 一次 verify。

---

## 5. 关键差异 Cheat Sheet

```
┌─────────────────────┬──────────────────────────┬──────────────────────────┐
│                     │ Stage 1 在线              │ Stage 2 在线              │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ 数据集               │ ShareGPT json (实时)      │ 预生成 .pt (token only)   │
│ DataLoader bs       │ 可 >1                    │ 强制 = 1                  │
│ DM 类                │ cnets.Model              │ cnets_ours.Model          │
│ DM 参数 input_ids   │ None                     │ None                     │
│ DM 参数 image_mask  │ 不传                      │ 必须传                    │
│ MTP                 │ 否                       │ 是 (mtp_steps=1)          │
│ Loss                │ vloss + ploss + 0.1·rloss│ 10·L1 + 0.1·rloss         │
│ 数据增强             │ uniform noise on hidden  │ 同左                      │
│ 视觉模块             │ 无                        │ ImgAdaptor + img_fc       │
│ 加载 ckpt            │ 可选(续训)                │ 必须加载 stage1 state_20  │
│ 默认 lr             │ 3e-5                     │ 3e-6                     │
│ 续训                 │ 自动从 cpdir 最大 state_*│ 同左                      │
└─────────────────────┴──────────────────────────┴──────────────────────────┘
```

---

## 6. 数值正确性验证

在线 vs 离线已经在本仓库内验证：
- `hidden_state` 逐元素一致（同 input、同精度、同 noise seed 下）
- Stage 1 / Stage 2 训练 loss 曲线 0.6% 内重合

详见 `MEMORY.md` 的 `online-offline-loss-alignment-verified.md`。

---

## 7. 引用源码位置

| 主题 | 文件:行 |
|---|---|
| Stage 1 在线主入口 | `vispec/train/main_online.py` |
| Stage 1 build_online_tensors | `vispec/train/main_online.py:312-342` |
| Stage 1 DM forward 调用 | `vispec/train/main_online.py:473-477` |
| Stage 1 compute_loss | `vispec/train/main_online.py:286-305` |
| Stage 2 在线主入口 | `vispec/train/main_mtp_online.py` |
| Stage 2 build_online_mm_tensors | `vispec/train/main_mtp_online.py:216-266` |
| Stage 2 DM forward + MTP | `vispec/train/main_mtp_online.py:407-428` |
| Stage 2 compute_loss | `vispec/train/main_mtp_online.py:167-182` |
| OnlineTextDataset / OnlineMMDataset | `vispec/train/online_dataset.py` |
| 草稿模型 (无视觉) | `vispec/model/cnets.py:Model` |
| 草稿模型 (ViSpec, 含 ImgAdaptor) | `vispec/model/cnets_ours.py:Model` / `:ImgAdaptor` |
| Stage 2 长回复预生成 | `vispec/ge_data/gen_stage2_responses.py` |
| 启动脚本 | `train_stage1_online.sh` / `train_stage2_online.sh` |
