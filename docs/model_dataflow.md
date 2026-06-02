# ViSpec 模型数据流图

## 整体架构

ViSpec 是一个**投机解码（Speculative Decoding）**框架，由一个轻量 **Draft Model（DM）** 和一个冻结的 **Target Model（TM）** 组成。DM 提前猜多个 token，TM 一次并行验证，接受的 token 直接用，拒绝则退回自回归。

```
输入图文
   │
   ▼
┌─────────────────────────────────┐
│        Target Model (冻结)       │
│   Qwen2.5-VL / LLaVA-1.6        │
│                                  │
│  视觉编码器 ──► inputs_embeds    │
│  语言模型   ──► hidden_states    │
└──────────┬──────────────────────┘
           │ hidden_states
           │ inputs_embeds
           ▼
┌─────────────────────────────────┐
│         Draft Model (可训练)     │
│         cnets_ours.Model         │
│                                  │
│  ImgAdaptor ◄── 图像 token emd  │
│  img_fc     ◄── txt_hidden +    │
│                  last_img_hidden │
│  fc         ◄── txt_emd + hidden│
│  Decoder Layer                   │
└──────────┬──────────────────────┘
           │ draft token 树
           ▼
┌─────────────────────────────────┐
│       Target Model 验证          │
│  接受 / 拒绝 draft tokens        │
└─────────────────────────────────┘
```

---

## Stage 1：纯文本 Draft Model 预训练

**脚本**：`vispec/train/main_online.py`（在线）/ `main.py`（离线）

**使用模型**：`cnets.Model`（**不含 ImgAdaptor**）

```
ShareGPT 文本数据
       │
       ▼
  Target Model (冻结)
  语言模型 forward
       │
       ├──► hidden_states   [seq, H]
       └──► inputs_embeds   [seq, H]
       │
       ▼
  Draft Model (cnets)
  ┌──────────────────────────┐
  │  fc(cat(emd, hidden))    │   ← 把 embedding 和 hidden 拼接后压缩
  │  Decoder Layer           │   ← 1 层 Transformer
  └──────────────────────────┘
       │
       ▼
  predict logits
       │
       ▼
  Loss = SmoothL1(predict, lm_head(target))
```

**可训练参数**：`fc`、`Decoder Layer`
**冻结参数**：`embed_tokens`、`lm_head`、Target Model 全部

---

## Stage 2：多模态 Draft Model 微调

**脚本**：`vispec/train/main_mtp_online.py`（在线）/ `main_mtp.py`（离线）

**使用模型**：`cnets_ours.Model`（**含 ImgAdaptor**）

**数据**：`gen_stage2_responses.py` 预生成的图文长回复数据

```
图文数据（LLaVA-Pretrain / ShareGPT4V 等）
       │
       ▼
  Target Model (冻结)
  视觉编码器 + 语言模型 forward
       │
       ├──► inputs_embeds   [seq, H]   ← 含图像 token embedding
       ├──► hidden_states   [seq, H]
       └──► image_mask      [seq]      ← 标记哪些位置是图像 token
```

### Draft Model 内部数据流（含 ImgAdaptor）

> 一句话先说清楚:序列里图像 token 太多(Qwen2.5-VL 1 张图 256 个),原样塞 attention 太重。
> 我们让 `ImgAdaptor` 把这 256 个图像 token **压缩成 `num_q=2` 个 query 向量**,
> 其中 **前 `num_q-1` = 1 个**留在序列里(代替原来 256 个图像 token 的位置),
> **最后 1 个**当作「整图摘要」`last_img_hidden`,**广播给后面所有文本 token** 作为视觉上下文。
> 这样原来 827 个 token 的序列被压缩成 `827 − 256 + 1 = 572`,attention FLOPs 大幅下降;
> 算完再用一个 0/1 还原矩阵 `trans_mat` einsum 映射回 827 与 teacher 对齐。

#### 0. 前置:image_mask 与「图像段切分」

```
image_mask:  [seq] bool      (例:位置 59..314 这 256 个为 True)

代码用 image_mask 找到「每段连续图像 token 的右端点」 last_img_ids:
  ends = image_mask[:-1] & ~image_mask[1:]   # 由 True 转 False 的位置
       | image_mask[-1:]                      # 末尾若仍是 True 也算一段结束
  → last_img_ids = [314]                      # 本例只有一段图像 (0..58 文本 / 59..314 图 / 315..826 文本)
```

之后按 `last_img_ids` 把整个序列切成「文本段 → 图像段 → 文本段 → 图像段 → ... → 尾部文本」,逐段处理。本例只有一张图,所以是「文本(59) + 图像(256) + 文本(512)」三段。

#### 1. 文本段(图像之前):用 last_img_hidden 注入视觉上下文

```
位置 0..58 是图像之前的纯文本(prompt 的开头部分)
当前还没遇到任何图,所以:
  last_img_hidden = zeros [1, H]               ← 初始化为零向量(没有视觉信息)

txt_emd     = inputs_embeds[非图位置]          [num_txt=59, H]    ← embedding 层输出
txt_hidden  = hidden_states[非图位置]          [num_txt=59, H]    ← 末层 hidden(+noise)
txt_img     = last_img_hidden.expand_as(txt_hidden)
            = [1, H] 复制成 [59, H]            ← ★「广播」就是这一步:把 1 个 H 维向量
                                                沿 token 维复制 59 次,让每个文本位置
                                                都看到「整图摘要」(此处还是 0)

# 第一次融合:把视觉摘要和文本 hidden 拼起来,过 img_fc 压回 H 维
hidden  = img_fc(cat(txt_hidden, txt_img))     [59, H]
       = Linear(2H → H)([txt_hidden | txt_img])

# 第二次融合(Eagle 风格):把上一步结果再和 token embedding 拼起来,过 fc 压回 H 维
h_s     = fc(cat(txt_emd, hidden))             [59, H]   ← 这一段就是这 59 个文本位置的草稿模型输入
```

**两层 fc 的分工**:
- **`img_fc`** —— 注入「视觉上下文」:hidden 直通 + last_img_hidden 注入(初始化时 `eye(H)` 拼 `zeros(H)`,即开训练时**完全等价于不注入,只放 hidden**,然后慢慢学出非零的视觉路径);
- **`fc`** —— Eagle 风格的「embedding × hidden」融合:把当前 token 的 embedding 和上一步带视觉的 hidden 一起送进 decoder。

#### 2. 图像段:ImgAdaptor 把 256 个图像 token 压成 2 个

```
img_emd = inputs_embeds[图像位置 59..314]      [256, H]
        unsqueeze(0)                            [1, 256, H]
            │
            ▼   ImgAdaptor (cross-attention,详见下方「ImgAdaptor 结构细节」节)
            │   Q: 可学习 [num_q=2, num_heads, head_dim]   ← 不依赖输入,纯参数
            │   K,V: img_emd 经 k_proj/v_proj
            │   sdpa(Q, K, V, is_causal=False)              ← 全局注意力
            │   o_proj
            ▼
img_adapted [1, 2, H] → squeeze(0) → [2, H]    ← 256 个图像 token 摘要成 2 个 query 向量

★ 关键拆分:
   img_adapted[:-1] = img_adapted[0:1]  → [1, H]  作为「图像段在序列里的代表」,
                                                   塞进 h_s 占用原来 256 个图像 token 的位置
                                                   (压缩比 256:1)
   img_adapted[-1:] = img_adapted[1:2]  → [1, H]  存进 self.last_img_hidden,
                                                   留给「图像之后」的文本段做广播
                                                   (代表整张图的全局摘要)

h_s.append(img_adapted[:-1])                   ← 序列里这一段长度从 256 → 1
self.last_img_hidden = img_adapted[-1:]        ← 更新「最近的图像摘要」,供后面文本段用
```

> 为什么要分成两个? `num_q=2` 是经验值,本质是:**1 个**留位置(让序列还有图像位置可以被 attention 看到),**1 个**摘要(给所有后续文本注入全局视觉)。`num_q` 加大会得到更细粒度的视觉特征,但序列变长、计算变多。

#### 3. 尾部文本段(图像之后):再次注入 last_img_hidden

```
位置 315..826 是图像之后的回复 token,共 512 个
此时 last_img_hidden 已经是图像段输出的「整图摘要」(非零),所有后续文本都能看到它

rst_emd     = inputs_embeds[315..826]          [512, H]
rst_hidden  = hidden_states[315..826]          [512, H]
rst_img     = last_img_hidden.expand_as(rst_hidden)   ★ 广播:1 个摘要复制 512 次
            = [1, H] → [512, H]                       让每个回复位置都"看到"整图

hidden  = img_fc(cat(rst_hidden, rst_img))     [512, H]    ← 同 §1
h_s.append(fc(cat(rst_emd, hidden)))           [512, H]
```

#### 4. 三段拼接 → 压缩序列

```
h_s = cat([
        前段文本 h_s    [59, H],     ← §1 输出
        图像段代表       [1, H],     ← §2 的 img_adapted[:-1]
        尾段文本 h_s    [512, H],     ← §3 输出
      ]) → [seq_compressed=572, H]            ← 原 827 - 255(图像段被压成 1) = 572
```

#### 5. 进 Decoder Layer + trans_mat 还原回原长度

```
h_s [572, H] ──► Decoder Layer (1 层 Transformer) ──► hidden_572 [572, H]
                                                              │
                                                              ▼
trans_mat [572, 827]   (0/1 矩阵,记录"压缩位 → 原始位"的对应)
   构造方式: 取出 eye(827) 在「文本段」与「图像段右端 num_q-1 个位置」对应的行,
            按拼接顺序堆起来 → 形状 [572, 827]
                                                              │
                                                              ▼
hidden_827 = einsum("bnh,bnm->bmh", hidden_572, trans_mat)    ← 把压缩位的输出
                                                                "粘"回原始位置
                                                                未被粘到的位置(图像段
                                                                内 256 - 1 = 255 个位置)
                                                                自然为 0
                                                              │
                                                              ▼
predict logits [827, H]  ← 与 target / loss_mask 形状对齐
```

> ⚠️ `trans_mat` 不是"反卷积",**它只是把压缩后的输出按位置映射回去,被压掉的图像位输出为 0**。
> loss 只在回复段(loss_mask=1, 共 512 位)算,图像位本来就被 mask 掉,所以"为 0 也无所谓"。

#### 6. MTP 多步展开 + Loss

```
predict [1, 827, H]
   │
   │  MTP (Multi-Token Prediction, mtp_steps=1):
   │  把 predict 当作"已接受的下一 token 表示",再 forward 一次,得到下下个的预测
   │
   ▼
mtp_predicts [mtp_steps+1=2, 827, H]
   │
   ▼  head() 是冻结的 lm_head, 把 H 维投影到 vocab=152064
   │
predict_logits [2, 827, 152064]   teacher_logits [2, 827, 152064] (= head(target))
   │                                    │
   └────────────┬───────────────────────┘
                ▼
           只保留 loss_mask=1 的位置 → [N, 152064] (N ≈ 1024)
                ▼
   ┌──────────────────────────────────────────────────────────┐
   │ ploss = mean( Σ_vocab |softmax(student) - softmax(teacher)| )│  ← 全词表 L1
   │ rloss = ListMLE 风格,top-10 token 上"排序一致"            │  ← 排序蒸馏
   │ loss  = 10 * ploss + 0.1 * rloss                          │
   └──────────────────────────────────────────────────────────┘
                │
                ▼
       backward → 只更新草稿模型(ImgAdaptor / img_fc / fc / Decoder Layer)
```

**可训练参数**：`ImgAdaptor`、`img_fc`、`fc`、`Decoder Layer`
**冻结参数**：`embed_tokens`、`lm_head`、Target Model 全部

---

## ImgAdaptor 结构细节

将数量不固定的图像 token（可能几十到几百个）**压缩为固定 `num_q` 个向量**（默认 `num_q=2`）：

```
img_emd [1, num_img_tok, H]
         │
         ├── k_proj ──► K [1, num_img_tok, H]
         └── v_proj ──► V [1, num_img_tok, H]

Q: 可学习参数 [num_q, num_heads, head_dim]
         │
         ▼
scaled_dot_product_attention(Q, K, V)   ← is_causal=False（全局注意力）
         │
         ▼
o_proj ──► [1, num_q, H]
```

**作用**：把变长的图像 token 序列摘要为固定长度的全局视觉特征，后续广播给所有文本 token 使用。

---

## 推理阶段（投机解码）

```
输入图文
    │
    ▼
Target Model → inputs_embeds, hidden_states
    │
    ▼
Draft Model → 生成 token 候选树（depth=3, top_k=8, total_token=30）
    │
    ▼
Target Model 并行验证候选树
    │
    ├── 接受的 token → 直接输出
    └── 第一个拒绝处 → 目标模型采样一个 token，截断候选树
    │
    ▼
循环直到 EOS 或 max_new_token
```

**关键指标**：
- **τ（acceptance length）**：每轮平均接受 token 数，τ 越大越好
- **ratio（speedup）**：`spec_tok/s ÷ baseline_tok/s`，即实际加速比
