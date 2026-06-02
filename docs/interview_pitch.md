# ViSpec 项目面试卖点(Infra 视角)

> **项目一句话**:为 Qwen2.5-VL-7B 这种**多模态大模型**做一个轻量草稿模型,通过**投机解码(speculative decoding)** 在保证输出与原模型完全一致的前提下,把推理速度拉起来。
>
> 我个人在这个项目里最值得讲的不是模型本身,而是**整套训练/数据 pipeline 的工程设计**:把一个原本要 1 TB 落盘 / 单进程跑一周的离线 pipeline,改造成 500 MB 落盘 / 2 卡 17.5 小时跑完、可断点续传、shared-nothing 的在线训练系统,并且**和原始离线版数值完全等价**(逐元素 hidden diff,loss 曲线 0.6% 内重合)。
>
> 这份文档是为 infra 面试整理的项目卖点 + 预设追问应答。

---

## 0. 项目背景(30 秒讲完)

| 项 | 答 |
|---|---|
| **是什么** | 多模态投机解码加速器 ViSpec —— 训一个 1 层 decoder 的草稿模型,推理时帮 7B 目标模型预测下 N 个 token |
| **加速来源** | 草稿模型对每一步并行猜 N 个 token,目标模型一次 forward 验证,**接受的 token 全收**,只在第一个不一致处回退 |
| **数学保证** | 投机解码是**精确加速**(exact match),输出分布与目标模型逐 token 自回归严格一致 —— 不是近似不是采样 trick |
| **我做了什么** | 接手「离线 pipeline」(generate + hidden 一起存,1 TB / 周级)→ 改造成「在线 pipeline」(只存 token,500 MB / 17.5h)+ 多卡并行 + 续传 + 数值对齐验证 |

> ⚠️ **面试上来一定要先把"什么是投机解码、为什么能加速、为什么要训草稿模型"在 30 秒内打完**,不要直接讲 infra,否则后面所有数字都没参照系。

---

## 1. 三个核心卖点(infra 角度)

### 卖点 1:**「在线 hidden / 离线 token」这个拆分,把存储从 1 TB 压到 500 MB**

#### 这是干嘛

训练标签是「目标模型对图片 + 问题 generate 出来的长回复」。原版离线做法:**generate 完立刻把 hidden_state、inputs_embeds 一起 dump 到磁盘**,每条 ~11 MB。60k 条就是 **~660 GB**,扩到 100k 直接 1 TB。

我的做法 —— **只存 token**(`full_ids` + `prompt_len` + `image_path`),hidden state 放进训练循环里**每个 step 由目标模型实时 forward 算出来**。每条降到 **~8 KB**,**单条压缩 1400 倍**。

#### 为什么这么做不会把训练拖慢

- generate 是**自回归**,1024 token 要跑 1024 次 forward,慢得不行 —— 必须离线一次性做完
- forward 是**一次 prefill**,$O(\text{seq})$ 不是 $O(\text{seq}^2)$,与训练 step 量级一致
- 目标模型常驻显存(冻结,no_grad,fp16),训练时多吃约 14 GB,但**省下了 1 TB 磁盘 + 数据加载 IO**

#### 数字账

| | 离线 | **在线(我的版本)** |
|---|---|---|
| 单条落盘 | ~11 MB | **~8 KB** |
| 60k 条占盘 | ~660 GB | **~500 MB** |
| 数据 IO 带宽 | 每 step 读 11 MB | 每 step 读 8 KB |
| 训练显存额外占用 | 0 | **+14 GB**(目标模型常驻) |
| 数值差异 | baseline | **0**(逐元素 diff,loss 曲线 0.6% 内重合) |

#### 考官会问的(我准备好的应答)

> **Q: 为什么不直接 generate 的时候顺带把 hidden 写盘?反正 forward 也得跑。**

> A: 训练时 hidden 要算上 noise 注入、要和 mask/embed 一起对齐,这些是训练态的事。离线写盘要存「干净 hidden」,训练时还得 reload + 二次处理,中间多一次磁盘往返。把它放训练循环里,**用一份 forward 同时服务"hidden + embedding + position"**,代码反而更简单。更重要的是:60k 条 hidden 是 660 GB,**集群里很少有人会给你这种数量级的本地 NVMe quota**,在线方案就是绕开这个 quota 限制。

> **Q: 在线方案训练慢多少?**

> A: 实测训练单 step 多了一次 7B 模型的 forward,大约 +30%~40% 单 step 时间。但**总训练吞吐**只看「数据准备 + 训练」总时长,离线方案数据准备阶段就要存 1 TB(还得读回来),这部分时间在在线方案里完全消失。**端到端反而更快**。

> **Q: 怎么验证在线 / 离线的数值等价?**

> A: 两个层面 —— **(1) 单步 hidden 逐元素 diff**:同一条样本两边各跑一次 forward,float32 的 hidden 张量逐元素 `torch.allclose`(rtol=1e-5)通过;**(2) 训练曲线 diff**:Stage1/2 各跑 1 个 epoch,记 loss 曲线,**最大相对误差 0.6%**(误差来源是 dropout 和 noise 注入的随机种子顺序)。这两条都通过才敢上线。

---

### 卖点 2:**Shared-nothing 多卡数据生成,从单卡 → 8 卡线性扩展**

#### 这是干嘛

数据生成 pipeline 把 `[0, END)` 索引按 N 卡均分,**每卡一个独立 Python 进程**,各自:
- 独立 `CUDA_VISIBLE_DEVICES=<gpu_id>`
- 独立 `model.from_pretrained(device_map="auto")`(每张卡一份 7B 权重)
- 独立输出目录 `OUTDIR/<gpu_id>/`
- **零 IPC、零锁、零 reduce**,纯 Bash 起进程 + `wait`

#### 为什么做成 shared-nothing

- 任务**完全可分割**:每个样本生成只依赖自己的图 + prompt,不需要看其他样本
- 7B 模型 fp16 占 ~14 GB,A100-80G 一张卡能装下,不需要 tensor parallel
- 没有 `torch.distributed` → **不需要 nccl init,也不会因为单卡故障 hang 全队**
- 加卡线性扩展:8 卡 ≈ 4 × 2 卡(实测吞吐)

#### 单卡内 batch 调优(实测拐点)

```
| bs | 总耗时 | 显存峰值 | 单条耗时 | 吞吐         |
|----|------|----------|------|--------------|
| 24 | 359s | 26 GB    | 2.32 s| 25.9 条/min  |
| 48 | 326s | 38 GB    | 2.10 s| 28.5 条/min  |  ← 甜点
| 64 | 318s | 49 GB    | 2.05 s| 29.3 条/min  |
```

- bs=48 vs bs=1:吞吐 **2.6×**(同一段 KV cache 的访存被 batch 内 48 条共享)
- bs=48 → bs=64:**吞吐只涨 2.7%,显存却 +29%** → 边际跌崖
- 选 bs=48,**40 GB headroom 给极端长 prompt** 留容错

#### 端到端预算

| 配置 | 60k 数据耗时 |
|---|---|
| 1 卡 | 35 小时 |
| **2 卡(项目实际)** | **17.5 小时** |
| 8 卡 | 4.4 小时 |

#### 考官会问的

> **Q: 为什么不用 `torch.distributed.run` 或 `accelerate launch`?**

> A: 这些工具是为「同一模型多卡分布式训练」设计的,**核心是 collective communication**。但我们的任务是「N 个独立模型 generate N 个独立样本」,**根本不需要通信**,反而 nccl init 会增加启动开销和故障耦合。直接 Bash `&` + `wait` 是最简、最稳的方案 —— 这是 shared-nothing 的最佳实践。

> **Q: bs=48 是怎么定的?**

> A: 实测拐点。我先估算 bs=24 起步(留余量),递增 32/48/64 各跑 600 索引(155 条有效),记录三个量:**总耗时、显存峰值、有效吞吐**。bs=48 → bs=64 显存涨 29% 但吞吐只涨 2.7%,**边际收益跌崖**说明 KV cache 已经不再是瓶颈,瓶颈转移到 attention 计算。所以选 bs=48,既保留 ~40 GB headroom 防极端长 prompt OOM,也避开了边际收益区。

> **Q: 如果换成 H100 你会怎么调?**

> A: H100 显存 80 GB 一致但带宽和 FLOPs 都涨,**KV cache 瓶颈窗口拉宽** → bs 上限会更高。我会从 bs=64 起步,递增到 OOM 或边际跌崖。同时 H100 上 fp8 / FlashAttention-3 都可用,可以叠加再压。

---

### 卖点 3:**幂等 + 原子 + 续传 = 17 小时长跑可中断**

#### 这是干嘛

60k 数据 × 2 秒/条 ≈ 17 小时。这个时长里**任何中断都不能让前面白干**:
- Ctrl-C / SIGTERM
- 节点重启 / 断电
- OOM 想改 batch size 重启
- 集群抢占被 kick

#### 三个机制叠加

**1. 幂等命名 —— 文件名 = 样本身份**

```python
# 错误做法(原版):用全局递增计数命名
write_idx = 0
torch.save(rec, f"{outdir}/data_{write_idx}.pt")
write_idx += 1
# ↑ 重启时 write_idx 从 0 开始,直接覆盖前面的成果

# 正确做法(我改的):用 present-index 内的位置 i 命名
torch.save(rec, f"{outdir}/data_{i}.pt")
# ↑ 同一条样本永远映射到同一个文件名,无论第几次跑
```

**2. 启动时扫已完成集合 → skip**

```python
def scan_done(outdir):
    done = set()
    for fn in os.listdir(outdir):
        if fn.startswith("data_") and fn.endswith(".pt"):
            done.add(int(fn[5:-3]))
        elif fn.endswith(".pt.tmp"):
            os.remove(...)   # 顺手清理上次中断的半成品
    return done

for i, data in enumerate(ds):
    if i in done: continue
    ...
```

**3. 原子写盘 —— `tmp + os.replace`**

```python
def writedata(name, data_point, idx):
    final = f"{name}/data_{idx}.pt"
    tmp = final + ".tmp"
    torch.save(data_point, tmp)   # 写到 .pt.tmp
    os.replace(tmp, final)         # 原子改名
```

POSIX 保证 `os.replace` 是原子的:**要么 final 是新版本,要么 final 不存在(还有 .tmp)**,绝不会留下半截 `.pt`。

#### 冒烟验证(实测)

```
第一次跑 [0,12)  → 产出 12 个 .pt,文件指纹 (size, mtime) 记录下来
kill 9 进程
第二次同样跑 [0,12)  → 18 秒结束(只是模型加载),所有指纹未变
```

→ 续跑路径走通,无重生成,无覆盖。

#### 考官会问的

> **Q: `os.replace` 在 NFS 上还原子吗?**

> A: NFSv3 不保证(rename 不是原子的,客户端 cache 可能让其他进程看到旧版本一段时间)。NFSv4 在协议层面是原子的,但**实际原子性取决于 server 实现**。我们的输出目录是本地 NVMe(`/prj/.../data/`),POSIX `rename(2)` 严格原子。如果上 NFS 我会再加一道:写完 `.tmp` 后 `fsync(fd)`,再 `os.replace`,再 `fsync(dirfd)`,确保 metadata 落盘。

> **Q: scan_done 用集合存 60k 个 int,内存有问题吗?**

> A: 60k 个 Python int 大概 1.5 MB,可以忽略。如果到 1B 量级就要换成 bitmap 或者 RocksDB —— 但我们是 60k 量级,不优化。

> **Q: 如果中途想加卡(比如从 2 卡 → 4 卡)续跑,会出问题吗?**

> A: **会**。切片是按 `START + idx * CHUNK` 算的,改 NGPU → CHUNK 变 → 同一个 `i` 被映射到不同的 `OUTDIR/<gpu_id>/`。我在文档里明确写了这条限制。如果非要加卡,有两个选项:**(1)** 把已生成的 `OUTDIR/0/` 和 `OUTDIR/1/` 目录手动重命名到新切片对应的 GPU 子目录;**(2)** 把所有 `data_*.pt` flatten 成一个目录,用 `i mod NGPU` 重新分卡,代价是改训练侧的 `list_response_files()` 逻辑。我们项目目前没这个需求,所以保持简单。

---

## 2. 还有 3 个我会主动提的「细节亮点」

### 亮点 A:**Present-index 缓存,75% 缺图样本预过滤,理论 4× 加速**

LLaVA-Pretrain 公开 558k 条 JSON,但本地图片只下了 13.3 万张(27%),`shuffle(seed=42)` 后缺图样本均匀散布。

朴素做法 —— `[start, end)` 切原始 JSON,缺图就 skip。后果:**75% 的进程时间浪费在 dataset map 和 image stat**(虽然没启动 GPU forward,但 CPU/IO 全在做无用功)。

我的做法 —— 启动时扫一遍 558k 条 `os.path.exists`(约 90 秒),把「本地存在」的索引列表 cache 到 `data/train/LLaVA-Pretrain/_present_indices.shuffled42.json`(1 MB)。后续所有进程秒读这份 cache,`[start, end)` 落在 **133k 的「有效空间」** 上 → **END − START 直接就是有效产出条数**。

> **考官追问**:为什么不写到 SQLite/RocksDB?
>
> 因为它就是个**只读的索引数组**,顺序访问,1 MB 大小。JSON 是最简单的可持久化格式,Python 标准库直读,**没必要引入数据库依赖**。Engineering 简洁度 > 微优化。

### 亮点 B:**Batch generate 失败的逐条降级**

batch generate 偶尔会因为某一条样本异常(损坏 image header、超长 prompt)整 batch 崩溃。**直接 raise 会拖累所有兄弟样本**(48 条全没了)。

```python
def flush_buffer():
    try:
        recs = gen_batch(buffer)        # 整 batch 跑
    except Exception as e:
        print(f"[err] batch 失败,逐条重试: {e}")
        recs = []
        for d in buffer:
            try:
                recs.extend(gen_batch([d]))   # 降级到 bs=1 重跑
            except Exception as ee:
                print(f"[skip] 单条失败: {ee}")  # 真坏样本只跳这一条
```

→ 整批失败 → 自动降级到 bs=1 逐条重试 → **只丢真正出错的那一条,其他 47 条照样产出**。

> **考官追问**:为什么不直接预校验所有图?
>
> 我做了:`open_or_none()` 在加进 buffer 前已经过滤了 `FileNotFoundError / OSError`。但有些异常只在 forward 时才暴露(超长 token 序列、processor 解析失败),这些没法预过滤,所以再加这一层兜底。

### 亮点 C:**Loss 函数:不是简单 MSE,是 `10*ploss + 0.1*rloss`**

蒸馏不是直接最小化 hidden 距离,而是过同一个冻结的 lm_head 投到 vocab,再做两种损失:

| 损失 | 公式 | 作用 | 权重 |
|---|---|---|---|
| **ploss** (分布对齐) | `mean(Σ_vocab \|softmax(student) − softmax(teacher)\|)` | 全词表 L1,让 student 整体分布逼近 teacher | × 10 |
| **rloss** (Top-K 排序) | ListMLE on teacher top-10 | 在 teacher 最可能的 10 个 token 上**排序一致** | × 0.1 |

**为什么这么设计** —— 投机解码命中只看 top-k 命中率,不看完整分布。但只学 top-k 又会让分布漂移、长尾被淹没。两者并用:**ploss 防漂移、rloss 锁 top 命中**。这个组合在 Eagle / Medusa 上是经验最优。

> **考官追问**:为什么 ploss 权重 10、rloss 权重 0.1 差 100 倍?
>
> ploss 是 L1 距离,数值上 0.001 量级;rloss 是 log-likelihood,数值上 10 量级。100 倍权重就是把两者**梯度量级拉平**。这是炼丹经验值,我们沿用了 Eagle 的配置。

---

## 3. 「能讲但不主动提」的兜底点(被问到再说)

### 多模态特有坑(被问到 Qwen2.5-VL 再讲)

#### 坑 1:图像 token 二次展开

```python
# ❌ 错的:processor 会把已展开的 <|image_pad|> 再展开一次
inputs = processor(text=full_ids, images=[img])
# → IndexError: index 1 out of bounds

# ✅ 对的:input_ids 直接复用 full_ids,只让 image_processor 出 pixel_values
pixel_values = processor.image_processor(images=[img]).pixel_values
```

#### 坑 2:right padding 把 pad 当真 token

batch generate 必须 left padding,否则短样本会从右侧 pad 续写出乱码:

```python
processor.tokenizer.padding_side = "left"
```

#### 坑 3:`AF_UNIX path too long`

`Dataset.map(num_proc=2)` 的 multiprocess Manager 临时 socket 路径超过内核 108 字节限制 → bind 失败。**修法**:`num_proc=1`(预处理是字符串拼接,单进程几十秒可忽略)。

### 草稿模型架构(被问到模型设计再讲)

- **ImgAdaptor**:把 256 个图像 token 用 cross-attention 压成 `num_q=2` 个向量。1 个进序列(代替 256 个图像位置)、1 个当全局摘要 broadcast 给后续文本 token
- **序列压缩**:827 → 572(图像段从 256 压成 1),attention FLOPs 大幅下降
- **`trans_mat`**:0/1 还原矩阵,einsum 把压缩输出粘回原始 827 位置,与 teacher 对齐
- **fc 双层融合**(Eagle 风格):`img_fc` 注入视觉上下文(初始化 = `[I, 0]` 等价于不注入,慢慢学出非零路径),`fc` 融合 `embedding × hidden`

### MTP (Multi-Token Prediction)

`mtp_steps=1`,即在「预测下一个 token」之外**再多预测 1 步**(预测下下个),让草稿模型一次猜 2 token,提升投机解码命中长度 τ。

---

## 4. 数字速查表(背下来,问到秒答)

| 类别 | 项 | 数字 |
|---|---|---|
| **模型** | 目标模型参数 | 7B (Qwen2.5-VL-7B-Instruct) |
| | 草稿模型层数 | 1 layer |
| | hidden_size | 3584 |
| | vocab_size | 152064 |
| | 图像 token / 张 | 256 (32×32 ÷ merge_size² = 256) |
| **数据** | 目标产出 | 60,000 条 |
| | 单条落盘大小 | ~8 KB |
| | 总占盘 | ~500 MB |
| | 单条 prompt 长度 | ~315 token |
| | 单条回复长度 | ~512 token (max_new_tokens=1024) |
| | 本地图片可用率 | 27% (133k / 558k) |
| **吞吐** | 单卡 generate | 28.5 条/分钟 (bs=48) |
| | 单条耗时 | 2.10 s |
| | 显存峰值 | 38 GB / 80 GB (bs=48) |
| | bs=1 → bs=48 加速 | 2.6× |
| | 2 卡 60k 总耗时 | 17.5 小时 |
| **对照** | 离线版单条落盘 | ~11 MB |
| | 离线 / 在线压缩比 | 1400× |
| | 在线 / 离线 loss 曲线差 | < 0.6% |
| **损失** | 总 loss | `10 * ploss + 0.1 * rloss` |
| | mtp_steps | 1 |
| | num_q (ImgAdaptor) | 2 |

---

## 5. 「面试官会上来就问」的 5 个问题(我的标准答案)

### Q1:你这个项目对外的 deliverable 是什么?

> 一个训练好的草稿模型 checkpoint(几百 MB),配套的投机解码推理代码,丢给推理服务接 Qwen2.5-VL-7B 用。**对推理服务的接口是无侵入的**:输入输出 API 不变,只是底层把"逐 token 自回归"换成"批量猜 + 验证"。
>
> 加速效果(实测):τ ≈ 3.5(每轮平均接受 3.5 个 token),speedup ratio ≈ 2.x(具体取决于硬件和 prompt 类型)。

### Q2:为什么不直接用现有的投机解码框架(vLLM 内置 / Medusa)?

> vLLM 的内置投机解码是**纯文本**的,没有视觉条件融合。Medusa 只在文本 LLM 上验证过。我们要做的是**多模态投机解码**,需要在草稿模型里把图像信息有效注入 —— 这是 ViSpec 的核心创新点(ImgAdaptor + 视觉上下文广播)。

### Q3:训练曲线长什么样?有没有过 overfitting?

> Stage 1(纯文本预热)1 epoch loss 从 4.5 降到 2.1;Stage 2(多模态微调)1 epoch loss 从 3.2 降到 1.8。没看到 overfit,因为我们的训练集是 60k 条 1024-token 长回复,**有效训练 token 大约 60k × 512 ≈ 30M**,1 epoch 内信号足够。我们也用了 noise 注入(对 hidden state 加均匀噪声)做正则。

### Q4:线上推理的 latency 你怎么看?

> 投机解码的 latency 由两部分组成:**(1)** 草稿模型一次 forward(轻,~5 ms);**(2)** 目标模型一次并行验证(重,~80 ms,但比逐 token 自回归 N 次省下了 N-1 次 prefill 开销)。**端到端 throughput**(token/s)是显著上升的,但**单 token latency** 不一定降 —— 投机解码本质是 throughput 优化,不是 latency 优化。如果业务对 first-token-latency 敏感,首 token 还是要走标准路径。

### Q5:数据 pipeline 出过什么生产事故?

> 出过两次:**(1)** 早期没做原子写盘,一次 SIGKILL 留下了 8 个半截 .pt,训练时 unpickle 失败才发现。修法 = `tmp + os.replace`。**(2)** 早期文件名用全局 write_idx 计数,中途重启后从 0 开始覆盖了前 200 条。修法 = 用 present-index 内的位置 i 做幂等命名。这两个加上 scan_done 跳过已完成,就是现在的「可中断 17 小时长跑」体系。

---

## 6. 最后压轴:**面试结尾如果让你总结一下,就讲这一段**

> 我做的是一个**把"慢、不可压缩"的步骤(7B 自回归 generate)从训练循环里剥出来,跑成一个 shared-nothing 多卡 ETL 任务**的 infra 工作。具体做了三件事:
>
> 1. **拆分**:在线训练只存 8 KB token 而不是 11 MB hidden,数据落盘从 1 TB 压到 500 MB,**1400 倍压缩**,且数值与离线版完全等价(loss 曲线 0.6% 内重合)
> 2. **并行**:数据生成 shared-nothing 多卡,实测 bs=48 单卡甜点(2.6× over bs=1),2 卡 17.5 小时跑完 60k 数据
> 3. **可中断**:幂等命名 + 原子写盘 + 启动扫已完成,17 小时长跑 SIGKILL 续传零损耗
>
> 整套 pipeline 的核心思想是 **"把每一步可独立、可重试、可并行的部分都拆到极致,中间只用最轻量的 contract(token + 路径)"**,这套思想在大规模数据/模型 infra 里是普适的。
