"""
ViSpec Stage 1 在线训练入口(main_online.py)
==============================================================================
基于 vispec/train/main.py 改造,唯一区别:目标模型 hidden state 由训练循环里
实时 forward 产出,而非读预存 .ckpt。下游(草稿 forward / 蒸馏 loss / 保存)与
离线版完全一致。

数据流对照:
  离线 main.py:  torch.load(ckpt) → hidden_state / inputs_embeds
  在线 (本文件): target_model(input_ids, output_hidden_states=True)
                    → hidden_states[-1] (末层) / hidden_states[0] (embedding 层)

用法见 train_stage1_online.sh。超参与 README 2.1 一致。
==============================================================================
"""

import argparse

parser = argparse.ArgumentParser(description="sp-online-stage1")
parser.add_argument("--basepath", type=str, default="./model/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--configpath", type=str, default="config.json")
parser.add_argument("--loadpath", type=str, default=None)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--bs", type=int, default=1)
parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
parser.add_argument(
    "--data-json",
    type=str,
    default="data/train/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json",
)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=20)
parser.add_argument("--cpdir", type=str, default="0")
parser.add_argument("--pw", type=float, default=0.1)
parser.add_argument("--num-workers", type=int, default=2)
parser.add_argument("--max-len", type=int, default=4096)
parser.add_argument("--begin-epoch", type=int, default=0)
args = parser.parse_args()

train_config = {
    "lr": args.lr,
    "bs": args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "is_warmup": True,
    "num_epochs": 20,
    "p_w": args.pw,
    "v_w": 1.0,
    "head_w": 0.1,
    "num_workers": args.num_workers,
    "data_noise": True,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "max_len": args.max_len,
    "config_path": args.configpath,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
    "save_freq": 5,
}
import json
import os

try:
    from torch_npu.contrib import transfer_to_npu
except:
    pass

import torch
from safetensors import safe_open

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(0)
accelerator = Accelerator(
    gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
)
# 多卡:显式把当前进程绑定到自己的物理 GPU,避免目标模型常驻时
# 多个 rank 抢同一张卡(NCCL "Duplicate GPU detected")
if torch.cuda.is_available():
    torch.cuda.set_device(accelerator.local_process_index)
from typing import Any, Dict, List

import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from ..model.cnets import Model
from ..model.configs import EConfig
from .online_dataset import OnlineTextDataset

if accelerator.is_main_process:
    import wandb

    wandb.init(project="ess", entity="yuhui-li", mode="offline", config=train_config)
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=f"{args.cpdir}/run")

# ---------------------------------------------------------------------------
# head(目标模型 lm_head,冻结)—— 与 main.py 完全一致
# ---------------------------------------------------------------------------
try:
    baseconfig = AutoConfig.from_pretrained(args.basepath)
    try:
        head = torch.nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size, bias=False)
    except:
        head = torch.nn.Linear(
            baseconfig.text_config.hidden_size,
            baseconfig.text_config.vocab_size,
            bias=False,
        )
    try:
        try:
            with open(
                os.path.join(args.basepath, "model.safetensors.index.json"), "r"
            ) as f:
                index_json = json.loads(f.read())
                head_path = index_json["weight_map"]["lm_head.weight"]
            with safe_open(
                os.path.join(args.basepath, head_path), framework="pt", device="cpu"
            ) as f:
                tensor_slice = f.get_slice("lm_head.weight")
                vocab_size, hidden_dim = tensor_slice.get_shape()
                tensor = tensor_slice[:, :hidden_dim].float()
        except:
            with open(
                os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r"
            ) as f:
                index_json = json.loads(f.read())
                head_path = index_json["weight_map"]["lm_head.weight"]
            weights = torch.load(os.path.join(args.basepath, head_path))
            tensor = weights["lm_head.weight"].float()
    except:
        m = AutoModelForImageTextToText.from_pretrained(args.basepath, torch_dtype="auto")
        try:
            tensor = m.language_model.lm_head.weight.float()
        except:
            tensor = m.lm_head.weight.float()
        del m
except:
    tensor = torch.load(args.basepath)["lm_head.weight"].float()
    head = torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False)

head.weight.data = tensor
head.eval()
for param in head.parameters():
    param.requires_grad = False


# ---------------------------------------------------------------------------
# Stage 1 文本预处理:复刻 ge_data_all_qwen_shargpt.py 的 build_dataset_rank
#   (不直接 import 该模块,因为它顶层会立刻加载 7B 模型)
# ---------------------------------------------------------------------------
def build_text_dataset(tokenizer, json_path, start, end):
    from datasets import load_dataset

    ds = load_dataset("json", data_files=json_path)["train"]
    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(start, end))
    original_columns1 = ds1.column_names

    def preprocess_function(examples):
        new_examples = {"input_ids": [], "loss_mask": []}
        for i in range(len(examples["id"])):
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            convroles = ["user", "assistant"]
            roles = {"human": "user", "gpt": "assistant"}
            source = examples["conversations"][i]
            if roles[source[0]["from"]] != "user":
                source = source[1:]
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                if role != convroles[j % 2]:
                    break
                messages.append({"role": role, "content": sentence["value"]})
            conversation = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id
            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=train_config["max_len"],
                add_special_tokens=False,
            ).input_ids[0]
            loss_mask = torch.ones_like(input_ids)

            sep = "<|im_end|>\n<|im_start|>assistant\n"
            sep2 = "<|im_end|>\n<|im_start|>user\n"
            turns = conversation.split(sep2)
            if len(turns) < 2:
                continue
            turns[1] = turns[0] + sep2 + turns[1]
            turns = turns[1:]
            cur_len = 1
            loss_mask[:cur_len] = 0
            for k, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)
                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                instruction_len = len(tokenizer(parts[0]).input_ids)
                if k == 0:
                    loss_mask[0 : cur_len + instruction_len - 2] = 0
                else:
                    loss_mask[cur_len - 6 : cur_len + instruction_len - 2] = 0
                cur_len += turn_len
                cur_len += 5
            loss_mask[cur_len:] = 0
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])
        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        remove_columns=original_columns1,
        load_from_cache_file=True,
    )
    ds1.set_format(type="torch")
    return ds1


# ---------------------------------------------------------------------------
# 在线 collator:只 pad input_ids / loss_mask / attention_mask
# ---------------------------------------------------------------------------
class OnlineCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(f["input_ids"].shape[0] for f in features)
        batch_input_ids = torch.stack(
            [
                torch.cat(
                    [
                        f["input_ids"],
                        torch.zeros(
                            max_length - f["input_ids"].shape[0], dtype=f["input_ids"].dtype
                        ),
                    ]
                )
                for f in features
            ]
        )
        batch_loss_mask = torch.tensor(
            [f["loss_mask"] + [0] * (max_length - len(f["loss_mask"])) for f in features]
        )
        batch_attention_mask = torch.tensor(
            [
                [1] * len(f["loss_mask"]) + [0] * (max_length - len(f["loss_mask"]))
                for f in features
            ]
        )
        return {
            "input_ids": batch_input_ids,
            "loss_mask": batch_loss_mask,
            "attention_mask": batch_attention_mask,
        }


def top_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


def compute_loss(target, target_p, predict, loss_mask):
    loss_mask = loss_mask.to(bool)
    out_head = head(predict)
    out_logp = nn.LogSoftmax(dim=-1)(out_head[loss_mask[..., 0]])
    if out_logp.numel() == 0:
        return out_logp.sum(), out_logp.sum(), out_head
    target_p = target_p[loss_mask[..., 0]]
    plogp = target_p * out_logp
    ploss = -torch.mean(plogp.sum(-1))
    vloss = criterion(predict[loss_mask[..., 0]], target[loss_mask[..., 0]])
    vloss = torch.mean(vloss.mean(-1))

    _, topk_indices = torch.topk(target_p, k=10, dim=-1)
    student_topk_logits = out_head[loss_mask[..., 0]].gather(-1, topk_indices)
    reversed_logits = torch.flip(student_topk_logits, dims=[-1])
    log_cumsum_exp = torch.logcumsumexp(reversed_logits, dim=-1)
    log_denominator = torch.flip(log_cumsum_exp, dims=[-1])
    log_likelihood = student_topk_logits - log_denominator
    rloss = -torch.mean(log_likelihood.sum(-1))
    return vloss, ploss + 0.1 * rloss, out_head


# ---------------------------------------------------------------------------
# 核心:用目标模型实时算 hidden state,并复刻离线的右移构造
#   返回与离线 batch 同构的 dict:hidden_states / inputs_embeds(右移) / target(右移)
# ---------------------------------------------------------------------------
def build_online_tensors(target_model, input_ids, attention_mask, add_noise):
    with torch.no_grad():
        outs = target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_full = outs.hidden_states[-1].float()  # [bs, seq, H]  末层
        embeds_full = outs.hidden_states[0].float()   # [bs, seq, H]  embedding 层

    # 右移一位构造 target / inputs_embeds(与 main.py __getitem__ 完全一致)
    inputs_embeds = torch.cat(
        [embeds_full[:, 1:, :], torch.zeros_like(embeds_full[:, :1, :])], dim=1
    )
    target = torch.cat(
        [hidden_full[:, 1:, :], torch.zeros_like(hidden_full[:, :1, :])], dim=1
    )
    hidden_states = hidden_full

    # uniform noise 增强(仅训练集;复刻 AddUniformNoise,作用于 hidden_state_big)
    if add_noise:
        seq = hidden_states.shape[1]
        noise = (
            (torch.rand_like(hidden_states) - 0.5)
            * train_config["std"]
            * 512
            / seq
        )
        hidden_states = hidden_states + noise

    return hidden_states, inputs_embeds, target


# ---------------------------------------------------------------------------
# 数据集 / dataloader
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(args.basepath, use_fast=False)
full_ds = build_text_dataset(tokenizer, args.data_json, args.start, args.end)

n = len(full_ds)
split = max(1, int(n * 0.95))
train_hf = full_ds.select(range(0, split))
test_hf = full_ds.select(range(split, n)) if split < n else full_ds.select(range(n - 1, n))

traindataset = OnlineTextDataset(train_hf, max_len=train_config["max_len"])
testdataset = OnlineTextDataset(test_hf, max_len=train_config["max_len"])

train_loader = DataLoader(
    traindataset,
    batch_size=train_config["bs"],
    shuffle=True,
    collate_fn=OnlineCollator(),
    num_workers=train_config["num_workers"],
    pin_memory=True,
)
test_loader = DataLoader(
    testdataset,
    batch_size=train_config["bs"],
    shuffle=False,
    collate_fn=OnlineCollator(),
    num_workers=train_config["num_workers"],
    pin_memory=True,
)

# ---------------------------------------------------------------------------
# 续训(与 main.py 一致)
# ---------------------------------------------------------------------------
if not os.path.exists(args.cpdir):
    if accelerator.is_main_process:
        os.makedirs(args.cpdir)
else:
    ckpts = os.listdir(args.cpdir)
    if ckpts:
        begin_epoch = max(
            int(c.split("_")[1]) + 1 if c.startswith("state") else 0 for c in ckpts
        )
        loadpath = os.path.join(args.cpdir, f"state_{begin_epoch - 1}", "model.safetensors")
        if os.path.exists(loadpath):
            print(f"resume from {loadpath}")
            args.loadpath = loadpath
            args.begin_epoch = begin_epoch

# ---------------------------------------------------------------------------
# 草稿模型
# ---------------------------------------------------------------------------
config = EConfig.from_pretrained(train_config["config_path"])
model = Model(config, load_emb=True, path=args.basepath)

if args.loadpath:
    with open(args.loadpath, "rb") as f:
        from safetensors.torch import load

        state_dict = load(f.read())
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if len(missing_keys) > 0:
            print(f"missing_keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"unexpected_keys: {unexpected_keys}")

# ---------------------------------------------------------------------------
# 目标模型:在线训练核心 —— 常驻显存,冻结,eval
# ---------------------------------------------------------------------------
# 目标模型是纯推理(eval + no_grad),用 sdpa 注意力比默认 eager 快 30%~50%
# 且数值精确(非近似),不影响 hidden_state 正确性。个别环境算子不支持时回退 eager。
try:
    target_model = AutoModelForImageTextToText.from_pretrained(
        args.basepath, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
except Exception as e:
    print(f"sdpa 加载失败,回退 eager: {e}")
    target_model = AutoModelForImageTextToText.from_pretrained(
        args.basepath, torch_dtype=torch.bfloat16
    )
target_model.eval()
for p in target_model.parameters():
    p.requires_grad = False
target_model = target_model.to(accelerator.device)

criterion = nn.SmoothL1Loss(reduction="none")
optimizer = optim.AdamW(
    model.parameters(), lr=train_config["lr"], betas=(train_config["b1"], train_config["b2"])
)

num_epochs = train_config["num_epochs"]
num_warmup_steps = len(train_loader) * 1
total_steps = len(train_loader) * num_epochs
is_warmup = train_config["is_warmup"]

if is_warmup:
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )
    model, head, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader, scheduler
    )
else:
    model, head, optimizer, train_loader, test_loader = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader
    )

if is_warmup:
    for i in range(args.begin_epoch * len(train_loader)):
        scheduler.step()

for epoch in range(args.begin_epoch, num_epochs + 1):
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    epoch_vloss = 0
    epoch_ploss = 0
    num_batches = 0
    model.train()
    for batch_idx, data in enumerate(
        tqdm(train_loader, disable=not accelerator.is_local_main_process)
    ):
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            # —— 在线:目标模型实时算 hidden state ——
            hidden_states, inputs_embeds, target = build_online_tensors(
                target_model,
                data["input_ids"],
                data["attention_mask"],
                add_noise=True,
            )
            attn = data["attention_mask"]
            loss_mask_t = data["loss_mask"].clone()
            loss_mask_t[:, -1] = 0  # 最后一位不算 loss(与离线一致)

            predict = model(
                hidden_states,
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
            )
            with torch.no_grad():
                target_head = head(target)
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()
            loss_mask = loss_mask_t[:, :, None]
            vloss, ploss, out_head = compute_loss(target, target_p, predict, loss_mask)
            loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
            optimizer.step()
            if is_warmup:
                scheduler.step()

        with torch.no_grad():
            _, predicted = torch.max(out_head, 2)
            _, target_ids = torch.max(target_head, 2)
            ct = loss_mask.sum().item()
            cc = ((predicted == target_ids) * loss_mask.squeeze()).sum().item()
            out_head_f = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            target_ids_f = target_ids.view(-1)[loss_mask.view(-1) == 1]
            if ct != 0:
                topkacc = top_accuracy(out_head_f, target_ids_f, (1, 2, 3))
                for top_i in range(len(topkacc)):
                    top_3acc[top_i] += topkacc[top_i]
                total += ct
                correct += cc
        if accelerator.is_main_process and ct != 0:
            logdict = {
                "train/lr": optimizer.optimizer.param_groups[0]["lr"],
                "train/vloss": vloss.item(),
                "train/ploss": ploss.item(),
                "train/loss": loss.item(),
                "train/acc": cc / ct,
            }
            for idx, i in enumerate(top_3acc):
                logdict[f"train/top_{idx + 1}_acc"] = topkacc[idx].item() / ct
            wandb.log(logdict)
            writer.add_scalars("train", logdict, epoch * len(train_loader) + batch_idx)

        epoch_loss += loss.item()
        epoch_vloss += vloss.item()
        epoch_ploss += ploss.item()
        num_batches += 1
        del ploss, vloss

    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    epoch_loss /= num_batches
    epoch_vloss /= num_batches
    epoch_ploss /= num_batches
    if accelerator.is_main_process:
        print(
            "Epoch [{}/{}], Loss: {:.4f}, Vloss: {:.4f}, Ploss: {:.4f}".format(
                epoch + 1, num_epochs, epoch_loss, epoch_ploss, epoch_vloss
            )
        )
        print("Train Accuracy: {:.2f}%".format(100 * correct / max(total, 1)))
        wandb.log({"train/epochacc": correct / max(total, 1), "train/epochloss": epoch_loss})
        writer.add_scalars(
            "train_epoch",
            {"train/epochacc": correct / max(total, 1), "train/epochloss": epoch_loss},
            epoch,
        )

    # 每 save_freq 个 epoch 评估+保存一次;最后一个 epoch 兜底
    if epoch % train_config["save_freq"] == 0 or epoch == num_epochs:
        correct = 0
        total = 0
        epoch_loss = 0
        num_batches = 0
        model.eval()
        for batch_idx, data in enumerate(
            tqdm(test_loader, disable=not accelerator.is_local_main_process)
        ):
            with torch.no_grad():
                hidden_states, inputs_embeds, target = build_online_tensors(
                    target_model, data["input_ids"], data["attention_mask"], add_noise=False
                )
                loss_mask_t = data["loss_mask"].clone()
                loss_mask_t[:, -1] = 0
                predict = model(
                    hidden_states,
                    inputs_embeds=inputs_embeds,
                    attention_mask=data["attention_mask"],
                )
                target_head = head(target)
                target_p = nn.Softmax(dim=2)(target_head).detach()
                loss_mask = loss_mask_t[:, :, None]
                vloss, ploss, out_head = compute_loss(target, target_p, predict, loss_mask)
                loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
                _, predicted = torch.max(out_head, 2)
                _, target_ids = torch.max(target_head, 2)
                ct = loss_mask.sum().item()
                cc = ((predicted == target_ids) * loss_mask.squeeze()).sum().item()
                total += ct
                correct += cc
            epoch_loss += loss.item()
            num_batches += 1

        correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
        correct, total = accelerator.gather_for_metrics((correct, total))
        correct, total = correct.sum().item(), total.sum().item()
        epoch_loss /= max(num_batches, 1)
        if accelerator.is_main_process:
            print(
                "Test Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, epoch_loss)
            )
            print("Test Accuracy: {:.2f}%".format(100 * correct / max(total, 1)))
            wandb.log({"test/epochacc": correct / max(total, 1), "test/epochloss": epoch_loss})
            writer.add_scalars(
                "test",
                {"test/epochacc": correct / max(total, 1), "test/epochloss": epoch_loss},
                epoch,
            )
            accelerator.save_state(output_dir=f"{args.cpdir}/state_{epoch}")
            import shutil

            shutil.copyfile(args.configpath, f"{args.cpdir}/state_{epoch}/config.json")

if accelerator.is_main_process:
    writer.close()
