"""
ViSpec Stage 2 在线训练入口(main_mtp_online.py)
==============================================================================
基于 vispec/train/main_mtp.py 改造。区别:目标模型 hidden state 由训练循环里
实时 forward 产出,而非读预存 .ckpt。ViSpec 独有逻辑(ImgAdaptor / 全局视觉特征
/ MTP / 蒸馏排序 loss)完全保留。

数据来源:gen_stage2_responses.py 预生成的「长回复 token」文件(只含 full_ids /
prompt_len / image_file),训练时:
  1. processor 把 image + full_ids 拼成目标模型输入
  2. 目标模型 forward → 末层 hidden_state(末层)+ embedding 层
  3. 右移构造 target,走 ImgAdaptor + MTP + 蒸馏 loss(与离线一致)

用法见 train_stage2_online.sh。超参与 README 2.2 一致(lr=3e-6 mtp-steps=1 num-q=2)。
==============================================================================
"""

import argparse

parser = argparse.ArgumentParser(description="sp-online-stage2")
parser.add_argument("--basepath", type=str, default="./model/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--configpath", type=str, default="config.json")
parser.add_argument("--loadpath", type=str, default=None)
parser.add_argument("--lr", type=float, default=3e-6)
parser.add_argument("--bs", type=int, default=1)
parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
parser.add_argument("--datapath", type=str, default="data/train/gen_mm_online")
parser.add_argument("--cpdir", type=str, default="0")
parser.add_argument("--pw", type=float, default=0.1)
parser.add_argument("--num-workers", type=int, default=2)
parser.add_argument("--max-len", type=int, default=4096)
parser.add_argument("--use-ours", type=bool, default=True)
parser.add_argument("--num-q", type=int, default=2)
parser.add_argument("--mtp-steps", type=int, default=1)
parser.add_argument("--begin-epoch", type=int, default=0)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--save-freq", type=int, default=5)
args = parser.parse_args()

train_config = {
    "lr": args.lr,
    "bs": args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "is_warmup": True,
    "num_epochs": args.epochs,
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
    "save_freq": args.save_freq,
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
    gradient_accumulation_steps=train_config["gradient_accumulation_steps"]
)
# 多卡:显式把当前进程绑定到自己的物理 GPU,避免目标模型常驻时
# 多个 rank 抢同一张卡(NCCL "Duplicate GPU detected")
if torch.cuda.is_available():
    torch.cuda.set_device(accelerator.local_process_index)
from typing import Any, Dict, List

import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    get_linear_schedule_with_warmup,
)

from ..model.configs import EConfig
from .online_dataset import OnlineMMDataset, list_response_files

if accelerator.is_main_process:
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=f"{args.cpdir}/run")

# ---------------------------------------------------------------------------
# head(目标模型 lm_head,冻结)—— 与 main_mtp.py 一致
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


def compute_loss(target_p, predict, loss_mask, topk=10):
    bsz, seq_len, vocab_size = target_p.shape
    out_head = head(predict)
    masked_logits = out_head[loss_mask[..., 0]]
    target_p = target_p[loss_mask[..., 0]]
    predict_p = F.softmax(masked_logits, dim=-1)
    l1_distance = torch.abs(predict_p - target_p)
    ploss = torch.mean(l1_distance.sum(dim=-1))
    _, topk_indices = torch.topk(target_p, k=topk, dim=-1)
    student_topk_logits = out_head[loss_mask[..., 0]].gather(-1, topk_indices)
    reversed_logits = torch.flip(student_topk_logits, dims=[-1])
    log_cumsum_exp = torch.logcumsumexp(reversed_logits, dim=-1)
    log_denominator = torch.flip(log_cumsum_exp, dims=[-1])
    log_likelihood = student_topk_logits - log_denominator
    rloss = -torch.mean(log_likelihood.sum(-1))
    return 10 * ploss + 0.1 * rloss, out_head[:bsz, ...]


# ---------------------------------------------------------------------------
# processor / image token
# ---------------------------------------------------------------------------
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    args.basepath, use_fast=True, min_pixels=min_pixels, max_pixels=max_pixels
)
image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)


# ---------------------------------------------------------------------------
# 在线 collator(bs=1):把 full_ids / loss_mask / image_mask / 图片打包
# ---------------------------------------------------------------------------
class OnlineMMCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(features) == 1, "Stage 2 仅支持 bs=1"
        f = features[0]
        return {
            "full_ids": f["full_ids"][None, :],          # [1, seq]
            "loss_mask": [f["loss_mask"]],               # list[list]
            "image_mask": [f["image_mask"]],             # list[list]
            "image_file": f["image_file"],
        }


# ---------------------------------------------------------------------------
# 核心:目标模型对 (图片 + full_ids) 实时 forward,取末层 hidden / embedding 层
#   返回与离线 batch 同构:hidden_states / inputs_embeds(右移)/ target(右移)
#   以及对齐后的 loss_mask / image_mask
# ---------------------------------------------------------------------------
def build_online_mm_tensors(target_model, batch, device, add_noise):
    full_ids = batch["full_ids"].to(device)            # [1, seq]
    image = Image.open(batch["image_file"])

    # full_ids 已是「图像 token 展开后」的完整序列(generate 时 processor 已把
    # <|image_pad|> 展开成数百个)。若再用 processor 处理 decode 文本,processor 会把
    # 序列里每个已展开的 <|image_pad|> 都当成一处新图像再次展开,触发
    # image_grid_thw[index] 越界(IndexError: index 1 out of bounds ...)。
    # 因此只用 image_processor 取视觉特征(pixel_values / image_grid_thw),
    # input_ids 直接复用 full_ids(与 loss_mask / image_mask 对齐)。
    # gen 与训练用同一 min/max_pixels,故 full_ids 里 image_pad 的数量与
    # image_grid_thw 算出的视觉 token 数严格一致。
    img_proc = processor.image_processor(images=[image], return_tensors="pt").to(device)

    model_inputs = {
        "input_ids": full_ids,
        "attention_mask": torch.ones_like(full_ids),
    }
    if "pixel_values" in img_proc:
        model_inputs["pixel_values"] = img_proc["pixel_values"]
    if "image_grid_thw" in img_proc:
        model_inputs["image_grid_thw"] = img_proc["image_grid_thw"]

    with torch.no_grad():
        outs = target_model(
            **model_inputs,
            output_hidden_states=True,
        )
        hidden_full = outs.hidden_states[-1].float()
        embeds_full = outs.hidden_states[0].float()

    inputs_embeds = torch.cat(
        [embeds_full[:, 1:, :], torch.zeros_like(embeds_full[:, :1, :])], dim=1
    )
    target = torch.cat(
        [hidden_full[:, 1:, :], torch.zeros_like(hidden_full[:, :1, :])], dim=1
    )
    hidden_states = hidden_full
    if add_noise:
        seq = hidden_states.shape[1]
        noise = (torch.rand_like(hidden_states) - 0.5) * train_config["std"] * 512 / seq
        hidden_states = hidden_states + noise

    seq = hidden_states.shape[1]
    lm = batch["loss_mask"][0][:seq] + [0] * max(0, seq - len(batch["loss_mask"][0]))
    im = batch["image_mask"][0][:seq] + [0] * max(0, seq - len(batch["image_mask"][0]))
    loss_mask = torch.tensor([lm[:seq]], dtype=torch.bool, device=device)
    loss_mask[:, -1] = 0
    image_mask = torch.tensor([im[:seq]], dtype=torch.bool, device=device)

    return hidden_states, inputs_embeds, target, full_ids, loss_mask, image_mask


# ---------------------------------------------------------------------------
# 数据集 / dataloader
# ---------------------------------------------------------------------------
files = list_response_files(args.datapath)
assert len(files) > 0, f"未找到预生成长回复数据: {args.datapath}"
split = max(1, int(len(files) * 0.95))
train_files = files[:split]
test_files = files[split:] if split < len(files) else files[-1:]

traindataset = OnlineMMDataset(
    train_files, processor, image_token_id, max_len=train_config["max_len"]
)
testdataset = OnlineMMDataset(
    test_files, processor, image_token_id, max_len=train_config["max_len"]
)
train_loader = DataLoader(
    traindataset,
    batch_size=1,
    shuffle=True,
    collate_fn=OnlineMMCollator(),
    num_workers=train_config["num_workers"],
)
test_loader = DataLoader(
    testdataset,
    batch_size=1,
    shuffle=False,
    collate_fn=OnlineMMCollator(),
    num_workers=train_config["num_workers"],
)

# ---------------------------------------------------------------------------
# 续训
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
# 草稿模型(ViSpec:cnets_ours,含 ImgAdaptor)
# ---------------------------------------------------------------------------
config = EConfig.from_pretrained(train_config["config_path"])
if args.use_ours:
    from ..model.cnets_ours import Model

    model = Model(config, load_emb=True, path=args.basepath, num_q=args.num_q)
else:
    from ..model.cnets import Model

    model = Model(config, load_emb=True, path=args.basepath)
model.gradient_checkpointing = False

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
# 目标模型:常驻显存,冻结,eval
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
    num_batches = 0
    model.train()
    for batch_idx, batch in enumerate(
        tqdm(train_loader, disable=not accelerator.is_local_main_process)
    ):
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            (
                hidden_states,
                inputs_embeds,
                target,
                full_ids,
                loss_mask_t,
                image_mask_t,
            ) = build_online_mm_tensors(
                target_model, batch, accelerator.device, add_noise=True
            )
            attn = torch.ones(
                hidden_states.shape[0], hidden_states.shape[1], device=hidden_states.device
            )

            # —— ViSpec 草稿 forward + MTP(与离线 main_mtp 完全一致)——
            # 离线草稿模型只吃 inputs_embeds(input_ids 传 None);cnets_ours.forward
            # 要求 input_ids / inputs_embeds 二选一。在线的 inputs_embeds 同样来自
            # 目标模型 embedding 层(hidden_states[0],已含图像 embedding),故与离线等价。
            predict = model(
                hidden_states,
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                image_mask=image_mask_t,
            )
            mtp_predicts = [predict]
            mtp_predict = predict
            for m in range(args.mtp_steps):
                mtp_predict = torch.cat(
                    (hidden_states[:, :1, ...], mtp_predict[:, :-1, ...]), dim=1
                )
                mtp_predict = model(
                    mtp_predict,
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn,
                    image_mask=image_mask_t,
                )
                mtp_predicts.append(mtp_predict)
            mtp_predicts = torch.cat(mtp_predicts, dim=0)

            with torch.no_grad():
                target_head = head(target)
                target_head = target_head.expand(
                    [args.mtp_steps + 1] + list(target_head.shape)
                ).flatten(0, 1)
                target_p = nn.Softmax(dim=2)(target_head).detach()
            loss_mask = loss_mask_t[:, :, None]
            loss_mask = loss_mask.expand(
                [args.mtp_steps + 1] + list(loss_mask.shape)
            ).flatten(0, 1)
            loss, out_head = compute_loss(target_p, mtp_predicts, loss_mask, 10)

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
            if ct != 0:
                out_head_f = out_head.view(-1, target_head.shape[-1])[
                    loss_mask.reshape(-1) == 1
                ]
                target_ids_f = target_ids.view(-1)[loss_mask.reshape(-1) == 1]
                topkacc = top_accuracy(out_head_f, target_ids_f, (1, 2, 3))
                for top_i in range(len(topkacc)):
                    top_3acc[top_i] += topkacc[top_i]
                total += ct
                correct += cc
        if accelerator.is_main_process and ct != 0:
            logdict = {
                "train/lr": optimizer.optimizer.param_groups[0]["lr"],
                "train/loss": loss.item(),
                "train/acc": cc / ct,
            }
            writer.add_scalars("train", logdict, epoch * len(train_loader) + batch_idx)

        epoch_loss += loss.item() if not loss.isnan() else 0
        num_batches += 1

    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    epoch_loss /= max(num_batches, 1)
    if accelerator.is_main_process:
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, epoch_loss))
        print("Train Accuracy: {:.2f}%".format(100 * correct / max(total, 1)))
        writer.add_scalars(
            "train_epoch",
            {"train/epochacc": correct / max(total, 1), "train/epochloss": epoch_loss},
            epoch,
        )

    if epoch % train_config["save_freq"] == 0 or epoch == num_epochs:
        correct = 0
        total = 0
        epoch_loss = 0
        num_batches = 0
        model.eval()
        for batch_idx, batch in enumerate(
            tqdm(test_loader, disable=not accelerator.is_local_main_process)
        ):
            with torch.no_grad():
                (
                    hidden_states,
                    inputs_embeds,
                    target,
                    full_ids,
                    loss_mask_t,
                    image_mask_t,
                ) = build_online_mm_tensors(
                    target_model, batch, accelerator.device, add_noise=False
                )
                attn = torch.ones(
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    device=hidden_states.device,
                )
                predict = model(
                    hidden_states,
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn,
                    image_mask=image_mask_t,
                )
                target_head = head(target)
                target_p = nn.Softmax(dim=2)(target_head).detach()
                loss_mask = loss_mask_t[:, :, None]
                loss, out_head = compute_loss(target_p, predict, loss_mask, 10)
                _, predicted = torch.max(out_head, 2)
                _, target_ids = torch.max(target_head, 2)
                ct = loss_mask.sum().item()
                cc = ((predicted == target_ids) * loss_mask.squeeze()).sum().item()
                total += ct
                correct += cc
            epoch_loss += loss.item() if not loss.isnan() else 0
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
