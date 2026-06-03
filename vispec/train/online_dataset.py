"""
ViSpec 在线训练数据集
==============================================================================
与离线训练的根本区别:
  - 离线:Dataset 直接 torch.load 预存的 hidden_state / inputs_embeds(目标模型
          已提前算好并落盘),训练时不碰目标模型。
  - 在线:Dataset 只返回「原始输入」(input_ids / loss_mask,Stage 2 额外含图像),
          目标模型在训练循环里实时 forward 算 hidden_state / inputs_embeds。

本模块提供两个 Dataset:
  - OnlineTextDataset  : Stage 1(纯文本,ShareGPT),实时构造 input_ids + loss_mask
  - OnlineMMDataset    : Stage 2(多模态),读「预生成的长回复 token + 图片」,
                          现场用 processor 构造目标模型输入

两个阶段共用的下游(目标模型 forward → 右移构造 target → 草稿 forward → loss)
都在各自的 main_*_online.py 训练循环里完成,与离线版逻辑保持一致。
==============================================================================
"""

import json
import os

import torch
from torch.utils.data import Dataset


# ============================================================================
# Stage 1: 纯文本在线数据集
#   预处理逻辑(input_ids / loss_mask 构造)严格复用
#   ge_data/ge_data_all_qwen_shargpt.py 的 build_dataset_rank,保证与离线一致。
# ============================================================================
class OnlineTextDataset(Dataset):
    """Stage 1 在线数据集:返回 input_ids + loss_mask,不含任何 hidden state。

    目标模型 forward 在训练循环里做。
    """

    def __init__(self, hf_dataset, max_len=4096):
        # hf_dataset: 已经过 build_dataset_rank 预处理的 datasets.Dataset
        #             (含 "input_ids" / "loss_mask" 两列,torch 格式)
        self.data = hf_dataset
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        # build_dataset_rank 里存的是 [1, seq] 形状
        input_ids = item["input_ids"]
        loss_mask = item["loss_mask"]
        if input_ids.dim() == 2:
            input_ids = input_ids[0]
        if loss_mask.dim() == 2:
            loss_mask = loss_mask[0]

        input_ids = input_ids[: self.max_len]
        loss_mask = loss_mask[: self.max_len].tolist()

        return {
            "input_ids": input_ids,          # [seq]  (long)
            "loss_mask": loss_mask,          # list[int], 长度 seq
        }


# ============================================================================
# Stage 2: 多模态在线数据集
#   读「预生成长回复」数据(gen_stage2_responses.py 产出的轻量 token 文件),
#   每条含: prompt 文本 / 生成的长回复 token / 图片路径。
#   训练循环里用 processor 拼成目标模型输入,再 forward。
# ============================================================================
class OnlineMMDataset(Dataset):
    """Stage 2 在线数据集:返回 processor 处理后的目标模型输入 + loss_mask + image_mask。

    每条预生成文件(.pt)应包含:
      - "full_ids":   完整 token 序列(prompt + 目标模型生成的长回复)  [seq] long
      - "prompt_len": prompt 部分长度(用于构造 loss_mask)              int
      - "image_file": 图片相对/绝对路径                                  str
    其中 full_ids 是离线用目标模型 generate 出来的,训练时只需对它 forward。
    """

    def __init__(self, samples, processor, image_token_id, max_len=4096):
        # samples: list[str],每个是预生成 .pt 文件路径
        self.samples = samples
        self.processor = processor
        self.image_token_id = image_token_id
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rec = torch.load(self.samples[index], map_location="cpu")
        full_ids = rec["full_ids"]
        if full_ids.dim() == 2:
            full_ids = full_ids[0]
        full_ids = full_ids[: self.max_len]
        prompt_len = int(rec["prompt_len"])
        image_file = rec["image_file"]

        # loss_mask: prompt 部分(含图像 token)mask 0,生成部分 mask 1
        seq = full_ids.shape[0]
        loss_mask = [0] * min(prompt_len, seq) + [1] * max(0, seq - prompt_len)
        loss_mask = loss_mask[:seq]

        # image_mask: 标记图像 token 位置(供 ImgAdaptor 定位)
        image_mask = (full_ids == self.image_token_id).tolist()

        return {
            "full_ids": full_ids,            # [seq]  long
            "loss_mask": loss_mask,          # list[int]
            "image_mask": image_mask,        # list[bool]
            "image_file": image_file,        # str(训练循环里 Image.open + processor 取 pixel_values)
            "prompt_len": prompt_len,
        }


# ============================================================================
# 文件收集工具(预生成长回复目录)
# ============================================================================
def list_response_files(path, suffix=".pt"):
    files = []
    # followlinks=True: 允许 DATAPATH 用符号链接拼装多份数据(如 60K 主 + 8K 评测 mix)
    for root, _dirs, fnames in os.walk(path, followlinks=True):
        for fn in fnames:
            if fn.endswith(suffix):
                files.append(os.path.join(root, fn))
    return sorted(files)
