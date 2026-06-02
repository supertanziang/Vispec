"""
====================================================================================
ViSpec 端到端速度评测脚本 (统一入口)
====================================================================================

一个脚本搞定:多模型 × 多 benchmark × 多温度 的 baseline + spec 评测,
最后汇总成一张表格,输出每个组合的:
    - τ (acceptance length, 接收长度): draft 每轮平均被目标模型接受的 token 数
    - ratio (speedup, 加速比):       spec 吞吐 / baseline 吞吐

用法:
    # 用脚本顶部 CONFIG 的默认配置直接跑
    python -m vispec.evaluation.run_eval

    # 命令行覆盖(逗号分隔):
    python -m vispec.evaluation.run_eval --models qwen --benchmarks mmvet,coco_caption --temperatures 0.0
    python -m vispec.evaluation.run_eval --only-summary          # 跳过推理,只读已有输出汇总表格
    python -m vispec.evaluation.run_eval --skip-baseline         # 只跑 spec(已有 baseline 时)
    python -m vispec.evaluation.run_eval --gpu 0                 # 指定 GPU

输出路径(baseline 与 spec 结果统一在 results/ 下,各占一个子目录):
    baseline -> results/baseline/{bench}_test/baseline_{model}/test-temperature-{t}.jsonl
    spec     -> results/spec/{bench}_test/{model}_{method}/test-temperature-{t}.jsonl
====================================================================================
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from transformers import AutoTokenizer

# ===================================================================================
# CONFIG —— 所有可配置项集中在这里
# ===================================================================================

# ---- 1. 默认要跑的组合(可被命令行覆盖)---------------------------------------------
DEFAULT_MODELS = ["qwen"]                               # 见下方 MODELS 注册表的 key
DEFAULT_BENCHMARKS = ["mmvet", "sqa", "mme"]            # 见下方 BENCHMARKS 注册表
DEFAULT_TEMPERATURES = [0.0]                             # 论文主表用 0.0 与 1.0 两档

# ---- 2. ViSpec 投机解码超参(spec 推理用)-----------------------------------------
#   depth:        草稿 token 树的深度(猜多少步)
#   top_k:        树的宽度(每步保留几个候选)
#   total_token:  从树里最终选多少 token 交给目标模型验证
#   num_q:        ImgAdaptor 的 query 向量数,必须与训练时一致(2q 档=2)
#   method_tag:   输出目录里的方法名标签,speed.py 用 "2q" 对应 num_q=2
# 支持用环境变量 VISPEC_DEPTH / VISPEC_TOP_K / VISPEC_TOTAL_TOKEN / VISPEC_NUM_Q 覆盖
# (evaluation.sh 即通过这些环境变量传入)
_num_q = int(os.environ.get("VISPEC_NUM_Q", 2))
SPEC_HPARAMS = {
    "depth": int(os.environ.get("VISPEC_DEPTH", 3)),
    "top_k": int(os.environ.get("VISPEC_TOP_K", 8)),
    "total_token": int(os.environ.get("VISPEC_TOTAL_TOKEN", 30)),
    "num_q": _num_q,
    "method_tag": f"{_num_q}q",
}

# ---- 3. 其它推理参数 -------------------------------------------------------------
MAX_NEW_TOKEN = int(os.environ.get("VISPEC_MAX_NEW_TOKEN", 1024))   # 单条样本最多生成多少 token
GPU = "0"                   # 默认用哪块 GPU(单卡评测)

# ---- 4. 输出根目录:baseline 与 spec 统一收纳到 results/ 下,各占一个子目录 ----------
BASELINE_DIR = "results/baseline"
RESULT_DIR = "results/spec"

# ---- 5. 模型注册表 ---------------------------------------------------------------
#   key            : 简称,同时作为输出路径里的 {model} 段(须与 speed.py 的循环一致)
#   base_path      : 目标 VLM 路径(本地目录 或 HF repo id)
#   spec_path      : 对应的 ViSpec draft 权重路径(本地 或 HF repo id)
#   family         : "qwen" 或 "llava",决定调用哪一套 gen_*_<family> 评测脚本
#   tokenizer_path : 汇总 baseline token 数时用的 tokenizer(应与目标模型同族)
MODELS = {
    "qwen": {
        "base_path": "model/Qwen2.5-VL-7B-Instruct",
        "spec_path": "model/ViSpec-Qwen2.5-VL-7B-Instruct",
        "family": "qwen",
        "tokenizer_path": "model/Qwen2.5-VL-7B-Instruct",
    },
    "qwen_3b": {
        "base_path": "model/Qwen2.5-VL-3B-Instruct",
        "spec_path": "model/ViSpec-Qwen2.5-VL-3B-Instruct",
        "family": "qwen",
        "tokenizer_path": "model/Qwen2.5-VL-3B-Instruct",
    },
    "llava": {
        "base_path": "model/llava-v1.6-vicuna-7b-hf",
        "spec_path": "model/ViSpec-llava-v1.6-vicuna-7b-hf",
        "family": "llava",
        "tokenizer_path": "model/llava-v1.6-vicuna-7b-hf",
    },
    "llava_13b": {
        "base_path": "model/llava-v1.6-vicuna-13b-hf",
        "spec_path": "model/ViSpec-llava-v1.6-vicuna-13b-hf",
        "family": "llava",
        "tokenizer_path": "model/llava-v1.6-vicuna-13b-hf",
    },
    "llava_1.5": {
        "base_path": "model/llava-1.5-7b-hf",
        "spec_path": "model/ViSpec-llava-1.5-7b-hf",
        "family": "llava",
        "tokenizer_path": "model/llava-1.5-7b-hf",
    },
}

# ---- 6. Benchmark 注册表 ----------------------------------------------------------
#   key          : 简称,作为输出路径里的 {bench}_test 段(须与 speed.py 循环一致)
#   data_folder  : 本地数据目录,会作为 --data-folder 传给 gen_*_<bench>.py
#                  统一约定:数据放在 data/eval/<name>/ 下(load_from_disk 可读)
#   extra_args   : 该 benchmark 特有的命令行参数(如 sqa)
BENCHMARKS = {
    # —— 本地 datasets(load_from_disk),数据在 data/eval/<name> 下 ——
    "mmvet": {"data_folder": "data/eval/mmvet", "extra_args": []},
    "coco_caption": {"data_folder": None, "extra_args": []},
    # HR-Bench 是 4K/8K 高分辨率基准,图像 token 极多,单卡 80G 会 OOM(attention 矩阵爆炸)。
    # 需限制 processor 的 max_pixels 或多卡分布才能跑,默认不放进 DEFAULT_BENCHMARKS。
    "hr_bench": {"data_folder": None, "extra_args": []},
    # —— 本地数据 ——
    "mme": {"data_folder": "data/eval/MME", "extra_args": []},
    "gqa": {"data_folder": "data/eval/gqa", "extra_args": []},
    "textvqa": {"data_folder": "data/eval/textvqa", "extra_args": []},
    "vqav2": {"data_folder": "data/eval/vqav2", "extra_args": []},
    "seed_bench": {"data_folder": "data/eval/seed_bench", "extra_args": []},
    "vizwiz": {"data_folder": "data/eval/vizwiz", "extra_args": []},
    # —— sqa: 数据(图片)在 data/eval/sqa/dataset, metadata 在 data/eval/sqa/metadata ——
    # test_number=100 与其他 benchmark 对齐(设为 -1 则跑全部 4241 条,约 6 小时)
    "sqa": {
        "data_folder": "data/eval/sqa/dataset",
        "extra_args": [
            "--test_split=test",
            "--test_number=100",
            "--shot_number=0",
            "--prompt_format=QCM-ALE",
            "--data_root=data/eval/sqa/metadata",
            "--caption_file=data/eval/sqa/metadata/captions.json",
        ],
    },
}

# HF token(若下载受限流可在此填,留空则不传)
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ===================================================================================
# 以下为执行逻辑,一般无需修改
# ===================================================================================


def out_paths(model, bench, method_tag, temp):
    """返回 (baseline_bench_name 目录, spec_bench_name 目录, baseline_jsonl, spec_jsonl)"""
    base_dir = f"{BASELINE_DIR}/{bench}_test/baseline_{model}"
    spec_dir = f"{RESULT_DIR}/{bench}_test/{model}_{method_tag}"
    # 评测脚本会把 model_id="test" 自动拼成 test-temperature-{t},再 +".jsonl"
    base_jsonl = f"{base_dir}/test-temperature-{temp:.1f}.jsonl"
    spec_jsonl = f"{spec_dir}/test-temperature-{temp:.1f}.jsonl"
    return base_dir, spec_dir, base_jsonl, spec_jsonl


def run_one(mode, model, bench, temp, gpu, force=False):
    """跑单个 (mode, model, bench, temp);mode ∈ {'baseline','spec'}。返回输出 jsonl 路径。"""
    m = MODELS[model]
    b = BENCHMARKS[bench]
    family = m["family"]
    method_tag = SPEC_HPARAMS["method_tag"]
    base_dir, spec_dir, base_jsonl, spec_jsonl = out_paths(model, bench, method_tag, temp)

    target_jsonl = base_jsonl if mode == "baseline" else spec_jsonl
    bench_name = base_dir if mode == "baseline" else spec_dir

    # 已存在且行数>0 则跳过(除非 force)
    if not force and os.path.exists(target_jsonl) and os.path.getsize(target_jsonl) > 0:
        print(f"  [skip] {mode} 已存在: {target_jsonl}")
        return target_jsonl

    # 重跑前清掉旧文件(评测脚本是 append 写入)
    if os.path.exists(target_jsonl):
        os.remove(target_jsonl)

    script = f"vispec.evaluation.gen_{'baseline' if mode=='baseline' else 'spec'}_answer_{bench}"
    cmd = [
        sys.executable, "-m", script,
        "--base-model-path", m["base_path"],
        "--spec-model-path", m["spec_path"],
        "--model-id", "test",
        "--bench-name", bench_name,
        "--temperature", str(temp),
        "--max-new-token", str(MAX_NEW_TOKEN),
    ]
    if b["data_folder"] is not None:
        cmd += ["--data-folder", b["data_folder"]]
    cmd += b["extra_args"]
    if mode == "spec":
        cmd += [
            "--use-ours", "True",
            "--num-q", str(SPEC_HPARAMS["num_q"]),
            "--depth", str(SPEC_HPARAMS["depth"]),
            "--top-k", str(SPEC_HPARAMS["top_k"]),
            "--total-token", str(SPEC_HPARAMS["total_token"]),
        ]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if HF_TOKEN:
        env["HF_TOKEN"] = HF_TOKEN

    print(f"  [run ] {mode} {model}/{bench}/T={temp} (family={family})")
    print("        " + " ".join(cmd))
    ret = subprocess.run(cmd, env=env)
    if ret.returncode != 0:
        print(f"  [FAIL] {mode} {model}/{bench}/T={temp} 退出码={ret.returncode}")
        return None
    return target_jsonl


def summarize(models, benchmarks, temperatures):
    """读取所有输出,计算 τ 和 ratio,打印汇总表。"""
    method_tag = SPEC_HPARAMS["method_tag"]
    rows = []
    # 按模型分组缓存 tokenizer
    tok_cache = {}
    for model in models:
        m = MODELS[model]
        if model not in tok_cache:
            try:
                tok_cache[model] = AutoTokenizer.from_pretrained(m["tokenizer_path"])
            except Exception as e:
                print(f"[warn] {model} tokenizer 加载失败({m['tokenizer_path']}): {e}")
                tok_cache[model] = None
        tokenizer = tok_cache[model]

        for bench in benchmarks:
            for temp in temperatures:
                _, _, base_jsonl, spec_jsonl = out_paths(model, bench, method_tag, temp)
                if not (os.path.exists(spec_jsonl) and os.path.exists(base_jsonl)):
                    continue

                # spec: τ 与吞吐
                speeds, acc_len, new_toks = [], [], []
                with open(spec_jsonl, encoding="utf-8") as f:
                    for line in f:
                        dp = json.loads(line)
                        c = dp["choices"][0]
                        tokens = sum(c["new_tokens"])
                        times = sum(c["wall_time"])
                        acc_len += c["acceptance_length"]
                        speeds.append(tokens / times)
                        new_toks.append(tokens)
                if not acc_len:
                    continue
                tau = sum(acc_len) / len(acc_len)
                spec_speed = float(np.mean(speeds))

                # baseline: 吞吐(从 turns 文本重新 tokenize)
                speeds0 = []
                with open(base_jsonl, encoding="utf-8") as f:
                    for line in f:
                        dp = json.loads(line)
                        answer = dp["choices"][0]["turns"]
                        if tokenizer is not None:
                            tokens = sum(len(tokenizer(i).input_ids) - 1 for i in answer)
                        else:
                            tokens = sum(len(i.split()) for i in answer)  # 兜底
                        times = sum(dp["choices"][0]["wall_time"])
                        speeds0.append(tokens / times)
                base_speed = float(np.mean(speeds0))
                ratio = spec_speed / base_speed if base_speed else float("nan")

                rows.append((model, bench, temp, tau, ratio, spec_speed, base_speed,
                             sum(new_toks) / len(new_toks)))

    # 打印表格
    print("\n" + "=" * 96)
    print("ViSpec 评测汇总  (τ=平均接收长度, ratio=加速比)")
    print("=" * 96)
    hdr = f"{'model':<12}{'benchmark':<16}{'T':<6}{'τ':<10}{'ratio':<10}{'spec tok/s':<14}{'base tok/s':<14}{'avg new tok':<12}"
    print(hdr)
    print("-" * 96)
    if not rows:
        print("(无可汇总结果:请先跑出 baseline 和 spec 的输出文件)")
    for r in rows:
        model, bench, temp, tau, ratio, ss, bs, nt = r
        print(f"{model:<12}{bench:<16}{temp:<6.1f}{tau:<10.3f}{ratio:<10.3f}{ss:<14.2f}{bs:<14.2f}{nt:<12.1f}")
    print("=" * 96)


def parse_list(s, default):
    if s is None:
        return default
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="ViSpec 统一速度评测")
    ap.add_argument("--models", type=str, default=None,
                    help=f"逗号分隔,可选 {list(MODELS)};默认 {DEFAULT_MODELS}")
    ap.add_argument("--benchmarks", type=str, default=None,
                    help=f"逗号分隔,可选 {list(BENCHMARKS)};默认 {DEFAULT_BENCHMARKS}")
    ap.add_argument("--temperatures", type=str, default=None,
                    help=f"逗号分隔;默认 {DEFAULT_TEMPERATURES}")
    ap.add_argument("--gpu", type=str, default=GPU, help=f"用哪块 GPU,默认 {GPU}")
    ap.add_argument("--skip-baseline", action="store_true", help="不跑 baseline(已有时)")
    ap.add_argument("--skip-spec", action="store_true", help="不跑 spec")
    ap.add_argument("--only-summary", action="store_true", help="跳过推理,只汇总已有输出")
    ap.add_argument("--force", action="store_true", help="即使输出已存在也重跑")
    args = ap.parse_args()

    models = parse_list(args.models, DEFAULT_MODELS)
    benchmarks = parse_list(args.benchmarks, DEFAULT_BENCHMARKS)
    temperatures = [float(x) for x in parse_list(args.temperatures, [str(t) for t in DEFAULT_TEMPERATURES])]

    # 校验
    for m in models:
        assert m in MODELS, f"未知 model: {m},可选 {list(MODELS)}"
    for b in benchmarks:
        assert b in BENCHMARKS, f"未知 benchmark: {b},可选 {list(BENCHMARKS)}"

    print(f"models      = {models}")
    print(f"benchmarks  = {benchmarks}")
    print(f"temperatures= {temperatures}")
    print(f"spec hparams= {SPEC_HPARAMS}")
    print(f"gpu         = {args.gpu}")

    if not args.only_summary:
        for model in models:
            for bench in benchmarks:
                for temp in temperatures:
                    print(f"\n>>> {model} | {bench} | T={temp}")
                    if not args.skip_baseline:
                        run_one("baseline", model, bench, temp, args.gpu, force=args.force)
                    if not args.skip_spec:
                        run_one("spec", model, bench, temp, args.gpu, force=args.force)

    summarize(models, benchmarks, temperatures)


if __name__ == "__main__":
    main()
