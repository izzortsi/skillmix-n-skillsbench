# vast.ai docker setup for skillmix-n-skillsbench

End-to-end recipe to test the v1.9 setup (Qwen3.5-2B + LoRA on procedural-skill SFT) on a larger model (4B / 9B) using a vast.ai GPU rental.

## what's in this folder

- `Dockerfile` — base image: pytorch/pytorch 2.5.1 + CUDA 12.1, with all training deps pre-installed and the skillmix-n-skillsbench repo baked in at `/workspace/skillmix-n-skillsbench`.
- `requirements.txt` — pinned floors for transformers/trl/peft/bitsandbytes/etc.
- `../.dockerignore` — keeps weights, probe logs, and caches out of the build context.

## one-time: build and push

From the repo root:

```bash
# pick a tag — your dockerhub user, or ghcr.io/<user>
export IMG=docker.io/<dockerhub-user>/skillmix-skillsbench:latest

docker build -t "$IMG" -f docker/Dockerfile .
docker push "$IMG"
```

Build size is ~10-12 GB. First push takes a while; subsequent pushes only ship the changed layers.

If you don't have docker hub, alternatives:
- ghcr.io: `docker login ghcr.io -u <gh-user>` then tag as `ghcr.io/<user>/...`
- vast.ai also accepts any HTTPS-reachable registry.

## launching on vast.ai

1. Go to https://vast.ai/console/create/
2. Filter by GPU/VRAM (see "GPU sizing" below).
3. Click **Edit Image & Config** on a chosen offer.
4. Set:
   - **Image Path/Tag**: `docker.io/<dockerhub-user>/skillmix-skillsbench:latest`
   - **Launch Mode**: SSH (or Jupyter — SSH is simpler for tmux + training)
   - **Disk Space**: 60 GB minimum (HF caches + checkpoints add up). 100 GB if running 9B.
5. Add **Environment Variables**:
   - `ANTHROPIC_API_KEY=sk-ant-...` (required for the Opus judge during eval)
   - `HF_TOKEN=hf_...` (only if pulling gated models)
6. Click **Rent**.
7. SSH in (vast.ai shows the command in the instance dashboard).

## first-shell sanity check

Once SSH'd in:

```bash
cd /workspace/skillmix-n-skillsbench

# verify GPU + CUDA
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# verify deps loadable
python -c "import transformers, peft, trl, bitsandbytes; print('ok')"

# verify the v1 SFT corpus is present
wc -l data/pipeline-runs/default/sft_dataset/dataset.jsonl
```

## v2.0 run on a larger model

The v1.9 commands transposed to a larger base. Use `tmux` so the training survives an SSH disconnect.

```bash
tmux new -s train

# 1. patch the chat template for the chosen base
python training/patch_qwen35_template.py \
    --model Qwen/Qwen3.5-4B \
    --out ./qwen35-4b-tokenizer-patched

# 2. train. r=32 / alpha=64 gives the larger base more LoRA capacity;
#    drop max_seq_length to 2048 if VRAM is tight.
python training/train_qwen35_lora.py \
    --data data/pipeline-runs/default/sft_dataset/dataset.jsonl \
    --model Qwen/Qwen3.5-4B \
    --tokenizer-path ./qwen35-4b-tokenizer-patched \
    --output-dir ./qwen35-4b-skill-lora-v2.0 \
    --mode lora \
    --r 32 --alpha 64 \
    --epochs 3 --lr 2e-4

# 3. eval against the same 200-task holdout
python training/eval_qwen35_lora.py \
    --base Qwen/Qwen3.5-4B \
    --tokenizer-path ./qwen35-4b-tokenizer-patched \
    --adapter ./qwen35-4b-skill-lora-v2.0 \
    --tasks data/pipeline-runs/default/synthesis/tasks_eval.json \
    --catalog data/pipeline-runs/default/bench_catalog/skills.json \
    --task-skill-map data/pipeline-runs/default/synthesis/task_skill_map_eval.json \
    --out-dir data/pipeline-runs/default/bench-eval-post-sft-v2_0
```

Detach with `Ctrl-b d`. Reattach later with `tmux a -t train`.

## GPU sizing

| Base                | Recipe                | VRAM (steady) | First-iter spike | Recommended GPU |
|---------------------|-----------------------|---------------|------------------|-----------------|
| Qwen3.5-2B          | LoRA r=16             | ~7-9 GB       | ~12 GB           | RTX 3090/4090 (24 GB) |
| Qwen3.5-4B          | LoRA r=32             | ~14-17 GB     | ~22 GB           | RTX 3090/4090 (24 GB) |
| Qwen3.5-9B          | LoRA r=32             | ~24-28 GB     | ~38 GB           | A100 40GB / RTX 6000 Ada |
| Qwen3.5-9B          | LoRA r=16, max_seq=2k | ~20-24 GB     | ~32 GB           | A100 40GB |

Notes:
- `bitsandbytes` 8-bit Adam (`--optim adamw_bnb_8bit`) is in the script and saves 1-2 GB.
- If still OOM on 9B at 40GB, add `--max-seq-length 2048` (halves the logits buffer, the dominant cost).
- Vast.ai pricing as of last check: 4090 ~$0.30-0.50/hr, A100 40GB ~$0.80-1.50/hr, A100 80GB ~$1.50-2.50/hr.

## getting results back

Outputs of interest after a run:

```bash
# the trained adapter (small, ~50-300 MB)
ls -lh ./qwen35-4b-skill-lora-v2.0/adapter_model.safetensors

# the eval episodes
ls -lh data/pipeline-runs/default/bench-eval-post-sft-v2_0/

# rsync down to your local box
rsync -avzP \
    -e 'ssh -p <vast-ssh-port>' \
    root@<vast-ip>:/workspace/skillmix-n-skillsbench/qwen35-4b-skill-lora-v2.0 \
    ./
rsync -avzP \
    -e 'ssh -p <vast-ssh-port>' \
    root@<vast-ip>:/workspace/skillmix-n-skillsbench/data/pipeline-runs/default/bench-eval-post-sft-v2_0 \
    ./data/pipeline-runs/default/
```

## destroying the instance

Don't forget — vast.ai bills by the second while running. Stop the instance from the dashboard once results are downloaded.

## iterating on the image

Code-only changes don't need a rebuild — just `git pull` inside the container. Rebuild + repush only when:
- Adding/removing a Python dep (`requirements.txt`)
- Changing the base CUDA / torch version
- Adding system packages

## known gotchas

- **bitsandbytes CUDA mismatch**: if `import bitsandbytes` fails on the rented GPU, the host's CUDA driver is older than 12.1. Either pick a different offer or downgrade the base to `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel`.
- **HF rate limits on first download**: pulling Qwen3.5-9B can hit anonymous rate limits. Set `HF_TOKEN` to use authenticated downloads.
- **Disk fills up mid-training**: Qwen weights + checkpoints + caches add up fast. Monitor with `df -h`. The 60 GB default vast.ai disk is fine for 4B; bump to 100+ GB for 9B.
