#!/usr/bin/env bash
# One-shot setup + zero-shot test on a fresh GPU box. Run inside tmux:
#   tmux new -s jx      (reattach later with: tmux attach -t jx)
#   bash ~/floorplan_env/gpu_setup.sh
# Everything is logged to ~/setup.log. At the end it prints the zero-shot result.
set -euo pipefail
exec > >(tee -a ~/setup.log) 2>&1
cd ~/floorplan_env

echo "== 1/5 python packages"
pip install -q --upgrade pip
pip install -q vllm openai pillow shapely opencv-python-headless scikit-image numpy ezdxf pymupdf \
               "trl>=0.20" peft accelerate datasets qwen-vl-utils
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "== 2/5 full dataset (background, ~10 GB)"
mkdir -p ~/data
( cd ~/data && wget -q -c "https://fwjydgxpttajiqpfmqtj.supabase.co/storage/v1/object/public/stoa-junction-2026/junction-dataset.zip" \
  && unzip -q -o junction-dataset.zip && echo "DATASET READY" > ~/data/READY ) > ~/data/download.log 2>&1 &

echo "== 3/5 start vLLM with Qwen2.5-VL-7B (downloads ~16 GB the first time)"
nohup vllm serve Qwen/Qwen2.5-VL-7B-Instruct --max-model-len 32768 --limit-mm-per-prompt '{"image":1}' \
      --gpu-memory-utilization 0.85 > ~/vllm.log 2>&1 &
for i in $(seq 1 120); do
  if curl -s localhost:8000/v1/models > /dev/null; then echo "vLLM up after $((i*10)) s"; break; fi
  sleep 10
done
curl -s localhost:8000/v1/models > /dev/null || { echo "vLLM did not start, last log lines:"; tail -40 ~/vllm.log; exit 1; }

echo "== 4/5 zero-shot eval: 15 test plans x 4 samples"
export IMAGES_DIR=~/junction-dataset-preview/data/swiss_dwellings
export QUESTIONS=~/junction-dataset-preview/questions.jsonl
python3 eval_policy.py --model Qwen/Qwen2.5-VL-7B-Instruct --split test --samples 4 --tag qwen7b_zeroshot \
        | tee ~/zeroshot.txt

echo "== 5/5 summary"
SNAP=1 python3 summarize.py results/policy_qwen7b_zeroshot | tee -a ~/zeroshot.txt
tail -3 ~/data/download.log 2>/dev/null; ls ~/data
echo "DONE. Paste the lines above starting from '== 4/5'."
