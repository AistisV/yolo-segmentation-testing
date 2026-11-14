"""GRPO on the FloorPlan environment: reward = env score of the generated schematic (snap on).
Start from the SFT checkpoint (or the base model). Untested on GPU yet: expect small fixes.

    pip install "trl>=0.20" peft transformers accelerate datasets vllm
    python train_grpo.py --model ckpt/sft-merged --out ckpt/grpo --steps 150

Notes
- Long outputs (2-9k tokens): keep num_generations small (4) and use vLLM generation (colocate).
- Reward integrity: questions with a zero true answer are not rewarded (empty schematic = 0), and
  item F1 uses one-to-one matching, so spamming items or rooms does not pay.
"""
import argparse, json, os, re
from PIL import Image
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from env import FloorPlanEnv
from qwen_grid import from_norm1000

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--out", default="ckpt/grpo")
ap.add_argument("--split", default="train")
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--gens", type=int, default=4)
ap.add_argument("--max-completion", type=int, default=8192)
ap.add_argument("--lr", type=float, default=5e-6)
ap.add_argument("--max-plans", type=int, default=0, help="0 = split.json train; N = up to N plans from GT_DIR (test excluded)")
a = ap.parse_args()

CFG = dict(images_dir=os.environ.get("IMAGES_DIR", "data/swiss_dwellings"), gt_dir=os.environ.get("GT_DIR", "gt"),
           questions_path=os.environ.get("QUESTIONS", "data/questions.jsonl"), snap=True)
PROMPT = open("policy_prompt.md", encoding="utf-8").read()
plans = json.load(open("split.json"))[a.split]
if a.max_plans:
    import glob, random
    test = set(json.load(open("split.json"))["test"])
    allp = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(CFG["gt_dir"], "*.json")))
    allp = [p for p in allp if p not in test and os.path.exists(os.path.join(CFG["images_dir"], p + ".png"))]
    random.Random(0).shuffle(allp); plans = allp[:a.max_plans]
print(len(plans), "training plans")

rows, ENVS = [], {}
for p in plans:
    env = ENVS[p] = FloorPlanEnv(p, CFG)
    W, H = env.img.size
    rows.append({"plan": p, "image": env.img,
                 "prompt": [{"role": "user", "content": [{"type": "image"},
                             {"type": "text", "text": PROMPT.replace("{W}", str(W)).replace("{H}", str(H))}]}]})
ds = Dataset.from_list(rows)


def parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def env_reward(completions, plan, **kw):
    out = []
    for c, p in zip(completions, plan):
        text = c[0]["content"] if isinstance(c, list) else c
        s = parse_json(text)
        if not isinstance(s, dict):
            out.append(0.0); continue
        env = ENVS[p]; W, H = env.img.size
        try:
            out.append(float(env.score(from_norm1000(s, W, H))["reward"]))
        except Exception:
            out.append(0.0)
    return out


cfg = GRPOConfig(output_dir=a.out, max_steps=a.steps, learning_rate=a.lr, per_device_train_batch_size=a.gens,
                 gradient_accumulation_steps=2, num_generations=a.gens, max_prompt_length=None,
                 max_completion_length=a.max_completion, temperature=1.0, beta=0.0, bf16=True,
                 gradient_checkpointing=True, logging_steps=1, save_steps=25, report_to="none",
                 use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.35)
peft = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
trainer = GRPOTrainer(model=a.model, reward_funcs=[env_reward], args=cfg, train_dataset=ds, peft_config=peft)
trainer.train()
trainer.save_model(a.out)
print("saved", a.out)
