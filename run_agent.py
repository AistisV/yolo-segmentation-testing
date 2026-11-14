"""Run any OpenAI-compatible vision model as an agent in FloorPlanEnv.

GPT-5 mini (baseline):
    export OPENAI_API_KEY=...
    python run_agent.py --model gpt-5-mini --split test --runs 3

Gemini (OpenAI-compatible endpoint):
    export GEMINI_API_KEY=...
    python run_agent.py --model gemini-3.8-flash --api-key-env GEMINI_API_KEY \
        --base-url https://generativelanguage.googleapis.com/v1beta/openai/ --split test

Our small model served with vLLM (same code, OpenAI-compatible endpoint):
    vllm serve Qwen/Qwen2.5-VL-7B-Instruct --enable-auto-tool-choice --tool-call-parser hermes
    python run_agent.py --model Qwen/Qwen2.5-VL-7B-Instruct --base-url http://localhost:8000/v1 --split test

Writes results/<model>/<plan>_run<k>.json (schematic + per-question scoring) and prints a summary.
"""
import argparse, base64, io, json, os, re, time
from openai import OpenAI
from env import FloorPlanEnv, TOOLS_SPEC

CFG = dict(images_dir=os.environ.get("IMAGES_DIR", "data/swiss_dwellings"),
           gt_dir=os.environ.get("GT_DIR", "gt"),
           questions_path=os.environ.get("QUESTIONS", "data/questions.jsonl"))

def img_url_from_pil(img, max_side=2048):
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)))
    buf = io.BytesIO(); img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def extract_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}

def run_episode(client, model, plan, max_turns=15, max_side=2048):
    env = FloorPlanEnv(plan, CFG)
    obs = env.reset()
    W, H = obs["image_size"]
    shown = env.img if max(env.img.size) <= max_side else env.img
    note = ""
    if max(env.img.size) > max_side:
        s = max_side / max(env.img.size)
        note = (f" The overview image you see is downscaled by {s:.3f}; multiply its coordinates by {1/s:.3f} "
                f"to get original pixels, or use crop for exact detail.")
    tools = [{"type": "function", "function": t} for t in TOOLS_SPEC]
    messages = [
        {"role": "system", "content": obs["instructions"]},
        {"role": "user", "content": [
            {"type": "text", "text": f"Floor plan image, original size {W}x{H} pixels.{note} "
                                     f"Use the tools, then call submit_schematic."},
            {"type": "image_url", "image_url": {"url": img_url_from_pil(shown, max_side)}}]},
    ]
    schematic, usage = None, {"prompt_tokens": 0, "completion_tokens": 0}
    for turn in range(max_turns):
        kw = {}
        if turn >= max_turns - 1:   # last turn: force submission
            kw["tool_choice"] = {"type": "function", "function": {"name": "submit_schematic"}}
        resp = client.chat.completions.create(model=model, messages=messages, tools=tools, **kw)
        if resp.usage:
            usage["prompt_tokens"] += resp.usage.prompt_tokens or 0
            usage["completion_tokens"] += resp.usage.completion_tokens or 0
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            schematic = extract_json(msg.content)
            break
        extra_images = []
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if tc.function.name == "submit_schematic":
                schematic = args.get("schematic", args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": "submitted"})
                continue
            out = env.step({"tool": tc.function.name, "args": args})
            if "image_png_b64" in out:
                extra_images.append(out["image_png_b64"])
                content = out["note"] + " (image attached in the next message)"
            else:
                content = json.dumps(out)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
        if schematic is not None:
            break
        left = max_turns - turn - 1
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}} for b64 in extra_images]
        content.append({"type": "text", "text": f"Turns left: {left}. " +
                        ("Submit your schematic NOW with submit_schematic." if left <= 2 else "")})
        messages.append({"role": "user", "content": content})
    _, reward, _, info = env.step({"submit": schematic or {}})
    info.update(plan=plan, turns=turn + 1, usage=usage, schematic=schematic)
    return info

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", help="env var holding the API key")
    ap.add_argument("--split", default="test")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=2048)
    a = ap.parse_args()
    client = OpenAI(base_url=a.base_url, api_key=os.environ.get(a.api_key_env, "EMPTY"))
    plans = json.load(open("split.json"))[a.split]
    if a.limit: plans = plans[:a.limit]
    outdir = os.path.join("results", a.model.replace("/", "_")); os.makedirs(outdir, exist_ok=True)
    summary = []
    for k in range(a.runs):
        for plan in plans:
            t0 = time.time()
            try:
                info = run_episode(client, a.model, plan, max_side=a.max_side)
            except Exception as ex:
                print("ERR", plan, ex); continue
            info["seconds"] = round(time.time() - t0, 1)
            json.dump(info, open(os.path.join(outdir, f"{plan}_run{k}.json"), "w"), ensure_ascii=False, indent=1)
            summary.append(info)
            print(f"run{k} {plan:34} reward={info['reward']:.3f} acc={info['answer_acc']:.3f} "
                  f"itemsF1={info['items_f1']:.3f} turns={info['turns']} {info['seconds']}s")
    if summary:
        avg = lambda k: sum(s[k] for s in summary) / len(summary)
        print(f"\nMEAN over {len(summary)} episodes: reward={avg('reward'):.3f} acc={avg('answer_acc'):.3f} itemsF1={avg('items_f1'):.3f}")
