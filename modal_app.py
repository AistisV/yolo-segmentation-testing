"""GPU pipeline on Modal. Deployed once, then started with launch.py, so nothing depends on your terminal.

    modal deploy modal_app.py        # upload code + build image (again after any code change)
    python3 launch.py                # start the pipeline and follow its log (Ctrl+C only stops following)
    python3 launch.py log            # print the log so far
    modal app stop junction-floorplans   # stop everything

Stages (each one is skipped if already done; training resumes from its last checkpoint):
  smoke      3 training steps on the longest plans -> merge -> serve -> evaluate 2 plans   (catches crashes early)
  fetch      full Stoa dataset -> Modal volume (Swiss plans + questions)
  prep       ground truth for every Swiss plan, checked against Stoa's answers; fixed split
  zeroshot   untrained Qwen3-VL-8B on the eval plans (15 test + 100 holdout)
  sft        supervised LoRA fine-tuning on the 'sft' plans, merged
  sft_eval   same eval plans
  exp_sample fine-tuned model samples 4 schematics for each 'exp' plan; the environment scores them
  exp_train  fine-tune on the best-scoring sample per plan (experience from the environment, no ground truth)
  exp_eval   same eval plans
"""
import glob, json, os, random, shutil, signal, subprocess, threading, time, zipfile
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
PREVIEW = os.environ.get("PREVIEW_DIR", os.path.expanduser("~/Downloads/junction-dataset-preview"))
APP = "junction-floorplans"
BASE = "Qwen/Qwen3-VL-8B-Instruct"
URL = "https://fwjydgxpttajiqpfmqtj.supabase.co/storage/v1/object/public/stoa-junction-2026/junction-dataset.zip"
SPLIT = "/vol/split_full.json"
FULL = dict(IMAGES_DIR="/vol/data/data/swiss_dwellings:/vol/data/data/cubicasa5k", QUESTIONS="/vol/data/questions.jsonl", GT_DIR="/vol/gt_full")
CUBI_SPLIT = "/vol/split_cubi.json"
PREV = dict(IMAGES_DIR="/preview/data/swiss_dwellings", QUESTIONS="/preview/questions.jsonl", GT_DIR="/code/gt")

image = (
    # CUDA *devel* base: vLLM compiles some GPU kernels at start-up and needs nvcc
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("wget", "unzip", "curl", "git", "libgl1", "libglib2.0-0")
    .pip_install("vllm", "openai", "pillow", "shapely", "opencv-python-headless", "scikit-image", "numpy",
                 "ezdxf", "pymupdf", "trl>=0.20", "peft", "accelerate", "datasets", "qwen-vl-utils")
    .pip_install("segmentation-models-pytorch", "ultralytics")
    .env({"HF_HOME": "/vol/hf", "CUDA_HOME": "/usr/local/cuda", "IMAGES_DIR": "/preview/data/swiss_dwellings",
          "QUESTIONS": "/preview/questions.jsonl", "GT_DIR": "/code/gt"})
    .add_local_dir(HERE, "/code", ignore=["results", "__pycache__", "*.tgz", "*.png", "_to_delete"])
    .add_local_dir(PREVIEW, "/preview")
)
vol = modal.Volume.from_name("jx", create_if_missing=True)
state = modal.Dict.from_name("jx-state", create_if_missing=True)
app = modal.App(APP, image=image)
RETRY = modal.Retries(max_retries=2, initial_delay=10.0, backoff_coefficient=1.0)
GPU = dict(gpu="H100", volumes={"/vol": vol}, cpu=8, memory=49152, retries=RETRY)


def run(cmd, env=None, log=False):
    """Run a shell command in /code, stream its output, return it; on failure raise with the last lines."""
    print("$", cmd, flush=True)
    e = dict(os.environ); e.update(env or {}); e.update(TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    p = subprocess.Popen(cmd, shell=True, cwd="/code", env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    lines = []
    for line in p.stdout:
        print(line, end="", flush=True); lines.append(line)
        if log: _log("   " + line.rstrip()[:300])
    if p.wait():
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}\n" + "".join(lines[-40:])[-3500:])
    return "".join(lines)


class committing:
    """Save the volume every few minutes while a long stage runs, so an interrupted container loses little."""
    def __enter__(self):
        self.stop = threading.Event()
        def loop():
            while not self.stop.wait(240):
                try: vol.commit()
                except Exception as e: print("volume commit:", repr(e)[:200], flush=True)
        threading.Thread(target=loop, daemon=True).start()
    def __exit__(self, *a):
        self.stop.set()
        try: vol.commit()
        except Exception as e: print("volume commit:", repr(e)[:200], flush=True)


def stop_vllm(srv):
    try: os.killpg(srv.pid, signal.SIGTERM)
    except Exception: pass
    try: srv.wait(timeout=60)
    except Exception:
        try: os.killpg(srv.pid, signal.SIGKILL)
        except Exception: pass


def start_vllm(model):
    srv = subprocess.Popen(["vllm", "serve", model, "--served-model-name", "policy", "--max-model-len", "32768",
                            "--limit-mm-per-prompt", '{"image":1}', "--gpu-memory-utilization", "0.85"],
                           stdout=open("/tmp/vllm.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    for i in range(240):                                     # up to 40 minutes (first download of the weights)
        if srv.poll() is not None:
            break
        if subprocess.run("curl -sf localhost:8000/v1/models > /dev/null", shell=True).returncode == 0:
            print(f"vLLM up after {i*10}s", flush=True); return srv
        time.sleep(10)
    tail = open("/tmp/vllm.log", errors="replace").read()[-3000:]
    stop_vllm(srv)
    raise RuntimeError("vLLM failed to start:\n" + tail)


@app.function(timeout=3 * 3600, **GPU)
def smoke_remote():
    with committing():
        for d in ("/vol/ckpt/smoke", "/vol/ckpt/smoke-merged"):
            shutil.rmtree(d, ignore_errors=True)
        run("python3 make_sft.py --split train --longest-first --limit 3 --out /tmp/smoke.jsonl", PREV)
        tr = run(f"python3 train_sft.py --data /tmp/smoke.jsonl --model {BASE} --out /vol/ckpt/smoke --max-steps 3 --accum 1 --save-steps 2")
        speed = next((l.strip() for l in tr.splitlines() if "train_runtime" in l), "")
        srv = start_vllm("/vol/ckpt/smoke-merged")
        try:
            txt = run("python3 eval_policy.py --split test --limit 2 --greedy --samples 2 --max-tokens 1500 --tag smoke "
                      "--out-dir /tmp/smoke_res --save-best /tmp/smoke_best.jsonl --min-reward 0", PREV)
        finally:
            stop_vllm(srv)
        for d in ("/vol/ckpt/smoke", "/vol/ckpt/smoke-merged"):
            shutil.rmtree(d, ignore_errors=True)
    return "train + merge + serve + eval all ran. 3 steps on the 3 longest plans: " + speed[-200:] + "\n" + txt[txt.rfind("=== "):]


@app.function(volumes={"/vol": vol}, timeout=3 * 3600, cpu=4, memory=8192, retries=RETRY)
def fetch_remote():
    sw, qf = FULL["IMAGES_DIR"].split(":")[0], FULL["QUESTIONS"]
    if os.path.exists(qf) and len(glob.glob(sw + "/*.dxf")) > 900:
        return f"already there: {len(glob.glob(sw + '/*.dxf'))} DXF"
    os.makedirs(sw, exist_ok=True)
    z = "/vol/data/junction-dataset.zip"
    for attempt in (1, 2):
        run(f"wget -q -c '{URL}' -O {z} || wget -q '{URL}' -O {z}")
        try:
            n = 0
            with zipfile.ZipFile(z) as zf:
                for name in zf.namelist():
                    base = os.path.basename(name)
                    if not base: continue
                    if "/swiss_dwellings/" in "/" + name: dst = os.path.join(sw, base)
                    elif base == "questions.jsonl": dst = qf
                    else: continue
                    with zf.open(name) as src, open(dst, "wb") as f:
                        shutil.copyfileobj(src, f, 1 << 20)
                    n += 1
            break
        except zipfile.BadZipFile:
            if attempt == 2: raise
            os.remove(z)
    vol.commit()
    c = lambda ext: len(glob.glob(sw + "/*." + ext))
    return f"extracted {n} files: {c('dxf')} DXF, {c('pdf')} PDF, {c('png')} PNG; questions {os.path.getsize(qf)/1e6:.0f} MB"


@app.function(volumes={"/vol": vol}, timeout=2 * 3600, cpu=16, memory=32768, retries=RETRY)
def prep_remote():
    txt = run(f"python3 prepare_full.py --data /vol/data --gt /vol/gt_full --out {SPLIT} --procs 16")
    vol.commit()
    return txt[-1800:]


@app.function(timeout=6 * 3600, **GPU)
def eval_remote(model: str, tag: str, key: str = "eval", greedy: bool = True, samples: int = 0, temperature: float = 0.7,
                max_tokens: int = 16000, save_best: str = "", min_reward: float = 0.5, guided: bool = False,
                shard: int = 0, nshards: int = 1, plans_file: str = SPLIT, limit: int = 0):
    with committing():
        srv = start_vllm(model)
        try:
            txt = run(f"python3 eval_policy.py --plans-file {plans_file} --key {key} --tag {tag} --out-dir /vol/results/{tag} "
                      f"--samples {samples} --temperature {temperature} --max-tokens {max_tokens} --shard {shard} --nshards {nshards} --limit {limit}"
                      + (" --greedy" if greedy else "") + (" --guided" if guided else "")
                      + (f" --save-best {save_best} --min-reward {min_reward} --require-complete" if save_best else ""), FULL)
        finally:
            stop_vllm(srv)
    return txt[txt.rfind("=== "):]


@app.function(timeout=6 * 3600, **GPU)
def gen_remote(model: str, tag: str, key: str = "eval", greedy: bool = True, samples: int = 0, temperature: float = 0.7,
                max_tokens: int = 16000, save_best: str = "", min_reward: float = 0.5, guided: bool = False,
                shard: int = 0, nshards: int = 1, plans_file: str = SPLIT, limit: int = 0):
    with committing():
        srv = start_vllm(model)
        try:
            txt = run(f"python3 eval_policy.py --plans-file {plans_file} --key {key} --tag {tag} --out-dir /vol/results/{tag} "
                      f"--samples {samples} --temperature {temperature} --max-tokens {max_tokens} --shard {shard} --nshards {nshards} --limit {limit}"
                      + (" --greedy" if greedy else "") + (" --guided" if guided else "")
                      + (f" --save-best {save_best} --min-reward {min_reward} --require-complete" if save_best else ""), FULL)
        finally:
            stop_vllm(srv)
    return txt[txt.rfind("=== "):]


@app.function(volumes={"/vol": vol}, timeout=3600, cpu=8, memory=16384)
def summary_remote(tag: str, key: str, greedy: bool, samples: int, temperature: float, save_best: str = "", min_reward: float = 0.5,
                   plans_file: str = SPLIT, limit: int = 0):
    vol.reload()
    txt = run(f"python3 eval_policy.py --plans-file {plans_file} --key {key} --tag {tag} --out-dir /vol/results/{tag} --summary-only --limit {limit} "
              f"--samples {samples} --temperature {temperature}" + (" --greedy" if greedy else "")
              + (f" --save-best {save_best} --min-reward {min_reward} --require-complete" if save_best else ""), FULL)
    vol.commit()
    return txt[txt.rfind("=== "):]


def sharded_eval(model, tag, key, greedy, samples, temperature, n, save_best="", min_reward=0.5, plans_file=SPLIT, limit=0):
    """Split the plans over n GPUs (each its own vLLM), then summarise everything on CPU."""
    calls = [gen_remote.spawn(model=model, tag=tag, key=key, greedy=greedy, samples=samples, temperature=temperature,
                              max_tokens=16000, min_reward=min_reward, guided=True, shard=i, nshards=n, plans_file=plans_file, limit=limit)
             for i in range(n)]
    errs = []
    for c in calls:
        try: c.get()
        except BaseException as e: errs.append(repr(e)[-600:])
    if len(errs) == n:
        raise RuntimeError(f"all {n} shards failed:\n" + "\n".join(errs[:2]))
    out = summary_remote.remote(tag, key, greedy, samples, temperature, save_best, min_reward, plans_file, limit)
    return out + (f"\n{len(errs)} of {n} shards failed (their missing plans count as 0):\n" + "\n".join(errs) if errs else "")


@app.function(timeout=8 * 3600, volumes={"/vol": vol})
def sharded_remote(model: str, tag: str, key: str, greedy: bool, samples: int, temperature: float, n: int,
                   save_best: str = "", min_reward: float = 0.5, plans_file: str = SPLIT, limit: int = 0):
    return sharded_eval(model, tag, key, greedy, samples, temperature, n, save_best, min_reward, plans_file, limit)


@app.function(timeout=10 * 3600, **GPU)
def train_remote(kind: str, exp_rows: str = "/vol/exp_rows.jsonl", out: str = "/vol/ckpt/exp"):
    with committing():
        if kind == "sft":
            run(f"python3 make_sft.py --plans-file {SPLIT} --key sft --out /vol/sft_rows.jsonl", FULL)
            txt = run(f"python3 train_sft.py --data /vol/sft_rows.jsonl --model {BASE} --out /vol/ckpt/sft --epochs 2")
        else:   # experience round: best-of-4 samples chosen by the environment + an equal number of supervised rows as replay
            exp = [l for f in exp_rows.split(",") if os.path.exists(f) for l in open(f, encoding="utf-8") if l.strip()]
            sft = [l for l in open("/vol/sft_rows.jsonl", encoding="utf-8") if l.strip()]
            random.Random(0).shuffle(sft)
            mix = exp + sft[:len(exp)]
            random.Random(1).shuffle(mix)
            open("/vol/exp_mix.jsonl", "w", encoding="utf-8").write("".join(l if l.endswith("\n") else l + "\n" for l in mix))
            print(f"{len(exp)} experience rows + {min(len(exp), len(sft))} replay rows", flush=True)
            assert len(exp) >= 20, f"only {len(exp)} experience rows passed the reward threshold"
            txt = run(f"python3 train_sft.py --data /vol/exp_mix.jsonl --model /vol/ckpt/sft-merged --out {out} --epochs 2 --lr 5e-5")
    return txt[-1200:]


@app.function(volumes={"/vol": vol}, timeout=2 * 3600, cpu=4, memory=8192, retries=RETRY)
def cubi_prep_remote():
    d = "/vol/data/data/cubicasa5k"
    if len(glob.glob(d + "/*.png")) < 4000:
        os.makedirs(d, exist_ok=True)
        with zipfile.ZipFile("/vol/data/junction-dataset.zip") as zf:
            for name in zf.namelist():
                if "/cubicasa5k/" in "/" + name and os.path.basename(name):
                    with zf.open(name) as src, open(os.path.join(d, os.path.basename(name)), "wb") as f:
                        shutil.copyfileobj(src, f, 1 << 20)
    plans = set()
    for line in open(FULL["QUESTIONS"], encoding="utf-8"):
        if "cubicasa5k" in line:
            plans.add(os.path.basename(json.loads(line)["path"]).rsplit(".", 1)[0])
    plans = sorted(p for p in plans if os.path.exists(f"{d}/{p}.png"))
    random.Random(13).shuffle(plans)
    json.dump(dict(cubi_eval=plans[:100], cubi_exp=plans[100:]), open(CUBI_SPLIT, "w"))
    vol.commit()
    return f"{len(glob.glob(d + '/*.png'))} CubiCasa images; split: eval 100, exp {len(plans) - 100}"


# ---------- round 4: two-part policy (rooms from the whole image, items tile by tile) ----------

@app.function(timeout=10 * 3600, **GPU)
def tiles_train_remote():
    with committing():
        run(f"python3 make_tile_rows.py --plans-file {SPLIT} --key sft --tiles-per-plan 4 --out /vol/tile_rows.jsonl", FULL)
        txt = run("python3 train_sft.py --data /vol/tile_rows.jsonl --model /vol/ckpt/sft-merged --out /vol/ckpt/tiles --epochs 1")
    return txt[-600:]


@app.function(timeout=6 * 3600, **GPU)
def tiles_gen_remote(model: str, tag: str, key: str, shard: int, nshards: int, plans_file: str = SPLIT, limit: int = 0):
    with committing():
        srv = start_vllm(model)
        try:
            run(f"python3 eval_tiles.py --plans-file {plans_file} --key {key} --tag {tag} --out-dir /vol/results/{tag} "
                f"--shard {shard} --nshards {nshards} --limit {limit}", FULL)
        finally:
            stop_vllm(srv)
    return "ok"


@app.function(volumes={"/vol": vol}, timeout=3600, cpu=8, memory=16384)
def tiles_summary_remote(tag: str, key: str, plans_file: str = SPLIT, limit: int = 0):
    vol.reload()
    txt = run(f"python3 eval_tiles.py --plans-file {plans_file} --key {key} --tag {tag} --out-dir /vol/results/{tag} --summary-only --limit {limit}", FULL)
    return txt[txt.rfind("=== "):]


@app.function(timeout=8 * 3600, volumes={"/vol": vol})
def tiles_sharded_remote(model: str, tag: str, key: str, n: int, plans_file: str = SPLIT, limit: int = 0):
    calls = [tiles_gen_remote.spawn(model, tag, key, i, n, plans_file, limit) for i in range(n)]
    errs = []
    for c in calls:
        try: c.get()
        except BaseException as e: errs.append(repr(e)[-600:])
    if len(errs) == n:
        raise RuntimeError(f"all {n} shards failed:\n" + "\n".join(errs[:2]))
    out = tiles_summary_remote.remote(tag, key, plans_file, limit)
    return out + (f"\n{len(errs)} of {n} shards failed:\n" + "\n".join(errs) if errs else "")


def _stage(name, fn, *args, needs=(), **kw):
    if state.get("done:" + name):
        _log(f"SKIP {name} (already done)"); return None
    missing = [n for n in needs if not state.get("done:" + n)]
    if missing:
        _log(f"SKIP {name}: needs {', '.join(missing)}"); return None
    _log(f"START {name}")
    return (name, time.time(), fn.spawn(*args, **kw))


def _finish(h):
    if h is None: return
    name, t0, fc = h
    try:
        res = fc.get()
    except BaseException as e:
        _log(f"FAIL {name} after {(time.time()-t0)/60:.0f} min: {repr(e)[-3500:]}"); return
    state.put("done:" + name, True)
    _log(f"DONE {name} in {(time.time()-t0)/60:.0f} min\n{res}")


@app.function(timeout=24 * 3600, nonpreemptible=True)
def tiles_remote():
    _log("==== round 4 (tiles) started ====")
    base = _stage("t_base_swiss", tiles_sharded_remote, "/vol/ckpt/sft-merged", "t_base_swiss", "eval", 4)   # no tile training yet
    train = _stage("t_train", tiles_train_remote)
    _finish(base); _finish(train)
    hs = [_stage("t_swiss", tiles_sharded_remote, "/vol/ckpt/tiles-merged", "t_swiss", "eval", 4, needs=["t_train"]),
          _stage("t_cubi", tiles_sharded_remote, "/vol/ckpt/tiles-merged", "t_cubi", "cubi_eval", 2, CUBI_SPLIT, needs=["t_train", "c_prep"])]
    for h in hs: _finish(h)
    _log("==== ALL DONE (round 4) ====")


# ---------- round 5: specialist vision models (room segmentation + item detector) ----------

SW = "/vol/data/data/swiss_dwellings"


@app.function(volumes={"/vol": vol}, timeout=2 * 3600, cpu=16, memory=32768)
def seg_prep_remote():
    if os.path.exists("/vol/segdata/yolo/data.yaml"):
        return "already there"
    txt = run(f"python3 seg_data.py --gt /vol/gt_full --images {SW} --split {SPLIT} --out /vol/segdata")
    vol.commit()
    return txt[-500:]


def _unet_train(steps, data, out):
    with committing():
        os.makedirs("/vol/ckpt", exist_ok=True)
        txt = run(f"python3 train_unet.py --data {data} --images {SW} --out {out} --steps {steps} --bs 16 --crop 512")
    return txt[-900:]


def _yolo_train(hours, data, name):
    with committing():
        # one tar file reads from the volume far faster than ~20k small files
        run(f"rm -rf /tmp/yolo && if [ -f {data}/yolo.tar ]; then tar xf {data}/yolo.tar -C /tmp; else cp -r {data}/yolo /tmp/yolo; fi "
            f"&& sed -i 's#path: .*#path: /tmp/yolo#' /tmp/yolo/data.yaml && echo 'tiles copied:' && ls /tmp/yolo/images/train | wc -l")
        txt = run(f"python3 yolo_train.py --data /tmp/yolo/data.yaml --hours {hours} --name {name}")
    return txt[-1500:]


def _seg_predict(key, out, unet, yolo, plans_file, images):
    with committing():
        txt = run(f"python3 seg_predict.py --unet {unet} --yolo {yolo} --images {images} "
                  f"--plans-file {plans_file} --key {key} --out {out}")
    return txt[-600:]


@app.function(timeout=3 * 3600, **GPU)
def unet_train_remote(steps: int = 5000):
    return _unet_train(steps, "/vol/segdata", "/vol/ckpt/unet.pt")


@app.function(timeout=3 * 3600, **GPU)
def yolo_train_remote(hours: float = 0.45):
    return _yolo_train(hours, "/vol/segdata", "yolo")


@app.function(timeout=2 * 3600, **GPU)
def seg_predict_remote(key: str = "eval", out: str = "/vol/seg_preds"):
    return _seg_predict(key, out, "/vol/ckpt/unet.pt", "/vol/ckpt/yolo/weights/best.pt", SPLIT, SW)


@app.function(timeout=3 * 3600, **GPU)
def unet2_train_remote(steps: int, data: str, out: str):
    return _unet_train(steps, data, out)


@app.function(timeout=3 * 3600, **GPU)
def yolo2_train_remote(hours: float, data: str, name: str):
    return _yolo_train(hours, data, name)


@app.function(timeout=2 * 3600, **GPU)
def seg2_predict_remote(key: str, out: str, unet: str, yolo: str, plans_file: str, images: str):
    return _seg_predict(key, out, unet, yolo, plans_file, images)


@app.function(timeout=12 * 3600, nonpreemptible=True)
def seg_remote():
    _log("==== round 5 (specialist vision models) started ====")
    _finish(_stage("s_prep", seg_prep_remote))
    hs = [_stage("s_unet", unet_train_remote, needs=["s_prep"]), _stage("s_yolo", yolo_train_remote, needs=["s_prep"])]
    for h in hs: _finish(h)
    _finish(_stage("s_predict", seg_predict_remote, needs=["s_unet", "s_yolo"]))
    _log("==== ALL DONE (round 5) ====")


# ---------- round 6: specialist models trained on Swiss + CubiCasa ----------

CUBI_IMG = "/vol/data/data/cubicasa5k"


@app.function(volumes={"/vol": vol}, timeout=3 * 3600, cpu=8, memory=16384)
def cubi_ann_remote():
    os.makedirs("/vol/cubi", exist_ok=True)
    z = "/vol/cubi/cubicasa5k.zip"
    import zipfile
    if not (os.path.exists(z) and zipfile.is_zipfile(z)):     # a half-finished earlier download is not a zip
        z = "/tmp/cubicasa5k.zip"                             # local disk; 16 parallel range requests, progress in the log
        run(f"python3 dl.py 'https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1' {z} --parts 16", log=True)
    txt = run(f"python3 cubi_gt.py --zip {z} --images {CUBI_IMG} --questions {FULL['QUESTIONS']} --out /vol/gt_cubi", log=True)
    vol.commit()
    return txt[-2500:]


@app.function(volumes={"/vol": vol}, timeout=3600, cpu=8, memory=16384, nonpreemptible=True)
def cubi_fit_remote():
    """Learn which CubiCasa classes / texts Stoa counted per question type (see cubi_fit.py). Result in the log + /vol/cubi_map.json."""
    _log("==== cubi_fit started ====")
    z = "/tmp/cubicasa5k.zip"
    run(f"python3 dl.py 'https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1' {z} --parts 16", log=True)
    txt = run(f"python3 cubi_fit.py --zip {z} --images {CUBI_IMG} --questions {FULL['QUESTIONS']} --out /vol/cubi_map.json", log=True)
    vol.commit()
    _log("==== ALL DONE (cubi_fit) ====")
    return txt[-3000:]


def _score(preds, variants="split,split_gtscale,split_snapscale"):
    """area_diag + score_seg on predictions in the volume. Gemini results are read from /vol/res (modal volume put jx results res)."""
    w = "rm -rf /tmp/w && cp -r /code /tmp/w && ln -sfn /vol/res /tmp/w/results && cd /tmp/w && "
    if variants == "cubi":                       # CubiCasa test plans: counts only, ground truth from the SVG annotations
        return run(w + f"python3 score_seg.py --preds {preds} --cubi --gt-cubi /vol/gt_cubi3 --plans-file {CUBI_SPLIT} --confs 0.1,0.2,0.3,0.5", env=FULL, log=True)[-9000:]
    a = run(w + f"python3 area_diag.py --preds {preds} --gt /vol/gt_full", env=FULL, log=True)
    b = run(w + f"python3 score_seg.py --preds {preds} --variants {variants}", env=FULL, log=True)
    return a[-1500:] + b[-6000:]


@app.function(volumes={"/vol": vol}, timeout=3600, cpu=4, memory=8192, nonpreemptible=True)
def cubi_check_remote(yolo: str = "/vol/ckpt/yolo2/weights/best.pt", out: str = "/vol/seg_preds2b"):
    """Re-predict the CubiCasa test plans with an existing detector, then score them (python3 launch.py cubicheck)."""
    _log(f"==== cubi check ({yolo}) started ====")
    vol.reload()
    unet = "/vol/ckpt/unet2.pt" if os.path.exists("/vol/ckpt/unet2.pt") else "/vol/ckpt/unet.pt"
    _log("   " + seg2_predict_remote.remote("cubi_eval", out, unet, yolo, CUBI_SPLIT, SW + ":" + CUBI_IMG)[-300:])
    vol.reload()
    _score(out, "cubi")
    _log("==== ALL DONE (cubi check) ====")


@app.function(volumes={"/vol": vol}, timeout=3600, cpu=8, memory=16384, nonpreemptible=True)
def score_remote(preds: str, variants: str = "split_snapscale"):
    """Score saved predictions (rooms + detector items) under Stoa's rule. python3 launch.py score /vol/seg_preds"""
    _log(f"==== score {preds} started ====")
    vol.reload(); _score(preds, variants)
    _log("==== ALL DONE (score) ====")


@app.function(volumes={"/vol": vol}, timeout=3600, cpu=8, memory=16384, nonpreemptible=True)
def quick_remote():
    """First end-to-end test: the round-5 room segmenter alone on the 115 Swiss eval plans, scored under Stoa's rule."""
    _log("==== quick test (room segmenter only, Swiss eval) started ====")
    if not os.path.exists("/vol/ckpt/unet.pt"): _log("no /vol/ckpt/unet.pt"); return
    vol.reload()
    if not os.path.exists("/vol/seg_preds_unet/" + json.load(open(SPLIT))["eval"][-1] + ".json"):
        _log("   predicting room outlines on GPU ...")
        _log("   " + seg2_predict_remote.remote("eval", "/vol/seg_preds_unet", "/vol/ckpt/unet.pt", "none", SPLIT, SW)[-300:])
        vol.reload()
    _score("/vol/seg_preds_unet")
    _log("==== ALL DONE (quick) ====")


@app.function(volumes={"/vol": vol}, timeout=2 * 3600, cpu=16, memory=32768)
def seg2_prep_remote():
    if os.path.exists("/vol/segdata2/yolo/data.yaml"):
        return "already there"
    txt = run(f"python3 seg_data.py --gt /vol/gt_full --images {SW} --split {SPLIT} --out /vol/segdata2 "
              f"--gt2 /vol/gt_cubi --images2 {CUBI_IMG} --split2 {CUBI_SPLIT} --limit2 2500")
    run("tar cf /vol/segdata2/yolo.tar -C /vol/segdata2 yolo && ls -la /vol/segdata2/yolo.tar")
    vol.commit()
    return txt[-500:]


@app.function(timeout=12 * 3600, nonpreemptible=True)
def seg2_remote():
    _log("==== round 6 (specialist models, Swiss + CubiCasa) started ====")
    _finish(_stage("r6_ann", cubi_ann_remote))
    _finish(_stage("r6_prep", seg2_prep_remote, needs=["r6_ann"]))
    hs = [_stage("r6_unet", unet2_train_remote, 6000, "/vol/segdata2", "/vol/ckpt/unet2.pt", needs=["r6_prep"]),
          _stage("r6_yolo", yolo2_train_remote, 0.5, "/vol/segdata2", "yolo2", needs=["r6_prep"])]
    for h in hs: _finish(h)
    hs = [_stage("r6_pred_swiss", seg2_predict_remote, "eval", "/vol/seg_preds2", "/vol/ckpt/unet2.pt", "/vol/ckpt/yolo2/weights/best.pt",
                 SPLIT, SW, needs=["r6_unet", "r6_yolo"]),
          _stage("r6_pred_cubi", seg2_predict_remote, "cubi_eval", "/vol/seg_preds2", "/vol/ckpt/unet2.pt", "/vol/ckpt/yolo2/weights/best.pt",
                 CUBI_SPLIT, SW + ":" + CUBI_IMG, needs=["r6_unet", "r6_yolo"])]
    for h in hs: _finish(h)
    _log("==== ALL DONE (round 6) ====")


# ---------- round 7: CubiCasa fixture labels fixed (fitted names, no items dropped), detector retrained ----------

@app.function(volumes={"/vol": vol}, timeout=3600, cpu=8, memory=16384)
def cubi_ann3_remote(out: str):
    z = "/tmp/cubicasa5k.zip"
    run(f"python3 dl.py 'https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1' {z} --parts 16", log=True)
    txt = run(f"python3 cubi_gt.py --zip {z} --images {CUBI_IMG} --questions {FULL['QUESTIONS']} --out {out}", log=True)
    vol.commit()
    return txt[-1500:]


@app.function(volumes={"/vol": vol}, timeout=2 * 3600, cpu=16, memory=32768)
def seg3_prep_remote(gt2: str, out: str):
    txt = run(f"python3 seg_data.py --gt /vol/gt_full --images {SW} --split {SPLIT} --out {out} "
              f"--gt2 {gt2} --images2 {CUBI_IMG} --split2 {CUBI_SPLIT} --limit2 2500", log=False)
    run(f"tar cf {out}/yolo.tar -C {out} yolo && ls -la {out}/yolo.tar", log=True)
    vol.commit()
    return txt[-500:]


@app.function(timeout=8 * 3600, nonpreemptible=True)
def seg3_remote():
    _log("==== round 7 (fixed CubiCasa fixture labels, detector retrained) started ====")
    _finish(_stage("r7_ann", cubi_ann3_remote, "/vol/gt_cubi3"))
    _finish(_stage("r7_prep", seg3_prep_remote, "/vol/gt_cubi3", "/vol/segdata3", needs=["r7_ann"]))
    _finish(_stage("r7_yolo", yolo2_train_remote, 0.5, "/vol/segdata3", "yolo3", needs=["r7_prep"]))
    vol.reload()
    unet = "/vol/ckpt/unet2.pt" if os.path.exists("/vol/ckpt/unet2.pt") else "/vol/ckpt/unet.pt"
    _log(f"   room segmenter used for predictions: {unet}")
    hs = [_stage("r7_pred_swiss", seg2_predict_remote, "eval", "/vol/seg_preds3", unet, "/vol/ckpt/yolo3/weights/best.pt",
                 SPLIT, SW, needs=["r7_yolo"]),
          _stage("r7_pred_cubi", seg2_predict_remote, "cubi_eval", "/vol/seg_preds3", unet, "/vol/ckpt/yolo3/weights/best.pt",
                 CUBI_SPLIT, SW + ":" + CUBI_IMG, needs=["r7_yolo"])]
    for h in hs: _finish(h)
    _log("==== ALL DONE (round 7) ====")


# ---------- round 8: small images (CubiCasa) upscaled to the size of the large plans; both models retrained ----------

@app.function(volumes={"/vol": vol}, timeout=2 * 3600, cpu=16, memory=65536)
def seg4_prep_remote(gt2: str, out: str, limit2: int):
    txt = run(f"python3 seg_data.py --gt /vol/gt_full --images {SW} --split {SPLIT} --out {out} "
              f"--gt2 {gt2} --images2 {CUBI_IMG} --split2 {CUBI_SPLIT} --limit2 {limit2}", log=True)
    run(f"tar cf {out}/yolo.tar -C {out} yolo && ls -la {out}/yolo.tar", log=True)
    vol.commit()
    return txt[-500:]


@app.function(volumes={"/vol": vol}, timeout=10 * 3600, cpu=8, memory=16384, nonpreemptible=True)
def seg4_remote():
    _log("==== round 8 (CubiCasa upscaled x4; U-Net + detector retrained; scored automatically) started ====")
    _finish(_stage("r8_prep", seg4_prep_remote, "/vol/gt_cubi3", "/vol/segdata4", 1500))
    hs = [_stage("r8_unet", unet2_train_remote, 6000, "/vol/segdata4", "/vol/ckpt/unet4.pt", needs=["r8_prep"]),
          _stage("r8_yolo", yolo2_train_remote, 0.6, "/vol/segdata4", "yolo4", needs=["r8_prep"])]
    for h in hs: _finish(h)
    vol.reload()
    hs = [_stage("r8_pred_swiss", seg2_predict_remote, "eval", "/vol/seg_preds4", "/vol/ckpt/unet4.pt", "/vol/ckpt/yolo4/weights/best.pt",
                 SPLIT, SW, needs=["r8_unet", "r8_yolo"]),
          _stage("r8_pred_cubi", seg2_predict_remote, "cubi_eval", "/vol/seg_preds4", "/vol/ckpt/unet4.pt", "/vol/ckpt/yolo4/weights/best.pt",
                 CUBI_SPLIT, SW + ":" + CUBI_IMG, needs=["r8_unet", "r8_yolo"])]
    for h in hs: _finish(h)
    vol.reload()
    try:
        _log("   ---- scoring: Swiss eval ----"); _score("/vol/seg_preds4", "split_bar")
        _log("   ---- scoring: CubiCasa eval ----"); _score("/vol/seg_preds4", "cubi")
    except Exception as e:
        _log(f"   scoring failed: {e!r}"[:2000])
    _log("==== ALL DONE (round 8) ====")


def _log(msg):
    line = time.strftime("%H:%M:%S UTC ") + msg
    print(line, flush=True)
    try: state.put("log", (state.get("log", "") + line + "\n")[-300000:])
    except Exception as e: print("log:", repr(e)[:200], flush=True)


@app.function(timeout=24 * 3600, nonpreemptible=True)
def guided_remote():
    """Round 2 (Sat morning): same models, constrained decoding that caps runaway outlines.
    The two baseline evals run in parallel on two GPUs; then a new experience round."""
    def stage_call(name, call_fn, needs=()):
        if state.get("done:" + name):
            _log(f"SKIP {name} (already done)"); return None
        missing = [n for n in needs if not state.get("done:" + n)]
        if missing:
            _log(f"SKIP {name}: needs {', '.join(missing)}"); return None
        _log(f"START {name}")
        return call_fn()

    def finish(name, fc, t0):
        try:
            res = fc.get()
        except BaseException as e:
            _log(f"FAIL {name} after {(time.time()-t0)/60:.0f} min: {repr(e)[-3500:]}"); return
        state.put("done:" + name, True)
        _log(f"DONE {name} in {(time.time()-t0)/60:.0f} min\n{res}")

    _log("==== round 2 (constrained decoding) started ====")
    t0 = time.time()
    fa = stage_call("g_sft", lambda: sharded_remote.spawn("/vol/ckpt/sft-merged", "g_sft", "eval", True, 3, 0.7, 4), needs=["sft"])
    fb = stage_call("g_zeroshot", lambda: sharded_remote.spawn(BASE, "g_zeroshot", "eval", True, 0, 0.7, 2))
    fc = stage_call("g_exp_sample", lambda: sharded_remote.spawn("/vol/ckpt/sft-merged", "g_exp_samples", "exp", False, 4, 0.8, 8,
                                                                 "/vol/exp_rows_g.jsonl", 0.5), needs=["sft"])
    for name, f in (("g_sft", fa), ("g_zeroshot", fb), ("g_exp_sample", fc)):
        if f is not None:
            finish(name, f, t0)
    t1 = time.time()
    f = stage_call("g_exp_train", lambda: train_remote.spawn("exp", "/vol/exp_rows_g.jsonl", "/vol/ckpt/exp_g"), needs=["g_exp_sample"])
    if f is not None: finish("g_exp_train", f, t1)
    t2 = time.time()
    f = stage_call("g_exp_eval", lambda: sharded_remote.spawn("/vol/ckpt/exp_g-merged", "g_exp", "eval", True, 3, 0.7, 4),
                   needs=["g_exp_train"])
    if f is not None: finish("g_exp_eval", f, t2)
    _log("==== ALL DONE (round 2) ====")


@app.function(timeout=24 * 3600, nonpreemptible=True)
def cubi_remote():
    """Round 3: CubiCasa (two thirds of Stoa's questions, all counts). Baselines, experience from Stoa's answers,
    then one combined training round on Swiss + CubiCasa experience, evaluated on both."""
    def stage(name, fn, *args, needs=(), **kw):
        if state.get("done:" + name):
            _log(f"SKIP {name} (already done)"); return None
        missing = [n for n in needs if not state.get("done:" + n)]
        if missing:
            _log(f"SKIP {name}: needs {', '.join(missing)}"); return None
        _log(f"START {name}")
        return (name, time.time(), fn.spawn(*args, **kw))

    def finish(h):
        if h is None: return
        name, t0, fc = h
        try:
            res = fc.get()
        except BaseException as e:
            _log(f"FAIL {name} after {(time.time()-t0)/60:.0f} min: {repr(e)[-3500:]}"); return
        state.put("done:" + name, True)
        _log(f"DONE {name} in {(time.time()-t0)/60:.0f} min\n{res}")

    _log("==== round 3 (CubiCasa) started ====")
    finish(stage("c_prep", cubi_prep_remote))
    hs = [stage("c_zeroshot", sharded_remote, BASE, "c_zeroshot", "cubi_eval", True, 0, 0.7, 2, plans_file=CUBI_SPLIT, needs=["c_prep"]),
          stage("c_sft", sharded_remote, "/vol/ckpt/sft-merged", "c_sft", "cubi_eval", True, 0, 0.7, 2, plans_file=CUBI_SPLIT,
                needs=["c_prep", "sft"]),
          stage("c_exp_sample", sharded_remote, "/vol/ckpt/sft-merged", "c_exp_samples", "cubi_exp", False, 4, 0.8, 8,
                "/vol/exp_rows_c.jsonl", 0.5, plans_file=CUBI_SPLIT, limit=400, needs=["c_prep", "sft"])]
    for h in hs: finish(h)
    _log("waiting for round 2's Swiss experience rows (/vol/exp_rows_g.jsonl)...")
    for _ in range(240):                                   # up to 4 h
        vol.reload()
        if os.path.exists("/vol/exp_rows_g.jsonl") and os.path.exists("/vol/exp_rows_c.jsonl"):
            break
        time.sleep(60)
    finish(stage("mix_train", train_remote, "exp", "/vol/exp_rows_g.jsonl,/vol/exp_rows_c.jsonl", "/vol/ckpt/mix2",
                 needs=["c_exp_sample"]))
    hs = [stage("mix_eval_swiss", sharded_remote, "/vol/ckpt/mix2-merged", "mix2_swiss", "eval", True, 3, 0.7, 4, needs=["mix_train"]),
          stage("mix_eval_cubi", sharded_remote, "/vol/ckpt/mix2-merged", "mix2_cubi", "cubi_eval", True, 0, 0.7, 2,
                plans_file=CUBI_SPLIT, needs=["mix_train"])]
    for h in hs: finish(h)
    _log("==== ALL DONE (round 3) ====")


@app.function(timeout=24 * 3600, nonpreemptible=True)
def overnight_remote():
    def stage(name, fn, *args, needs=()):
        if state.get("done:" + name):
            _log(f"SKIP {name} (already done)"); return
        missing = [n for n in needs if not state.get("done:" + n)]
        if missing:
            _log(f"SKIP {name}: needs {', '.join(missing)}"); return
        _log(f"START {name}"); t0 = time.time()
        try:
            res = fn.remote(*args)
        except BaseException as e:
            _log(f"FAIL {name} after {(time.time()-t0)/60:.0f} min: {repr(e)[-3500:]}"); return
        state.put("done:" + name, True)
        _log(f"DONE {name} in {(time.time()-t0)/60:.0f} min\n{res}")

    _log("==== pipeline started ====")
    stage("smoke", smoke_remote)
    stage("fetch", fetch_remote)
    stage("prep", prep_remote, needs=["fetch"])
    stage("zeroshot", eval_remote, BASE, "zeroshot", "eval", True, 1, 0.7, 10000, needs=["prep"])
    stage("sft", train_remote, "sft", needs=["prep"])
    stage("sft_eval", eval_remote, "/vol/ckpt/sft-merged", "sft", "eval", True, 3, 0.7, 16000, needs=["sft"])
    stage("exp_sample", eval_remote, "/vol/ckpt/sft-merged", "exp_samples", "exp", False, 4, 0.8, 16000, "/vol/exp_rows.jsonl", 0.5, needs=["sft"])
    stage("exp_train", train_remote, "exp", needs=["exp_sample"])
    stage("exp_eval", eval_remote, "/vol/ckpt/exp-merged", "exp", "eval", True, 3, 0.7, 16000, needs=["exp_train"])
    _log("==== ALL DONE ====")
