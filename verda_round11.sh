#!/bin/bash
# Round 11 on a single GPU box (Verda): CubiCasa detector trained on CubiCasa's own images, tested on Stoa's images.
#   bash verda_round11.sh            (everything; log in ~/r11/r11.log)
# Needs only internet access; downloads the code, Stoa's public dataset and CubiCasa5k itself.
set -euo pipefail
W=~/r11; mkdir -p $W; cd $W
exec > >(tee -a $W/r11.log) 2>&1
echo "$(date -u +%H:%M:%S) start"

[ -d code ] || git clone -q https://github.com/AistisV/yolo-segmentation-testing code
pip install -q ultralytics segmentation-models-pytorch shapely opencv-python-headless pillow numpy 2>&1 | tail -2

# Stoa dataset: questions + CubiCasa images as Stoa renders them (the test images)
if [ ! -f data/questions.jsonl ]; then
  python3 code/dl.py "https://fwjydgxpttajiqpfmqtj.supabase.co/storage/v1/object/public/stoa-junction-2026/junction-dataset.zip" stoa.zip --parts 16
  mkdir -p data/cubicasa5k
  python3 - <<'EOF'
import zipfile, os, shutil
with zipfile.ZipFile("stoa.zip") as zf:
    for n in zf.namelist():
        b = os.path.basename(n)
        if b == "questions.jsonl" and n.count("/") <= 1:
            with zf.open(n) as s, open("data/questions.jsonl", "wb") as d: shutil.copyfileobj(s, d)
        elif "/cubicasa5k/" in "/" + n and b.endswith(".png"):
            with zf.open(n) as s, open("data/cubicasa5k/" + b, "wb") as d: shutil.copyfileobj(s, d, 1 << 20)
print(len(os.listdir("data/cubicasa5k")), "CubiCasa images from Stoa")
EOF
  rm -f stoa.zip
fi

# the same plan split as on Modal (seeded shuffle of the plans that have questions)
python3 - <<'EOF'
import json, os, random
d = "data/cubicasa5k"; plans = set()
for line in open("data/questions.jsonl", encoding="utf-8"):
    if "cubicasa5k" in line: plans.add(os.path.basename(json.loads(line)["path"]).rsplit(".", 1)[0])
plans = sorted(p for p in plans if os.path.exists(f"{d}/{p}.png"))
random.Random(13).shuffle(plans)
json.dump(dict(cubi_eval=plans[:100], cubi_exp=plans[100:]), open("split_cubi.json", "w"))
json.dump(dict(cubi_eval=plans[:100], cubi_val=plans[100:][3200:3300]), open("split_r10.json", "w"))
print("split: eval 100, exp", len(plans) - 100)
EOF

# CubiCasa5k itself: SVG labels + the images they are drawn on
[ -f cubicasa5k.zip ] || python3 code/dl.py "https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1" cubicasa5k.zip --parts 16
python3 code/cubi_gt.py --zip cubicasa5k.zip --images data/cubicasa5k --questions data/questions.jsonl \
    --out gt11 --orig-out orig11 --overlays check11
echo "$(date -u +%H:%M:%S) label overlays in $W/check11 (look at one: boxes must sit on doors/windows)"

cd code
python3 seg_data.py --gt none --images none --split split_full.json --out $W/sd \
    --gt2 $W/gt11 --images2 $W/orig11 --split2 $W/split_cubi.json --limit2 3000 --only2
echo "$(date -u +%H:%M:%S) training (watch metrics/mAP50 per epoch in $W/ckpt/yolo11c/results.csv)"
python3 yolo_train.py --data $W/sd/yolo/data.yaml --hours 1.0 --name yolo11c --model yolo11m.pt --batch 32 --project $W/ckpt
for k in cubi_eval cubi_val; do
  python3 seg_predict.py --unet none --yolo $W/ckpt/yolo11c/weights/best.pt --images $W/data/cubicasa5k \
      --plans-file $W/split_r10.json --key $k --out $W/preds11 | tail -2
done
python3 cubi_count_eval.py --preds $W/preds11 --questions $W/data/questions.jsonl --split $W/split_r10.json
echo "$(date -u +%H:%M:%S) ALL DONE (round 11). Weights: $W/ckpt/yolo11c/weights/best.pt"
