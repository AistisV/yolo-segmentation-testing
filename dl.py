"""Fast download: N parallel HTTP range requests, progress printed every 20 s.
    python3 dl.py URL OUT [--parts 16]
"""
import os, sys, threading, time, urllib.request

url, out = sys.argv[1], sys.argv[2]
parts = int(sys.argv[sys.argv.index("--parts") + 1]) if "--parts" in sys.argv else 16
req = urllib.request.Request(url, method="HEAD")
with urllib.request.urlopen(req, timeout=60) as r:
    size = int(r.headers["Content-Length"]); final = r.geturl()
try:
    with urllib.request.urlopen(urllib.request.Request(final, headers={"Range": "bytes=0-0"}), timeout=60) as r:
        if r.status != 206: parts = 1
except Exception:
    parts = 1
print(f"size {size / 1e9:.2f} GB, {parts} parallel parts", flush=True)
with open(out, "wb") as f: f.truncate(size)
got = [0] * parts; err = []


def worker(i):
    lo = size * i // parts; hi = size * (i + 1) // parts - 1; pos = lo
    for attempt in range(20):
        try:
            rq = urllib.request.Request(final, headers={"Range": f"bytes={pos}-{hi}"} if parts > 1 or pos else {})
            with urllib.request.urlopen(rq, timeout=120) as r, open(out, "r+b") as f:
                f.seek(pos)
                while pos <= hi:
                    b = r.read(1 << 20)
                    if not b: break
                    f.write(b); pos += len(b); got[i] = pos - lo
            if pos > hi: return
        except Exception as e:
            print(f"part {i} retry {attempt}: {e!r}"[:200], flush=True); time.sleep(3)
    err.append(i)


ts = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(parts)]
for t in ts: t.start()
t0 = time.time()
while any(t.is_alive() for t in ts):
    time.sleep(20); s = sum(got); el = time.time() - t0
    print(f"{s / 1e9:.2f}/{size / 1e9:.2f} GB  {s / el / 1e6:.0f} MB/s  eta {(size - s) / max(s / el, 1) / 60:.0f} min", flush=True)
if err or sum(got) != size:
    sys.exit(f"incomplete: parts {err}, {sum(got)}/{size}")
print("download complete", flush=True)
