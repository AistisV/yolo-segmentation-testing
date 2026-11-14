"""Start / follow the Modal pipeline. Run `modal deploy modal_app.py` first.

    python3 launch.py          start (stages already done are skipped) and follow the log
    python3 launch.py log      print the whole log
    python3 launch.py follow   follow the log of a running pipeline
    python3 launch.py fresh    forget finished stages, then start
Ctrl+C here only stops following. The pipeline keeps running on Modal; stop it with: modal app stop junction-floorplans
"""
import sys, time
import modal

APP = "junction-floorplans"
state = modal.Dict.from_name("jx-state", create_if_missing=True)
cmd = sys.argv[1] if len(sys.argv) > 1 else "start"

if cmd == "log":
    print(state.get("log", "(no log yet)")); raise SystemExit

if cmd == "reset":          # forget some finished stages:  python3 launch.py reset c_zeroshot c_sft ...
    for k in sys.argv[2:]:
        try: state.pop("done:" + k)
        except Exception: pass
    print("reset:", sys.argv[2:]); raise SystemExit

if cmd == "cancel":         # stop the last launched pipeline (its running GPU jobs finish on their own)
    cid = state.get("call")
    if cid: modal.FunctionCall.from_id(cid).cancel(); print("cancelled", cid)
    raise SystemExit

if cmd in ("round2", "round3", "round4", "round5", "round6", "round7", "fit", "quick", "score", "cubicheck", "round8"):
    sys.argv += [cmd, "force"]; cmd = "start"

if cmd == "fresh":
    state.clear(); cmd = "start"

shown = len(state.get("log", ""))
if cmd == "start":
    cid = state.get("call")
    if cid and "force" not in sys.argv:
        try:
            modal.FunctionCall.from_id(cid).get(timeout=0); running = False
        except (TimeoutError, modal.exception.TimeoutError):
            running = True
        except Exception:
            running = False
        if running:
            print("A pipeline is already running. Following its log (use `python3 launch.py start force` to start another)."); cmd = "follow"
    if cmd == "start":
        fn = "seg4_remote" if "round8" in sys.argv else "cubi_check_remote" if "cubicheck" in sys.argv else "seg3_remote" if "round7" in sys.argv else "quick_remote" if "quick" in sys.argv else "cubi_fit_remote" if "fit" in sys.argv else "seg2_remote" if "round6" in sys.argv else "seg_remote" if "round5" in sys.argv else "tiles_remote" if "round4" in sys.argv else "cubi_remote" if "round3" in sys.argv else "guided_remote" if "round2" in sys.argv else "overnight_remote"
        if "score" in sys.argv:
            fn = "score_remote"; args = [a for a in sys.argv[2:] if a not in ("score", "force")]
            call = modal.Function.from_name(APP, fn).spawn(*args)
        else:
            call = modal.Function.from_name(APP, fn).spawn()
        state.put("call", call.object_id)
        print("Pipeline started on Modal:", call.object_id)
        print("It runs on its own now. You can close this terminal or the laptop once you have seen 'DONE smoke'.\n")

try:
    while True:
        log = state.get("log", "")
        if len(log) < shown: shown = 0
        if len(log) > shown:
            print(log[shown:], end="", flush=True); shown = len(log)
            mine = {"round5": "(round 5)", "round6": "(round 6)", "round7": "(round 7)", "quick": "(quick)", "score": "(score)", "cubicheck": "(cubi check)", "round8": "(round 8)", "fit": "(cubi_fit)"}
            tag = next((v for k, v in mine.items() if k in sys.argv), "")
            if f"ALL DONE {tag}".strip() in log[-400:]: break      # stop only when this job is done
        time.sleep(20)
except KeyboardInterrupt:
    print("\nStopped following. The pipeline is still running. See it again with: python3 launch.py log")
