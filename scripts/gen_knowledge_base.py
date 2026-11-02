"""Generate the 114-class aircraft semantic knowledge base (paper section 3.3.1)
by querying the Qwen3-VL teacher model over its OpenAI-compatible endpoint."""

import argparse
import base64
import concurrent.futures as cf
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROP_DIR = os.path.join(ROOT, "crop")
OUT_PATH = os.path.join(ROOT, "knowledge", "aircraft_semantic_kb.json")
LOG_PATH = os.path.join(ROOT, "knowledge", "gen_log.jsonl")

BASE_URL = "http://localhost:8000/v1"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# Long side window for representative crops: large enough to show structure,
# small enough to stay cheap in tokens. Crops are sent at native resolution.
PREF_MIN, PREF_MAX = 350, 900
N_REPS = 3

# Appended when regenerating classes whose first pass yielded only livery
# descriptions: the teacher defaults to paint schemes when the sample photos
# happen to show prominent national markings.
STRICT_BAN = """
5. 禁止把以下内容作为 discriminative_features 或 structural_attributes：机身涂装颜色、
   迷彩、国籍标志/机徽/国旗、部队字样、注册号或机身编号、以及当前挂载的导弹/炸弹/副油箱等外挂物。
6. 只描述机体本身的结构（翼面、尾翼、进气道、发动机布局与数量、机鼻、座舱、起落架、旋翼）。
   若某结构在图中确实看不到，填"不可见"，但必须优先寻找可见的几何结构特征。"""

PROMPT = """你是军用飞机细粒度识别专家。上面 {n} 张图片都是同一型号「{cls}」的真实照片。
请基于这些图片归纳该型号的结构化语义知识，用于辅助细粒度识别。只输出 JSON，不要任何解释文字。

字段要求：
{{
  "overall_description": "整体外观描述，1句话，只描述图中可见内容",
  "structural_attributes": {{
    "机翼布局": "", "发动机数量与布局": "", "尾翼形式": "",
    "进气道": "", "机鼻特征": "", "座舱与起落架": ""
  }},
  "discriminative_features": ["3-6条最具类别判别力的视觉特征，必须是在图中可观察到的"],
  "similar_class_differences": ["3-6条，每项形如 \\"与XXX: 差异说明\\"，XXX 必须是真实存在的飞机型号"]
}}

严格要求：
1. 只写你在图片中确实能看到或能可靠推断的结构，不要凭型号名先验编造。
2. 若某个结构在给定图片中看不清，该字段填 "不可见"，不要猜测。
3. discriminative_features 要具体到可判别的形态（如数量、位置、形状），避免"先进""流线型"这类空泛表述。
4. structural_attributes 的六个键必须全部存在。"""


def jpeg_size(path):
    with open(path, "rb") as fh:
        data = fh.read()
    i = 2
    while i < len(data) - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return w, h
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if i + 4 > len(data):
            return None
        i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    return None


def pick_representatives(cls, n=None):
    """Choose up to `n` crops: in the preferred size window, from distinct
    source images so the teacher sees varied viewpoints."""
    n = n or N_REPS
    files = []
    for name in os.listdir(os.path.join(CROP_DIR, cls)):
        if not name.lower().endswith(".jpg"):
            continue
        path = os.path.join(CROP_DIR, cls, name)
        size = jpeg_size(path)
        if not size:
            continue
        src = name.rsplit("_", 1)[0]
        files.append((max(size), size[0] * size[1], src, path))
    if not files:
        return []

    in_win = [f for f in files if PREF_MIN <= f[0] <= PREF_MAX]
    pool = in_win or sorted(files, key=lambda f: abs(f[0] - PREF_MAX))
    # one crop per source image, preferring the largest within the window
    by_src = {}
    for long_side, area, src, path in pool:
        if src not in by_src or area > by_src[src][1]:
            by_src[src] = (long_side, area, path)
    chosen = sorted(by_src.values(), key=lambda t: -t[1])[:n]
    return [c[2] for c in chosen]


def image_block(path):
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    return {"type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + b64}}


def extract_json(text):
    """Model sometimes wraps output in ```json fences or appends prose."""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    body = fence.group(1) if fence else text
    start = body.find("{")
    if start < 0:
        raise ValueError("no JSON object in response")
    depth, in_str, esc = 0, False, False
    for idx in range(start, len(body)):
        ch = body[idx]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(body[start:idx + 1])
    raise ValueError("unbalanced braces in response")


ATTR_KEYS = ["机翼布局", "发动机数量与布局", "尾翼形式", "进气道", "机鼻特征", "座舱与起落架"]


def validate(kb):
    problems = []
    if not isinstance(kb.get("overall_description"), str) or len(kb["overall_description"]) < 5:
        problems.append("overall_description missing/short")
    attrs = kb.get("structural_attributes", {})
    if not isinstance(attrs, dict):
        problems.append("structural_attributes not an object")
    else:
        missing = [k for k in ATTR_KEYS if k not in attrs]
        if missing:
            problems.append("missing attrs: " + ",".join(missing))
    for key in ("discriminative_features", "similar_class_differences"):
        val = kb.get(key)
        if not isinstance(val, list) or not val:
            problems.append(key + " empty/not a list")
    return problems


def call_api(reps, cls, strict=False, timeout=240):
    content = [image_block(p) for p in reps]
    prompt = PROMPT.format(n=len(reps), cls=cls) + (STRICT_BAN if strict else "")
    content.append({"type": "text", "text": prompt})
    payload = {"model": MODEL_ID,
               "messages": [{"role": "user", "content": content}],
               "temperature": 0.2, "max_tokens": 1000}
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode()), time.time() - t0


def do_class(cls, strict=False, nreps=None, max_attempts=4):
    reps = pick_representatives(cls, nreps)
    if not reps:
        return {"class": cls, "ok": False, "error": "no usable crop"}
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            res, dt = call_api(reps, cls, strict=strict)
            raw = res["choices"][0]["message"]["content"]
            kb = extract_json(raw)
            problems = validate(kb)
            if problems and attempt < max_attempts:
                last_err = "validation: " + "; ".join(problems)
                time.sleep(2 * attempt)
                continue
            return {"class": cls, "ok": True, "attempts": attempt, "strict": strict,
                    "latency_s": round(dt, 1),
                    "prompt_tokens": res["usage"]["prompt_tokens"],
                    "completion_tokens": res["usage"]["completion_tokens"],
                    "images": [os.path.relpath(p, ROOT).replace("\\", "/") for p in reps],
                    "knowledge": kb, "validation_issues": problems,
                    "raw": raw if problems else None}
        except urllib.error.HTTPError as exc:
            last_err = "HTTP %s %s" % (exc.code, exc.read().decode()[:200])
        except Exception as exc:  # noqa: BLE001 - retry on any transport/parse fault
            last_err = repr(exc)[:200]
        time.sleep(3 * attempt)
    return {"class": cls, "ok": False, "error": last_err}


def load_existing():
    if not os.path.exists(OUT_PATH):
        return {}
    with open(OUT_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    return {e["class"]: e for e in data.get("entries", []) if e.get("ok")}


def save(entries):
    ordered = [entries[c] for c in sorted(entries)]
    confusable = defaultdict(set)
    for e in ordered:
        for item in (e.get("knowledge", {}).get("similar_class_differences") or []):
            m = re.match(r"\s*与\s*([^:：]+)", str(item))
            if not m:
                continue
            other = m.group(1).strip()
            if other and other != e["class"]:
                confusable[e["class"]].add(other)
    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "teacher_model": MODEL_ID,
            "endpoint": BASE_URL,
            "prompt": PROMPT,
            "n_reps_per_class": N_REPS,
            "crop_size_window_pref": [PREF_MIN, PREF_MAX],
            "temperature": 0.2,
            "note": "Raw MLLM output. NOT yet filtered by the visual-consistency "
                    "check of section 3.3.2; treat as candidates, not ground truth.",
        },
        "n_classes_ok": len(ordered),
        "entries": ordered,
        "confusable_pairs_candidate": {k: sorted(v) for k, v in sorted(confusable.items())},
    }
    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, OUT_PATH)
    return len(ordered)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="append the no-livery / structure-only instruction")
    ap.add_argument("--nreps", type=int, default=N_REPS)
    args = ap.parse_args()

    classes = sorted(d for d in os.listdir(CROP_DIR)
                     if os.path.isdir(os.path.join(CROP_DIR, d)))
    entries = load_existing()
    scope = args.only or classes
    if args.force:
        # regenerate only the requested classes, keep every other entry
        dropped = [c for c in scope if entries.pop(c, None) is not None]
        print("force-refresh %d classes: %s" % (len(dropped), ", ".join(dropped)), flush=True)
    todo = [c for c in classes if c in scope and c not in entries]
    print("classes=%d already_ok=%d todo=%d workers=%d"
          % (len(classes), len(entries), len(todo), args.workers), flush=True)

    t_start = time.time()
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(do_class, c, args.strict, args.nreps): c for c in todo}
        for fut in cf.as_completed(futs):
            rec = fut.result()
            cls = rec["class"]
            done += 1
            with open(LOG_PATH, "a", encoding="utf-8") as lg:
                lg.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec.get("ok"):
                entries[cls] = rec
                tag = "ok " if not rec.get("validation_issues") else "WARN"
                print("[%3d/%3d %5.1fs] %s %-14s %s tok  lat=%.1fs imgs=%d%s"
                      % (done, len(todo), time.time() - t_start, tag, cls,
                         rec["completion_tokens"], rec["latency_s"], len(rec["images"]),
                         ("  " + ";".join(rec["validation_issues"])) if rec["validation_issues"] else ""),
                      flush=True)
            else:
                print("[%3d/%3d %5.1fs] FAIL %-14s %s"
                      % (done, len(todo), time.time() - t_start, cls, rec.get("error")),
                      flush=True)
            if done % 10 == 0:
                save(entries)

    n = save(entries)
    failed = [c for c in classes if c not in entries]
    print("\nDONE in %.1f min | saved=%d/%d | missing=%d"
          % ((time.time() - t_start) / 60, n, len(classes), len(failed)), flush=True)
    if failed:
        print("missing: " + ", ".join(failed), flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
