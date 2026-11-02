"""Build class-level metadata the rest of the pipeline depends on:

  knowledge/domain_map.json      military / civilian attribute (absent from raw data)
  knowledge/confusable_pairs.json  MLLM-cited similar types normalized back to the
                                   114 label vocabulary, for Hard Pair mining (3.3.3)
"""

import json
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The livery-filtered copy is canonical once it exists; the raw file is kept
# only as the audit trail of what the teacher originally said.
FILTERED = os.path.join(ROOT, "knowledge", "aircraft_semantic_kb_filtered.json")
KB_PATH = FILTERED if os.path.exists(FILTERED) else os.path.join(
    ROOT, "knowledge", "aircraft_semantic_kb.json")
CROP_DIR = os.path.join(ROOT, "crop")

CIVILIAN = [
    "AirbusA320", "AirbusA330", "AirbusA340", "AirbusA350", "AirbusA380",
    "Boeing737", "Boeing747", "Boeing757", "Boeing767", "Boeing777", "Boeing787",
]

# Prefixes / designations the teacher model writes in Chinese or hyphenated form.
ALIAS = {
    "苏": "SU", "米格": "MIG", "米": "MI", "歼": "J", "轰": "H", "运": "Y",
    "直": "Z", "教": "JL", "初": "CJ", "强": "Q", "无侦": "WZ", "彩虹": "CH",
    "翼龙": "WL", "攻击": "GJ", "飞鸿": "FH", "神雕": "SD", "暗剑": "AJ",
    "伊尔": "IL", "安": "AN", "图": "TU", "别": "BE", "雅克": "YAK",
    "卡": "KA", "拉": "LA", "苏霍伊": "SU", "米里": "MI",
    "幻影": "MIRAGE", "阵风": "RAFALE", "鹰狮": "JAS39", "台风": "EF2000",
    "狂风": "TORNADO", "火神": "VULCAN", "黑鸟": "SR71", "同温层": "U2",
    "波音": "BOEING", "空客": "AIRBUS", "空中客车": "AIRBUS",
    "协和": "CONCORDE", "福克": "F28", "EMBRAER": "EMB", "巴航": "EMB",
    "枭龙": "JF17", "光辉": "TEJAS", "KC-390": "C390", "C-390": "C390",
    "FCK": "FCK1", "T500": "T50",
}

# Applied after norm(): western names the model writes in English that differ
# from our label spelling.
POST_SYNONYM = {
    "FA18": "F18", "FA22": "F22", "GRIPEN": "JAS39", "TYPHOON": "EF2000",
    "RAFAEL": "RAFALE", "FALCON": "F20", "PREDATOR": "MQ9", "REAPER": "MQ9",
    "GLOBALHAWK": "RQ4", "STEALTHFIGHTER": "F117", "LIGHTNINGII": "F35",
    "RAPTOR": "F22", "FLANKER": "SU27", "FOGHTRACER": "MIG31",
    "BACKFIRE": "TU22M", "BLACKJACK": "TU160", "BEAR": "TU95",
    "STRYKER": "A10", "WARHOG": "A10", "HUDUD": "AV8B", "HARRIER": "AV8B",
    "TOMCAT": "F14", "HORNET": "F18", "VIGGEN": "AJ37", "KAFIR": "FCK1",
    "NEMESIS": "KF21", "BOEING": "BOEING", "KC390": "C390",
}

# Exact-string fixes for labels whose digits are glued to a latin prefix.
GLUE_FIX = {
    "AIRBUS": "AIRBUS", "BOEING": "BOEING", "MIRAGE": "MIRAGE", "EF": "EF",
    "EMB": "EMB", "KC": "KC", "KJ": "KJ", "JH": "JH", "JF": "JF", "JAS": "JAS",
    "MQ": "MQ", "TB": "TB", "SR": "SR", "UH": "UH", "NH": "NH", "CH": "CH",
    "AG": "AG", "AV": "AV", "XB": "XB", "XQ": "XQ", "F": "F", "B": "B",
    "C": "C", "A": "A", "H": "H", "P": "P", "U": "U", "V": "V", "WZ": "WZ",
    "T": "T", "TU": "TU", "IL": "IL", "AN": "AN", "SU": "SU", "MI": "MI",
    "MIG": "MIG", "KA": "KA", "BE": "BE", "Z": "Z", "Y": "Y", "J": "J",
    "K": "K", "R": "R", "E": "E", "G": "G", "L": "L", "N": "N", "Q": "Q",
    "S": "S", "X": "X",
}


def norm(s):
    """Canonical form shared by our labels and by free-text MLLM output."""
    t = str(s).strip().lower()
    t = {"\uf02d": "-", "\u2013": "-", "\u2014": "-", "\u3000": " "}.get(t, t)
    t = re.sub(r"[\s\u3000]+", "", t)
    t = t.replace("/", "").replace("-", "").replace("_", "").replace(".", "")
    t = t.replace("（", "").replace("）", "").replace("(", "").replace(")", "")
    t = t.replace("、", "").replace("：", "").replace(":", "")
    # Chinese prefix -> latin, longest match first
    for cn in sorted(ALIAS, key=len, reverse=True):
        if t.startswith(cn.lower()):
            rest = t[len(cn):]
            return ALIAS[cn] + norm(rest) if re.match(r"^[a-z0-9\u4e00-\u9fff]", rest) else ALIAS[cn]
    return POST_SYNONYM.get(re.sub(r"[^a-z0-9]", "", t).upper(),
                            re.sub(r"[^a-z0-9]", "", t).upper())


def build_vocab(labels):
    """normalized label -> label; plus digit-suffix index for airliner shorthand."""
    exact = {}
    for lb in labels:
        exact.setdefault(norm(lb), lb)
    return exact


DIGIT_ONLY = re.compile(r"^(A3\d{2}|B7\d{2}|BOEING7\d{2}|AIRBUS3\d{2})$")
BARE_AIRLINER = re.compile(r"^(3\d{2}|7\d{2})$")


def resolve(token, exact):
    n = norm(token)
    if not n:
        return None
    if n in exact:
        return exact[n]
    m = DIGIT_ONLY.match(n)
    if m:
        # our labels keep the family letter: AirbusA320 -> AIRBUSA320,
        # Boeing747 -> BOEING747
        for cand in ("AIRBUS" + n.replace("AIRBUS", ""), "BOEING" + n.replace("BOEING", ""),
                     "AIRBUS" + n[1:], "BOEING" + n[1:], n):
            if cand in exact:
                return exact[cand]
    m = BARE_AIRLINER.match(n)
    if m:
        digits = m.group(1)
        cands = (["BOEING" + digits] if digits.startswith("7")
                 else ["AIRBUS" + digits, "AIRBUS" + "A" + digits])
        for cand in cands:
            if cand in exact:
                return exact[cand]
    # tolerate a trailing variant letter (F15A -> F15, SU30SM -> none)
    for k, v in exact.items():
        if len(n) > 2 and (n.startswith(k) or k.startswith(n)) and \
           sum(c.isdigit() for c in n) and n[:2] == k[:2]:
            nd = int(re.search(r"\d+", n).group())
            kd = int(re.search(r"\d+", k).group())
            if nd == kd:
                return v
    return None


def main():
    labels = sorted(d for d in os.listdir(CROP_DIR)
                    if os.path.isdir(os.path.join(CROP_DIR, d)))
    exact = build_vocab(labels)

    domain = {lb: ("civilian" if lb in CIVILIAN else "military") for lb in labels}
    assert len(CIVILIAN) == 11 and set(CIVILIAN) <= set(labels)
    with open(os.path.join(ROOT, "knowledge", "domain_map.json"), "w", encoding="utf-8") as fh:
        json.dump({"n_military": sum(1 for v in domain.values() if v == "military"),
                   "n_civilian": sum(1 for v in domain.values() if v == "civilian"),
                   "civilian_classes": CIVILIAN, "domain": domain},
                  fh, ensure_ascii=False, indent=2)
    print("domain_map.json: %d military / %d civilian"
          % (sum(1 for v in domain.values() if v == "military"), len(CIVILIAN)))

    if not os.path.exists(KB_PATH):
        print("KB not found, skipping confusable pairs")
        return
    kb = json.load(open(KB_PATH, encoding="utf-8"))
    entries = {e["class"]: e for e in kb.get("entries", []) if e.get("ok")}
    print("KB entries: %d" % len(entries))

    graph = defaultdict(lambda: defaultdict(int))
    unresolved, total_refs = defaultdict(int), 0
    for cls, e in entries.items():
        for item in (e["knowledge"].get("similar_class_differences") or []):
            token = re.split(r"[:：]", str(item))[0]
            token = re.sub(r"^\s*与", "", token).strip()
            if not token:
                continue
            total_refs += 1
            hit = resolve(token, exact)
            if hit and hit != cls:
                graph[cls][hit] += 1
            elif not hit:
                unresolved[re.sub(r"\s+", "", token)] += 1

    sym = defaultdict(set)
    for a, nbrs in graph.items():
        for b in nbrs:
            sym[a].add(b)
            sym[b].add(a)
    deg = {k: len(v) for k, v in sym.items()}
    mutual = {k: sorted(v) for k, v in sorted(sym.items())}
    recips = {k: sorted(g for g in v if k in graph.get(g, {})) for k, v in mutual.items()}

    out = {
        "meta": {"source": "Qwen3-VL similar_class_differences, normalized to label vocab",
                 "n_refs": total_refs,
                 "resolution_rate": round(1 - sum(unresolved.values()) / max(total_refs, 1), 3)},
        "confusable_pairs": {k: v for k, v in mutual.items() if v},
        "mutually_named_pairs": {k: v for k, v in recips.items() if v},
        "unresolved_tokens": dict(sorted(unresolved.items(), key=lambda x: -x[1])),
    }
    with open(os.path.join(ROOT, "knowledge", "confusable_pairs.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print("refs=%d resolved=%.1f%% classes_with_pairs=%d/%d"
          % (total_refs, 100 * (1 - sum(unresolved.values()) / max(total_refs, 1)),
             sum(1 for v in mutual.values() if v), len(labels)))
    isolated = [c for c in entries if not mutual.get(c)]
    print("classes with no confusable link: %d %s" % (len(isolated), isolated[:10]))
    top = sorted(deg.items(), key=lambda x: -x[1])[:10]
    print("most confusable:", top)
    print("unresolved head:", dict(sorted(unresolved.items(), key=lambda x: -x[1])[:20]))


if __name__ == "__main__":
    main()
