#!/usr/bin/env bash
# Fetch the source dataset straight from Kaggle, on the server.
#
# The dataset is public: a2015003713/militaryaircraftdetectiondataset. Pulling it
# server-side runs at ~9 MB/s, versus ~100-150 KB/s pushing it up the 10.1.1.x
# VPN tunnel from a laptop -- roughly 60x faster and no 8 GB upload.
#
# Auth: a KGAT_ API token in ~/.kaggle/access_token, presented through a 600-perm
# curl config file so the token never appears in `ps` on this shared machine.
# The download endpoint answers 302, so -L is required.
#
# Usage:  bash tools/kaggle_download.sh [workdir]
set -u

URL="https://www.kaggle.com/api/v1/datasets/download/a2015003713/militaryaircraftdetectiondataset"
WORK="${1:-$HOME/dmt/_kaggle}"
HDR="$HOME/.kaggle/hdr.cnf"
PY="${PY:-python3}"

[[ -f "$HDR" ]] || { echo "missing $HDR -- see README_SERVER.md"; exit 1; }
mkdir -p "$WORK" && cd "$WORK" || exit 1

echo "[$(date +%H:%M:%S)] downloading -> $WORK/military.zip"
curl -sSL --config "$HDR" --retry 8 --retry-delay 15 --retry-all-errors \
     -C - -o military.zip "$URL"
rc=$?
echo "curl_exit=$rc" > dl.status
if [[ $rc -ne 0 ]]; then echo "curl failed rc=$rc"; exit $rc; fi
ls -la military.zip

echo "[$(date +%H:%M:%S)] verifying the zip and listing its layout"
"$PY" - <<'PY'
import sys, zipfile
try:
    z = zipfile.ZipFile("military.zip")
except Exception as e:
    print("ZIP CORRUPT:", e); sys.exit(1)
bad = z.testzip()
if bad is not None:
    print("ZIP CORRUPT at member:", bad); sys.exit(1)
names = z.namelist()
jpg = [n for n in names if n.lower().endswith(".jpg")]
csv = [n for n in names if n.lower().endswith(".csv")]
print("zip OK -- %d entries, %d jpg, %d csv" % (len(names), len(jpg), len(csv)))
tops = sorted({n.split("/")[0] for n in names})
print("top level:", tops[:8])
for n in names[:6]:
    print("  ", n)
print("first jpg:", jpg[0] if jpg else "(none)")
PY
echo "[$(date +%H:%M:%S)] done"
