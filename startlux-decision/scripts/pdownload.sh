#!/bin/bash
# 多连接并行分块下载（走本地代理，从 huggingface.co 直源）
# 用法: pdownload.sh <hf仓库路径> <文件名> <输出文件> [并行数]
#   例: pdownload.sh startlux-models/StartLux-Decision-v1-4B-Q8_0-GGUF \
#         StartLux-Decision-4B-Q8_0.gguf models/StartLux-Decision-v1-4B-Q8_0-GGUF/StartLux-Decision-4B-Q8_0.gguf 8
set -u
# 两种用法:
#   pdownload.sh <hf仓库> <文件名> <输出> [并行数]   —— HF, 需刷新签名 URL
#   pdownload.sh <完整URL> <忽略> <输出> [并行数]    —— 直链(GitHub 等), 第一参数以 http 开头
PROXY="http://127.0.0.1:7897"
if [[ "$1" == http* ]]; then
  BASE="$1"
  refresh() { echo "$BASE"; }            # 直链无签名, 不用刷新
else
  REPO_PATH="$1"; FILENAME="$2"
  BASE="https://huggingface.co/${REPO_PATH}/resolve/main/${FILENAME}"
  refresh() { curl -sI --max-time 20 -x "$PROXY" "$BASE" | sed -n 's/^[Ll]ocation: //p' | tr -d '\r'; }
fi
OUT="$3"; NPARTS="${4:-8}"

get_url() { refresh; }

SIZE=$(curl -sIL --max-time 30 -x "$PROXY" "$BASE" | sed -n 's/^[Cc]ontent-[Ll]ength: //p' | tr -d '\r' | tail -1)
[ -z "$SIZE" ] && { echo "ERROR: 无法获取文件大小"; exit 1; }
echo "total size: $SIZE bytes, parts: $NPARTS"

CHUNK=$(( (SIZE + NPARTS - 1) / NPARTS ))
PDIR="$(dirname "$OUT")/.parts"; PF="$(basename "$OUT")"
mkdir -p "$PDIR"

download_part() {
  local i=$1 start=$2 end=$3
  local part="$PDIR/${PF}.part$i"
  local want=$(( end - start + 1 ))
  local attempt have
  for attempt in 1 2 3 4 5 6; do
    have=0; [ -f "$part" ] && have=$(stat -c %s "$part")
    if [ "$have" -ge "$want" ]; then echo "part$i done ($have/$want)"; return 0; fi
    local url; url=$(get_url)
    [ -z "$url" ] && { echo "part$i: 获取签名URL失败,重试$attempt"; sleep 3; continue; }
    curl -sL --max-time 3600 -x "$PROXY" -r $((start+have))-"$end" -o - "$url" >> "$part" 2>/dev/null
  done
  have=0; [ -f "$part" ] && have=$(stat -c %s "$part")
  echo "part$i final: $have/$want"
  [ "$have" -ge "$want" ]
}

PIDS=()
for ((i=0; i<NPARTS; i++)); do
  s=$(( i * CHUNK )); e=$(( s + CHUNK - 1 )); [ $e -ge $SIZE ] && e=$(( SIZE - 1 ))
  [ $s -ge $SIZE ] && break
  download_part "$i" "$s" "$e" & PIDS+=($!)
done

FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || { echo "part进程 $p 失败"; FAIL=1; }; done
[ $FAIL -eq 1 ] && { echo "ERROR: 部分分块失败，可重跑本脚本断点续传"; exit 2; }

> "$OUT.tmp"
for ((i=0; i<NPARTS; i++)); do
  s=$(( i * CHUNK )); [ $s -ge $SIZE ] && break
  cat "$PDIR/${PF}.part$i" >> "$OUT.tmp"
done
GOT=$(stat -c %s "$OUT.tmp")
if [ "$GOT" -eq "$SIZE" ]; then
  mv "$OUT.tmp" "$OUT"; rm -rf "$PDIR"
  echo "OK: $OUT ($GOT bytes)"
else
  echo "ERROR: 合并后大小不符 $GOT != $SIZE"; exit 3
fi
