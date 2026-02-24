#!/usr/bin/env bash
set -euo pipefail

# =========================
# YouTube 下载脚本（可改常量）
# =========================

# 1) 下载模式：single=只下载一个视频（即使URL里有playlist参数） / all=下载整个合集
PLAYLIST_MODE="single"   # single | all

# 2) YouTube URL
YOUTUBE_URL="https://www.youtube.com/watch?v=Zs8jUFaqtCI"

# 3) 输出目录（默认 output/videos）
OUTPUT_DIR="output/videos"

# =========================
# 执行逻辑
# =========================

if ! command -v yt-dlp >/dev/null 2>&1; then
  echo "❌ 未检测到 yt-dlp，请先安装：pip install yt-dlp"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [[ "$PLAYLIST_MODE" != "single" && "$PLAYLIST_MODE" != "all" ]]; then
  echo "❌ PLAYLIST_MODE 只能是 single 或 all"
  exit 1
fi

# noplaylist: single=true, all=false
if [[ "$PLAYLIST_MODE" == "single" ]]; then
  NOPLAYLIST_FLAG="--no-playlist"
else
  NOPLAYLIST_FLAG="--yes-playlist"
fi

echo "▶ 开始下载"
echo "  mode: $PLAYLIST_MODE"
echo "  url : $YOUTUBE_URL"
echo "  out : $OUTPUT_DIR"

# 文件名用视频标题（若重名会自动加序号）
# 输出模板：标题.扩展名
yt-dlp \
  "$NOPLAYLIST_FLAG" \
  -f "bv*+ba/b" \
  --merge-output-format mp4 \
  -o "$OUTPUT_DIR/%(title)s.%(ext)s" \
  "$YOUTUBE_URL"

echo "✅ 下载完成：$OUTPUT_DIR"
