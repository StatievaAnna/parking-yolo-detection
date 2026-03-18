#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MEDIAMTX_BIN="/usr/local/bin/mediamtx"
if [[ ! -x "$DEFAULT_MEDIAMTX_BIN" && -x "$ROOT_DIR/mediamtx/mediamtx" ]]; then
  DEFAULT_MEDIAMTX_BIN="$ROOT_DIR/mediamtx/mediamtx"
fi
MEDIAMTX_BIN="${MEDIAMTX_BIN:-$DEFAULT_MEDIAMTX_BIN}"
MEDIAMTX_CONFIG="${MEDIAMTX_CONFIG:-$ROOT_DIR/mediamtx/mediamtx.yml}"
VIDEO_FILE_1="${VIDEO_FILE_1:-$ROOT_DIR/cvpipeline/crop_flows/4/1.mp4}"
VIDEO_FILE_2="${VIDEO_FILE_2:-$ROOT_DIR/cvpipeline/crop_flows/4/2.mp4}"
VIDEO_FILE_3="${VIDEO_FILE_3:-$ROOT_DIR/cvpipeline/crop_flows/4/3.mp4}"
VIDEO_FILE_4="${VIDEO_FILE_4:-$ROOT_DIR/cvpipeline/crop_flows/4/4.mp4}"
RTSP_HOST="${RTSP_HOST:-127.0.0.1}"
PIPELINE_STREAM_HOST="${PIPELINE_STREAM_HOST:-$RTSP_HOST}"
if [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
  DEFAULT_PYTHON_BIN="$ROOT_DIR/venv/bin/python"
else
  DEFAULT_PYTHON_BIN="python3"
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
PIPELINE_CMD="${PIPELINE_CMD:-$PYTHON_BIN cvpipeline/src/pipeline.py}"

if [[ ! -x "$MEDIAMTX_BIN" ]]; then
  echo "Ошибка: mediamtx не найден или не исполняемый: $MEDIAMTX_BIN"
  exit 1
fi

if [[ ! -f "$MEDIAMTX_CONFIG" ]]; then
  echo "Ошибка: конфиг mediamtx не найден: $MEDIAMTX_CONFIG"
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Ошибка: интерпретатор не найден в PATH: $PYTHON_BIN"
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Ошибка: ffmpeg не найден в PATH"
  exit 1
fi

for f in "$VIDEO_FILE_1" "$VIDEO_FILE_2" "$VIDEO_FILE_3" "$VIDEO_FILE_4"; do
  if [[ ! -f "$f" ]]; then
    echo "Ошибка: видеофайл не найден: $f"
    echo "Подсказка: проверьте файлы crop_flows/1/1.mp4 ... 4.mp4"
    exit 1
  fi
done

PIDS=()

start_bg_process() {
  local command="$1"
  bash -lc "$command" &
  local pid=$!
  PIDS+=("$pid")
}

cleanup() {
  local exit_code=$?

  trap - EXIT INT TERM

  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done

  sleep 1

  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
  done

  exit "$exit_code"
}

trap cleanup EXIT INT TERM

wait_for_port() {
  local host="$1"
  local port="$2"
  local retries="${3:-20}"
  local delay="${4:-0.5}"

  for _ in $(seq 1 "$retries"); do
    if nc -z "$host" "$port" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$delay"
  done
  return 1
}

FFMPEG_CODEC_OPTS="-map 0:v:0 -an -c:v libx264 -pix_fmt yuv420p -preset veryfast -tune zerolatency -x264-params keyint=30:min-keyint=15 -rtsp_transport tcp -f rtsp"

echo "1/4 Запускаю mediamtx..."
start_bg_process "\"$MEDIAMTX_BIN\" \"$MEDIAMTX_CONFIG\""

echo "2/4 Жду пока mediamtx начнет слушать порт 8554..."
if ! wait_for_port "127.0.0.1" "8554" 40 0.5; then
  echo "Ошибка: mediamtx не поднялся на 127.0.0.1:8554. Завершаю запуск."
  exit 1
fi

echo "3/4 Запускаю локальные RTSP потоки cam1..cam4 на $RTSP_HOST:8554..."
start_bg_process "ffmpeg -re -stream_loop -1 -i \"$VIDEO_FILE_1\" $FFMPEG_CODEC_OPTS rtsp://$RTSP_HOST:8554/cam1"
start_bg_process "ffmpeg -re -stream_loop -1 -i \"$VIDEO_FILE_2\" $FFMPEG_CODEC_OPTS rtsp://$RTSP_HOST:8554/cam2"
start_bg_process "ffmpeg -re -stream_loop -1 -i \"$VIDEO_FILE_3\" $FFMPEG_CODEC_OPTS rtsp://$RTSP_HOST:8554/cam3"
start_bg_process "ffmpeg -re -stream_loop -1 -i \"$VIDEO_FILE_4\" $FFMPEG_CODEC_OPTS rtsp://$RTSP_HOST:8554/cam4"
sleep 2

echo "4/4 Запускаю pipeline..."
cd "$ROOT_DIR"
PIPELINE_STREAM_HOST="$PIPELINE_STREAM_HOST" bash -lc "$PIPELINE_CMD"
