#! /bin/bash
#!/bin/bash
set -e

python aircon_auto_prompt.py ./prompt/aircon_prompt/v2_prompt/normal.txt --condition L4_normal --log-file aircon_logs/desktop_log/L4_normal.jsonl

python aircon_auto_prompt.py ./prompt/aircon_prompt/v2_prompt/j1.txt --condition L4_j1 --log-file aircon_logs/desktop_log/L4_j1.jsonl

python aircon_auto_prompt.py ./prompt/aircon_prompt/v2_prompt/j2.txt --condition L4_j2 --log-file aircon_logs/desktop_log/L4_j2.jsonl

python aircon_auto_prompt.py ./prompt/aircon_prompt/v2_prompt/j3.txt --condition L4_j3 --log-file aircon_logs/desktop_log/L4_j3.jsonl

echo "L4の全実験が終了しました。"
