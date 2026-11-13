#! /bin/bash
#!/bin/bash
set -e

python aircon_auto_prompt.py ./prompt/aircon_prompt/v2_prompt/normal.txt --condition L3_normal --log-file aircon_logs/desktop_log/L3_normal.jsonl

python aircon_auto_prompt.py ./prompt/aircon_prompt/v2_prompt/j1.txt --condition L3_j1 --log-file aircon_logs/desktop_log/L3_j1.jsonl

python aircon_auto_prompt.py ./prompt/aircon_prompt/v2_prompt/j2.txt --condition L3_j2 --log-file aircon_logs/desktop_log/L3_j2.jsonl

python aircon_auto_prompt.py ./prompt/aircon_prompt/v2_prompt/j3.txt --condition L3_j3 --log-file aircon_logs/desktop_log/L3_j3.jsonl

echo "L3の全実験が終了しました。"
