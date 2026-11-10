#! /bin/bash
#!/bin/bash
set -e

python aircon_auto_prompt.py ./prompt/aircon_prompt/v2_prompt/normal.txt --condition L3_normal

python aircon_auto_prompt.py ./prompt/aircon_prompt/v2_prompt/j1.txt --condition L3_j1

python aircon_auto_prompt.py ./prompt/aircon_prompt/v2_prompt/j2.txt --condition L3_j2

python aircon_auto_prompt.py ./prompt/aircon_prompt/v2_prompt/j3.txt --condition L3_j3

echo "L3の全実験が終了しました。"