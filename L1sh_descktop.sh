#! /bin/bash
#!/bin/bash
set -e

python aircon_auto_prompt.py ./prompt/aircon_prompt/v2_prompt/normal.txt --condition L0_normal

python aircon_auto_prompt.py ./prompt/aircon_prompt/v2_prompt/j1.txt --condition L0_j1

python aircon_auto_prompt.py ./prompt/aircon_prompt/v2_prompt/j2.txt --condition L0_j2

python aircon_auto_prompt.py ./prompt/aircon_prompt/v2_prompt/j3.txt --condition L0_j3

echo "L0の全実験が終了しました。"