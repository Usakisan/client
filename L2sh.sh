#! /bin/bash
#!/bin/bash
set -e

python aircon_auto_prompt.py ./prompt/aircon_prompt/v2_prompt/normal.txt --condition L2_normal

python aircon_auto_prompt.py ./prompt/aircon_prompt/v2_prompt/j1.txt --condition L2_j1

python aircon_auto_prompt.py ./prompt/aircon_prompt/v2_prompt/j2.txt --condition L2_j2

python aircon_auto_prompt.py ./prompt/aircon_prompt/v2_prompt/j3.txt --condition L2_j3

echo "L2の全実験が終了しました。"