#! /bin/bash
#!/bin/bash
set -e

python aircon_auto_prompt.py ./prompt/aircon_prompt/normal.txt --condition L0_normal

python aircon_auto_prompt.py ./prompt/aircon_prompt/distractors.txt --condition L0_distractors

python aircon_auto_prompt.py ./prompt/aircon_prompt/normal_logic.txt --condition L0_normal_logic

python aircon_auto_prompt.py ./prompt/aircon_prompt/prefix.txt --condition L0_prefix

echo "L0の全実験が終了しました。"