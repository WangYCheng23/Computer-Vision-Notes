#!/bin/bash
cd /project/train/src_repo
rm -rf /project/train/models/*
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

python scripts/convert_tusimple.py --data_root=/home/data/67

python train.py configs/tusimple.py --data_root=/home/data/67 --epoch=10 --batch_size=32
