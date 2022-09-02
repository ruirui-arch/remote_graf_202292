#!/usr/bin/env bash
config=$1
cd /ghome/zhangkd/newJob/nerf/graf-main
python train.py  ${config}
