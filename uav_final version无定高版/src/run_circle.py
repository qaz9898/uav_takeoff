#!/usr/bin/env python3
"""
无风平面圆轨迹 HITL：起飞 → OFFBOARD → 论文式 (16)(24) 路径跟随。

用法：
  cd "/home/cfx25/uav_final version"
  python3 src/run_circle.py
"""
import sys

from circle_hitl import main

if __name__ == '__main__':
    print('=' * 50)
    print(' 无风平面圆轨迹 HITL（论文最优路径跟随）')
    print('=' * 50)
    sys.exit(main(path_type='circle'))
