#!/usr/bin/env python3
"""
PX4 固定翼一键起飞（自 uav_takeoff 移植）

流程：Hold → 解锁 → Takeoff → 爬升到目标高度 → OFFBOARD → 平稳前飞

用法：
  cd "/home/cfx25/uav_final version"
  python3 src/run_takeoff.py
"""
from __future__ import annotations

import os
import sys

import yaml

from mavlink_takeoff import TakeoffLink


def load_config():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, 'config', 'params.yaml')) as f:
        return yaml.safe_load(f), root


def main() -> int:
    cfg, _ = load_config()
    mcfg = cfg['mavlink']
    tc = cfg.get('takeoff', {})
    oc = cfg.get('offboard', {})

    link = TakeoffLink(device=mcfg['device'], baud=mcfg['baudrate'])
    if not link.wait_heartbeat(timeout=mcfg.get('heartbeat_timeout', 5)):
        return 1
    link.request_data_stream(rate=int(oc.get('stream_hz', 10)))

    if not link.wait_position(timeout=tc.get('position_timeout_sec', 15)):
        return 1

    print('=' * 50)
    print(' 固定翼起飞 → OFFBOARD → 前飞')
    print('=' * 50)
    print(
        f'  目标高度 {tc.get("altitude_m", 30):.0f} m | '
        f'前飞 {oc.get("hold_after_sec", 5):.0f} s'
    )

    try:
        if not link.run_takeoff_climb_offboard_cruise(tc, oc):
            print('❌ 起飞链未完成')
            return 1
        print('✅ 起飞链完成')
        return 0
    except KeyboardInterrupt:
        print('\n⚠️  用户中断，尝试切回 Hold')
        link.exit_offboard_to_hold()
        return 130


if __name__ == '__main__':
    sys.exit(main())
