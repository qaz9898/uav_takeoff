#!/usr/bin/env python3
"""
风扰观测器独立监视 — 仅打印观测器估计的风速，不修改控制律。

用法：
  cd "/home/cfx25/uav_final version"
  python3 src/run_wind_monitor.py
"""
from __future__ import annotations

import sys
import time

import yaml

from mavlink_takeoff import TakeoffLink
from wind_observer import WindDisturbanceObserver, format_wind_log_line, wind_magnitude


def load_config():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, 'config', 'params.yaml')) as f:
        return yaml.safe_load(f), root


def main() -> int:
    cfg, _ = load_config()
    mcfg = cfg['mavlink']
    oc = cfg.get('offboard', {})
    obs_cfg = cfg.get('observer', {})
    duration = float(obs_cfg.get('monitor_duration_sec', 120.0))
    stream_hz = float(obs_cfg.get('stream_hz', oc.get('stream_hz', 25.0)))
    print_iv = float(obs_cfg.get('print_interval_sec', 0.5))
    dt = 1.0 / stream_hz
    default_V = float(cfg.get('flight', {}).get('airspeed', 15.0))

    link = TakeoffLink(device=mcfg['device'], baud=mcfg['baudrate'])
    if not link.wait_heartbeat(timeout=mcfg.get('heartbeat_timeout', 5)):
        return 1
    link.request_telemetry_streams(rate_hz=int(stream_hz))
    if not link.wait_position(timeout=cfg.get('takeoff', {}).get('position_timeout_sec', 15)):
        return 1

    observer = WindDisturbanceObserver.from_config(obs_cfg)
    print('=' * 60)
    print(' 风扰观测器监视（论文式 20–23）')
    print('=' * 60)
    print(
        f' 增益 l1={observer.l1} l2={observer.l2} l3={observer.l3} L={observer.L}  '
        f'Hz={stream_hz:.0f}  时长={duration:.0f}s'
    )
    print('  NED: x北 y东 z地  wind̂=(w_x,w_y,w_z)  |wind̂|=矢量模长')
    print('')

    t0 = time.monotonic()
    last_print = 0.0
    wind_mag_sum = 0.0
    wind_mag_count = 0
    try:
        while time.monotonic() - t0 < duration:
            snap = link.control_snapshot(default_airspeed=default_V)
            wind_hat = observer.update(snap, dt)

            elapsed = time.monotonic() - t0
            if elapsed - last_print >= print_iv:
                last_print = elapsed
                w_mag = wind_magnitude(wind_hat)
                wind_mag_sum += w_mag
                wind_mag_count += 1
                w_mag_avg = wind_mag_sum / wind_mag_count
                print(f'[{elapsed:5.1f}s] {format_wind_log_line(wind_hat, w_mag_avg)}')

            time.sleep(max(0.0, dt - (time.monotonic() - t0 - elapsed)))
    except KeyboardInterrupt:
        print('\n⚠️  用户中断')
    print('✅ 风扰监视结束')
    return 0


if __name__ == '__main__':
    sys.exit(main())
