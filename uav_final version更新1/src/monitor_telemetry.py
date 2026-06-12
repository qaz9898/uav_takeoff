#!/usr/bin/env python3
"""
实时打印 PX4 MAVLink 读数，用于核对单位与数值是否合理。

用法：
  cd "/home/cfx25/uav_final version"
  python3 src/monitor_telemetry.py
  python3 src/monitor_telemetry.py --hz 2    # 2 Hz 刷新
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import yaml

from mavlink_takeoff import TakeoffLink


def load_config():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, 'config', 'params.yaml')) as f:
        return yaml.safe_load(f), root


def _deg(rad):
    return math.degrees(rad)


def print_block(link: TakeoffLink, snap: dict, seq: int):
    sep = '=' * 60
    print(sep)
    print(f'#{seq}  t={time.strftime("%H:%M:%S")}')
    print(sep)

    print('[链路状态]')
    print(f'  armed        = {link._armed}')
    print(f'  main_mode    = {link._main_mode}  (6=OFFBOARD)')
    print(f'  position_ok  = {link._position_ready}')
    print(f'  ned_source   = {"LOCAL_POSITION_NED" if link._local_ned_ready else "GLOBAL+HOME(回退)"}')
    if link._last_status:
        print(f'  last_statustext = {link._last_status!r}')

    print('\n[HOME_POSITION]')
    if link._home_lat is not None:
        print(f'  home lat,lon = {link._home_lat:.7f}°, {link._home_lon:.7f}°')
    else:
        print('  home lat,lon = (尚未收到)')

    print('\n[GLOBAL_POSITION_INT]')
    print('  原始含义: lat/lon=degE7, relative_alt=mm')
    print(f'  lat, lon     = {link._lat_deg:.7f}°, {link._lon_deg:.7f}°')
    print(f'  relative_alt = {link.relative_alt_m:.2f} m  (相对 Home)')
    print(f'  alt_amsl     = {link.alt_amsl_m:.2f} m  (海拔)')

    print('\n[LOCAL_POSITION_NED → 控制用]')
    print('  与 PX4 / 地面站 LOCAL_POSITION_NED 同源')
    print(f'  x,y,z        = {link._ned_x:.2f}, {link._ned_y:.2f}, {link._ned_z:.2f} m')
    print(f'               (z<0 表示在 Home 上方)')
    print(f'  vx,vy,vz     = {link._vx:.2f}, {link._vy:.2f}, {link._vz:.2f} m/s  (NED 地速)')

    print('\n[ATTITUDE]')
    print('  原始含义: roll/pitch/yaw = rad')
    print(f'  roll, pitch, yaw = {_deg(link._roll_rad):+.2f}°, '
          f'{_deg(link._pitch_rad):+.2f}°, {_deg(link._yaw_rad):+.2f}°')

    print('\n[VFR_HUD]')
    print('  原始含义: airspeed/groundspeed=m/s, throttle=%')
    print(f'  airspeed     = {link._airspeed:.2f} m/s')
    print(f'  groundspeed  = {link._groundspeed:.2f} m/s')
    throttle_src = 'HIL_ACTUATOR_CONTROLS' if link._throttle_from_hil else 'VFR_HUD'
    print(f'  throttle     = {link._throttle_pct:.1f} %  ({throttle_src})')

    print('\n[control_snapshot() 控制律用]')
    print(f'  x,y,z        = {snap["x"]:.2f}, {snap["y"]:.2f}, {snap["z"]:.2f} m')
    print(f'  vx,vy,vz     = {snap["vx"]:.2f}, {snap["vy"]:.2f}, {snap["vz"]:.2f} m/s')
    print(f'  V            = {snap["V"]:.2f} m/s')
    print(f'  psi          = {_deg(snap["psi"]):+.2f}°  (机头 ATTITUDE.yaw)')
    print(f'  course       = {_deg(snap["course"]):+.2f}°  (地速航向 atan2(vy,vx))')
    print(f'  gamma        = {_deg(snap["gamma"]):+.2f}°  (航迹角)')
    print(f'  roll         = {_deg(snap["roll"]):+.2f}°  (实测滚转)')
    print(f'  relative_alt = {snap["relative_alt_m"]:.2f} m')
    print(f'  throttle_pct = {snap["throttle_pct"]:.1f} %')
    print(f'  main_mode    = {snap["main_mode"]}')

    v_hor = math.hypot(snap['vx'], snap['vy'])
    print('\n[交叉核对]')
    print(f'  水平地速     = {v_hor:.2f} m/s  (应≈groundspeed {link._groundspeed:.2f})')
    print(f'  高度 -z      = {-snap["z"]:.2f} m  (LOCAL_POSITION_NED)')
    print(f'  GLOBAL rel   = {link.relative_alt_m:.2f} m  (应≈-z)')
    psi_course_diff = _deg(_angle_diff(snap['psi'], snap['course']))
    print(f'  psi-course   = {psi_course_diff:+.2f}°  (侧风时会有偏差)')
    print()


def _angle_diff(a, b):
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def main() -> int:
    parser = argparse.ArgumentParser(description='PX4 MAVLink 遥测实时监视')
    parser.add_argument(
        '--hz', type=float, default=2.0,
        help='打印频率 Hz（默认 2）',
    )
    parser.add_argument(
        '--default-airspeed', type=float, default=None,
        help='control_snapshot 默认空速回退值（默认读 config flight.airspeed）',
    )
    args = parser.parse_args()

    cfg, _ = load_config()
    mcfg = cfg['mavlink']
    default_V = args.default_airspeed
    if default_V is None:
        default_V = float(cfg.get('flight', {}).get('airspeed', 15.0))

    interval = 1.0 / max(args.hz, 0.2)

    link = TakeoffLink(device=mcfg['device'], baud=mcfg['baudrate'])
    if not link.wait_heartbeat(timeout=mcfg.get('heartbeat_timeout', 5)):
        return 1
    link.request_data_stream(rate=int(max(args.hz * 5, 10)))

    print('⏳ 等待 LOCAL_POSITION_NED / HOME_POSITION / GPS（最多 15s）...')
    if not link.wait_position(timeout=15.0):
        print('⚠️  尚未收到位置，仍继续打印（字段可能为 0）')

    print(f'✅ 开始监视 @ {args.hz:.1f} Hz，Ctrl+C 退出')
    print(f'   串口 {mcfg["device"]} @ {mcfg["baudrate"]}')

    seq = 0
    try:
        while True:
            link.drain_messages(limit=100)
            snap = link.control_snapshot(default_airspeed=default_V)
            seq += 1
            print_block(link, snap, seq)
            time.sleep(interval)
    except KeyboardInterrupt:
        print('\n✅ 监视结束')
        return 0


if __name__ == '__main__':
    sys.exit(main())
