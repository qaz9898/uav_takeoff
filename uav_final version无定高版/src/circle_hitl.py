#!/usr/bin/env python3
"""
HITL 平面路径跟随（圆 / 蝴蝶结），PX4 OFFBOARD 姿态执行。

圆:     x_d = cx + R cosθ,  y_d = cy + R sinθ
蝴蝶结: x_d = cx + A cosθ,  y_d = cy + B sin(2θ)

流程：起飞 → OFFBOARD → 锁路径 → θ 积分 → 发 φ_c / 固定俯仰。
"""
from __future__ import annotations

import math
import os
import sys
import time
from datetime import datetime
from typing import List, Optional, TextIO

import yaml

from mavlink_takeoff import TakeoffLink, _PX4_MAIN_MODE_OFFBOARD
from path_follower import OptimalPathFollower, ZERO_WIND
from wind_observer import (
    WindDisturbanceObserver,
    format_wind_log_line,
    wind_magnitude,
)

G = 9.81


class PathLogWriter:
    """每次路径任务写入 path_logs/logN.txt。"""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.file_path, self.log_name = _alloc_path_log_file(root_dir)
        self._fp: Optional[TextIO] = open(self.file_path, 'w', encoding='utf-8')

    def write_line(self, line: str = '') -> None:
        if self._fp is None:
            return
        self._fp.write(line + '\n')
        self._fp.flush()

    def write_lines(self, lines: List[str]) -> None:
        for line in lines:
            self.write_line(line)

    def close(self, *, elapsed_sec: float, ok: bool) -> None:
        if self._fp is None:
            return
        status = '完成' if ok else '中止'
        self.write_line(f'# 任务{status} 历时={elapsed_sec:.1f}s')
        self._fp.close()
        self._fp = None
        print(f'📝 路径日志已保存: {self.file_path}')


def _alloc_path_log_file(root_dir: str) -> tuple[str, str]:
    os.makedirs(root_dir, exist_ok=True)
    n = 1
    while os.path.exists(os.path.join(root_dir, f'log{n}.txt')):
        n += 1
    log_name = f'log{n}.txt'
    return os.path.join(root_dir, log_name), log_name


def _path_log_root(cfg: dict, project_root: str) -> str:
    s5 = cfg.get('stage5', {})
    rel = str(s5.get('path_log_root', 'path_logs'))
    if os.path.isabs(rel):
        return rel
    return os.path.join(project_root, rel)


def load_config():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, 'config', 'params.yaml')) as f:
        return yaml.safe_load(f), root


def _angle_diff(a, b):
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def _path_forward_sign(cfg_path):
    sign = float(cfg_path.get('path_forward_sign', 1.0))
    return 1.0 if sign >= 0.0 else -1.0


def _resolve_path_type(cfg, override: Optional[str] = None) -> str:
    if override:
        t = str(override).lower().replace('-', '_')
        return 'circle' if t == 'circle' else 'bow_tie'
    t = str(cfg.get('path', {}).get('path_type', 'bow_tie')).lower().replace('-', '_')
    return 'circle' if t == 'circle' else 'bow_tie'


# ---------------------------------------------------------------------------
# 圆
# ---------------------------------------------------------------------------

def min_circle_radius_for_speed(V_mps, max_roll_deg, roll_fraction=0.88):
    phi = math.radians(max_roll_deg * roll_fraction)
    tan_phi = max(math.tan(phi), 0.12)
    return (max(V_mps, 1.0) ** 2) / (G * tan_phi)


def build_circle_at_lock(lock_snap, cfg):
    p = cfg['path']
    flight = cfg.get('flight', {})
    R_nom = float(p.get('radius_m', 100.0))
    max_roll_deg = float(flight.get('max_roll', 35.0))
    roll_frac = float(p.get('radius_roll_fraction', 0.88))
    path_sign = _path_forward_sign(p)
    theta_dot_max = float(p.get('theta_dot_max', 0.35))

    V_lock = max(float(lock_snap.get('V', flight.get('airspeed', 15.0))), 8.0)
    R_min = min_circle_radius_for_speed(V_lock, max_roll_deg, roll_frac)
    scale_with_speed = bool(p.get('radius_scale_with_speed', True))
    R = max(R_nom, R_min) if scale_with_speed else R_nom

    x0, y0 = lock_snap['x'], lock_snap['y']
    psi = float(lock_snap.get('psi', 0.0))
    course = float(lock_snap.get('course', psi)) if V_lock > 3.0 else psi

    if p.get('circle_align_heading', True):
        theta0 = course + math.pi / 2.0
    elif 'circle_theta0_deg' in p:
        theta0 = math.radians(float(p['circle_theta0_deg']))
    else:
        theta0 = float(p.get('circle_theta0', 0.0))

    cx = x0 - R * math.cos(theta0)
    cy = y0 - R * math.sin(theta0)
    theta2_0 = path_sign * min(abs(V_lock / R), theta_dot_max)

    return {
        'kind': 'circle',
        'radius': R,
        'radius_nom': R_nom,
        'radius_min_lock': R_min,
        'radius_scaled': scale_with_speed and R > R_nom + 0.5,
        'radius_roll_fraction': roll_frac,
        'center_x': cx,
        'center_y': cy,
        'theta0': theta0,
        'path_forward_sign': path_sign,
        'theta2_0': theta2_0,
        'V_lock_mps': V_lock,
    }


def _circle_geom(controller, path, theta, z):
    return controller.circle_geometry(
        theta, path['radius'], path['center_x'], path['center_y'], z,
    )


def _circle_theta2_nom(V, path, theta_dot_max):
    R = max(float(path['radius']), 1.0)
    return path['path_forward_sign'] * min(abs(V / R), theta_dot_max)


def _circle_tangent_ned(path, theta, theta2):
    R = path['radius']
    nx = -R * math.sin(theta)
    ny = R * math.cos(theta)
    return nx * theta2, ny * theta2


# ---------------------------------------------------------------------------
# 蝴蝶结
# ---------------------------------------------------------------------------

def _bow_tie_x_sign(cfg_path):
    sign = float(cfg_path.get('bow_tie_x_sign', 1.0))
    return 1.0 if sign >= 0.0 else -1.0


def _bow_tie_y_sign(cfg_path):
    sign = float(cfg_path.get('bow_tie_y_sign', 1.0))
    return 1.0 if sign >= 0.0 else -1.0


def _resolve_theta0(p, lock_snap=None):
    """
    切入路径参数 θ₀（论文 x=A cosθ, y=B sin(2θ)，NED x=北 y=东）。

    lower_loop_east  → θ=+45°，下环东侧
    lower_loop_south → θ=-45°，下环西侧最南点
    """
    preset = str(p.get('entry_point', '')).strip().lower()
    if preset in ('lower_loop_east', 'lower_loop_east_north', 'east'):
        return math.pi / 4.0
    if preset in ('lower_loop_south', 'south', 'lower_south'):
        return -math.pi / 4.0
    if preset in ('upper_loop_east', 'upper_east'):
        return 3.0 * math.pi / 4.0
    if preset in ('upper_loop_west', 'upper_west'):
        return -3.0 * math.pi / 4.0
    if 'bow_tie_theta0_deg' in p:
        return math.radians(float(p['bow_tie_theta0_deg']))
    return float(p.get('bow_tie_theta0', 0.0))


def build_bow_tie_at_lock(lock_snap, cfg):
    """在切入点对齐路径：θ₀ 处 (x_d, y_d) = (x₀, y₀)。"""
    p = cfg['path']
    flight = cfg.get('flight', {})
    amplitude_x = float(p.get('bow_tie_amplitude_x', 140.0))
    amplitude_y = float(p.get('bow_tie_amplitude_y', 45.0))
    theta0 = _resolve_theta0(p, lock_snap)
    x_sign = _bow_tie_x_sign(p)
    y_sign = _bow_tie_y_sign(p)
    path_sign = _path_forward_sign(p)
    theta_dot_max = float(p.get('theta_dot_max', 0.35))

    x0, y0 = lock_snap['x'], lock_snap['y']
    cx = x0 - x_sign * amplitude_x * math.cos(theta0)
    cy = y0 - y_sign * amplitude_y * math.sin(2.0 * theta0)

    V_lock = max(float(lock_snap.get('V', flight.get('airspeed', 15.0))), 8.0)
    ds_dtheta = max(
        OptimalPathFollower.bow_tie_ds_dtheta(
            theta0, amplitude_x, amplitude_y, x_sign=x_sign, y_sign=y_sign,
        ),
        1e-3,
    )
    theta2_0 = path_sign * min(abs(V_lock / ds_dtheta), theta_dot_max)

    return {
        'kind': 'bow_tie',
        'amplitude_x': amplitude_x,
        'amplitude_y': amplitude_y,
        'x_sign': x_sign,
        'y_sign': y_sign,
        'center_x': cx,
        'center_y': cy,
        'theta0': theta0,
        'path_forward_sign': path_sign,
        'theta2_0': theta2_0,
        'V_lock_mps': V_lock,
    }


def _bow_tie_geom(controller, path, theta, z):
    return controller.bow_tie_geometry(
        theta,
        path['amplitude_x'],
        path['amplitude_y'],
        path['center_x'],
        path['center_y'],
        z,
        x_sign=path['x_sign'],
        y_sign=path['y_sign'],
    )


def _bow_tie_theta2_nom(V, theta, path, theta_dot_max):
    ds = max(
        OptimalPathFollower.bow_tie_ds_dtheta(
            theta, path['amplitude_x'], path['amplitude_y'],
            x_sign=path['x_sign'], y_sign=path['y_sign'],
        ),
        1e-3,
    )
    return path['path_forward_sign'] * min(abs(V / ds), theta_dot_max)


def _bow_tie_tangent_ned(path, theta, theta2):
    nx = path['x_sign'] * (-path['amplitude_x'] * math.sin(theta))
    ny = path['y_sign'] * 2.0 * path['amplitude_y'] * math.cos(2.0 * theta)
    return nx * theta2, ny * theta2


# ---------------------------------------------------------------------------
# 通用
# ---------------------------------------------------------------------------

def build_path_at_lock(lock_snap, cfg, path_type):
    if path_type == 'circle':
        return build_circle_at_lock(lock_snap, cfg)
    return build_bow_tie_at_lock(lock_snap, cfg)


def _path_geom(controller, path, theta, z):
    if path['kind'] == 'circle':
        return _circle_geom(controller, path, theta, z)
    return _bow_tie_geom(controller, path, theta, z)


def _theta2_nom(V, theta, path, theta_dot_max):
    if path['kind'] == 'circle':
        return _circle_theta2_nom(V, path, theta_dot_max)
    return _bow_tie_theta2_nom(V, theta, path, theta_dot_max)


def _path_tangent_ned(path, theta, theta2):
    if path['kind'] == 'circle':
        return _circle_tangent_ned(path, theta, theta2)
    return _bow_tie_tangent_ned(path, theta, theta2)


def _ned_dir_label(vx, vy, *, min_speed=0.3):
    parts = []
    if abs(vx) >= min_speed:
        parts.append(f'{"北" if vx > 0 else "南"}{abs(vx):.1f}m/s')
    if abs(vy) >= min_speed:
        parts.append(f'{"东" if vy > 0 else "西"}{abs(vy):.1f}m/s')
    if not parts:
        return '近零'
    return ' + '.join(parts)


def _print_path_sketch(path, theta, theta2):
    vx, vy = _path_tangent_ned(path, theta, theta2)
    th_deg = math.degrees(theta)
    th2_deg = math.degrees(theta2)
    print('路径走向（θ 积分方向）:')
    print(
        f'  θ₀={th_deg:+.1f}°  θ̇={th2_deg:+.2f}°/s  '
        f'切向速度 NED: {_ned_dir_label(vx, vy)}'
    )
    if path['kind'] == 'circle':
        print(f'  圆 R={path["radius"]:.1f}m  圆心=({path["center_x"]:.1f},{path["center_y"]:.1f})')
    else:
        print('  示意图: 下环东侧(θ=+45°) → 向北爬升，东向坐标略减小(向西收)')
    print('  NED: x↑北  y→东  | err=(pos-path) | path.y↓=向西 path.y↑=向东')


def format_wind_disturbance_rejection_status(
    compensate_control: bool,
    observer_enabled: bool,
) -> str:
    """本次实验风扰抗扰开关说明（控制台与 path log 共用）。"""
    if compensate_control and observer_enabled:
        return '风扰抗扰: ON'
    return '风扰抗扰: OFF'


def _format_runtime_status(
    elapsed, phi_c, V, ex, ey, snap, geom, theta, theta2, mu,
    path=None, flight=None,
    wind_hat: Optional[dict] = None,
    wind_mag_avg: Optional[float] = None,
) -> List[str]:
    h_geom = math.hypot(ex, ey)
    alt_m = -snap['z']
    alt_tag = f' H={alt_m:.1f} m' if math.isfinite(alt_m) else ''
    pos_age = snap.get('pos_age_ms', -1.0)
    r_tag = ''
    if path and path.get('kind') == 'circle':
        R = path['radius']
        R_nom = path.get('radius_nom', R)
        if abs(R - R_nom) > 0.5:
            r_tag = f' R={R:.0f}m(cfg{R_nom:.0f}→放大)'
        else:
            r_tag = f' R={R:.0f}m(cfg{R_nom:.0f})'
        if flight is not None:
            R_req = min_circle_radius_for_speed(
                V,
                float(flight.get('max_roll', 35.0)),
                float(path.get('radius_roll_fraction', 0.88)),
            )
            if R_req > R + 0.5:
                r_tag += f' V需≥{R_req:.0f}m'
    lines = [
        (
            f'[{elapsed:5.1f}s] φ={math.degrees(phi_c):+5.1f}° '
            f'V={V:.1f} m/s{alt_tag} |h|={h_geom:.1f} m{r_tag}'
        ),
        (
            f'         pos=({snap["x"]:.1f}, {snap["y"]:.1f}) '
            f'path=({geom["xd"]:.1f}, {geom["yd"]:.1f}) '
            f'err=({ex:+.1f}, {ey:+.1f}) '
            f'pos_age={pos_age:.0f}ms'
        ),
        (
            f'         θ={math.degrees(theta):+.1f}° '
            f'θ̇={theta2:.4f} μ={mu:+.3f} '
            f'ψ={math.degrees(snap["psi"]):+.1f}°'
        ),
    ]
    if wind_hat is not None and wind_mag_avg is not None:
        lines.append(f'         {format_wind_log_line(wind_hat, wind_mag_avg)}')
    return lines


def _emit_log(lines: List[str], log_writer: Optional[PathLogWriter] = None) -> None:
    for line in lines:
        print(line)
    if log_writer is not None:
        log_writer.write_lines(lines)


def _print_runtime_status(
    elapsed, phi_c, V, ex, ey, snap, geom, theta, theta2, mu,
    path=None, flight=None, log_writer: Optional[PathLogWriter] = None,
    wind_hat: Optional[dict] = None,
    wind_mag_avg: Optional[float] = None,
) -> None:
    _emit_log(
        _format_runtime_status(
            elapsed, phi_c, V, ex, ey, snap, geom, theta, theta2, mu,
            path=path, flight=flight,
            wind_hat=wind_hat, wind_mag_avg=wind_mag_avg,
        ),
        log_writer,
    )


def enter_offboard(link, cfg):
    oc = dict(cfg.get('offboard', {}))
    tc = cfg.get('takeoff', {})
    s5 = cfg.get('stage5', {})
    default_V = float(cfg.get('flight', {}).get('airspeed', 15.0))

    oc['hold_after_sec'] = float(
        cfg.get('circle', {}).get('hold_after_offboard_sec', 0.0),
    )

    if s5.get('skip_takeoff', False):
        print('ℹ️  跳过起飞：假定已在空中')
        if not link.wait_position(timeout=s5.get('position_timeout_sec', 15)):
            return False
        if not link.try_set_offboard_mode(
            warmup_sec=oc.get('warmup_sec', 2.0),
            timeout=oc.get('confirm_timeout', 15.0),
            hold_after_sec=0.0,
            stream_hz=float(oc.get('stream_hz', 25.0)),
            warmup_style=oc.get('warmup_style', 'cruise'),
            cruise_speed_mps=float(oc.get('cruise_speed_mps', 12.0)),
            lookahead_m=float(oc.get('lookahead_m', 40.0)),
        ):
            return False
    else:
        if not link.wait_position(timeout=tc.get('position_timeout_sec', 15)):
            return False
        if not link.run_takeoff_climb_offboard_cruise(tc, oc):
            print('❌ 起飞链未完成')
            return False

    link.drain_messages()
    link._path_lock_snap = link.control_snapshot(default_airspeed=default_V)
    snap = link._path_lock_snap
    print(
        f'📌 OFFBOARD 切入点 NED=({snap["x"]:.1f}, {snap["y"]:.1f}, '
        f'z={snap["z"]:.1f}) V≈{snap["V"]:.1f} m/s'
    )
    return True


def run_path_loop(link, cfg, path_type=None):
    s5 = cfg.get('stage5', {})
    flight = cfg.get('flight', {})
    p = cfg['path']
    path_type = _resolve_path_type(cfg, path_type)

    stream_hz = float(s5.get('stream_hz', 25.0))
    dt = 1.0 / stream_hz
    duration = float(s5.get('duration_sec', 120.0))
    thrust = float(s5.get('thrust', 0.35))
    print_interval = float(s5.get('print_interval_sec', 0.5))
    alpha_trim = float(flight.get('alpha_trim', 0.05))
    default_V = float(flight.get('airspeed', 15.0))
    theta_dot_max = float(p.get('theta_dot_max', 0.35))
    mu_max = float(p.get('mu_max', 1.0))
    phi_rate_limit = math.radians(float(s5.get('phi_rate_limit_deg_s', 15.0)))
    obs_cfg = cfg.get('observer', {})
    wind_observer_enabled = bool(obs_cfg.get('enabled', True))
    compensate_requested = bool(obs_cfg.get('compensate_control', False))
    wind_rejection_status = format_wind_disturbance_rejection_status(
        compensate_requested, wind_observer_enabled,
    )
    wind_compensate = compensate_requested
    wind_observer = (
        WindDisturbanceObserver.from_config(obs_cfg) if wind_observer_enabled else None
    )
    if wind_compensate and wind_observer is None:
        print('⚠️  compensate_control=true 但观测器未启用，风补偿无效（仍用 ZERO_WIND）')
        wind_compensate = False

    controller = OptimalPathFollower(cfg['controller'], flight)

    lock_snap = getattr(link, '_path_lock_snap', None)
    if lock_snap is None:
        link.drain_messages()
        lock_snap = link.control_snapshot(default_airspeed=default_V)

    path = build_path_at_lock(lock_snap, cfg, path_type)
    if path['kind'] == 'circle':
        cfg_r = float(p.get('radius_m', 100.0))
        lock_r = path['radius']
        scale_note = '按速度放大' if path.get('radius_scaled') else '与 cfg 一致'
        print(
            f'📋 params.yaml radius_m={cfg_r:.0f}  '
            f'V_lock={path["V_lock_mps"]:.1f}m/s → 锁定 R={lock_r:.1f}m ({scale_note})'
        )
    theta = path['theta0']
    path_sign = path['path_forward_sign']
    z_hold = lock_snap['z']
    theta2 = path['theta2_0']

    geom0 = _path_geom(controller, path, theta, lock_snap['z'])
    h0 = math.hypot(lock_snap['x'] - geom0['xd'], lock_snap['y'] - geom0['yd'])

    if path['kind'] == 'circle':
        title = '无风平面绕圆：论文式 (16)(24)'
        R, R_nom = path['radius'], path['radius_nom']
        eq_line = f'x_d=cx+{R:.0f}cosθ, y_d=cy+{R:.0f}sinθ'
        if path['radius_scaled']:
            r_note = (
                f'R={R:.1f}m（名义{R_nom:.0f}m，V_lock={path["V_lock_mps"]:.1f}→按速度放大）'
            )
        else:
            r_note = f'R={R:.1f}m（名义{R_nom:.0f}m，未放大）'
    else:
        title = '无风平面蝴蝶结：论文式 (16)(24)'
        xsgn = path['x_sign']
        ysgn = path['y_sign']
        x_op = '-' if xsgn < 0 else '+'
        y_op = '-' if ysgn < 0 else '+'
        eq_line = (
            f'x_d=cx{x_op}{path["amplitude_x"]:.0f}cosθ, '
            f'y_d=cy{y_op}{path["amplitude_y"]:.0f}sin(2θ)'
        )

    nx0, ny0 = _path_tangent_ned(path, theta, theta2)
    tan0 = math.degrees(math.atan2(ny0, nx0)) if math.hypot(nx0, ny0) > 1e-6 else float('nan')

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_writer: Optional[PathLogWriter] = None
    if s5.get('save_path_log', True):
        log_writer = PathLogWriter(_path_log_root(cfg, project_root))
        kind_label = '圆轨迹' if path['kind'] == 'circle' else '蝴蝶结'
        log_writer.write_line(f'# {kind_label}  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        log_writer.write_line(f'# 文件={log_writer.log_name}  计划时长={duration:.0f}s  记录间隔={print_interval:.1f}s')
        if path['kind'] == 'circle':
            cfg_r = float(p.get('radius_m', 100.0))
            log_writer.write_line(
                f'# params.yaml radius_m={cfg_r:.0f}  锁定R={path["radius"]:.1f}m'
            )
        log_writer.write_line('')

    startup_lines = [
        '=' * 50,
        f' {title}',
        '=' * 50,
        eq_line,
    ]
    if path['kind'] == 'circle':
        startup_lines.append(r_note)
        startup_lines.append(
            f'  V_lock={path["V_lock_mps"]:.1f}m/s  R_min@V_lock={path["radius_min_lock"]:.1f}m'
        )
    startup_lines.extend([
        f'锚点 cx,cy=({path["center_x"]:.1f}, {path["center_y"]:.1f})',
        (
            f'θ₀={math.degrees(theta):.1f}° θ̇₀={theta2:.4f} rad/s '
            f'|h|₀={h0:.1f} m 定高 z={-z_hold:.1f} m'
        ),
        '|h| = 当前 NED 位置到路径点 (x_d,y_d) 的水平距离（≈√(ex²+ey²)）',
        f'固定油门 thrust={thrust:.2f}  Tp={controller.Tp:.0f} s',
    ])
    ep = cfg['path'].get('entry_point', '')
    if ep:
        startup_lines.append(f'起始途径点 preset={ep} 路径切线≈{tan0:.1f}°（0°=北，+90°=东）')
    _emit_log(startup_lines, log_writer)
    _print_path_sketch(path, theta, theta2)
    if log_writer is not None:
        vx, vy = _path_tangent_ned(path, theta, theta2)
        log_writer.write_lines([
            '路径走向（θ 积分方向）:',
            (
                f'  θ₀={math.degrees(theta):+.1f}°  θ̇={math.degrees(theta2):+.2f}°/s  '
                f'切向速度 NED: {_ned_dir_label(vx, vy)}'
            ),
            '  NED: x↑北  y→东  | err=(pos-path) | path.y↓=向西 path.y↑=向东',
            '',
        ])

    if wind_observer is not None:
        _emit_log([
            f'风扰观测器: ON  l1={wind_observer.l1} l2={wind_observer.l2} '
            f'l3={wind_observer.l3} L={wind_observer.L}  {wind_rejection_status}',
            '  每拍日志第4行: wind̂=(wx,wy,wz) m/s  |wind̂|=... m/s  |wind̂|_avg=... m/s',
        ], log_writer)
    else:
        _emit_log([f'风扰观测器: OFF  {wind_rejection_status}'], log_writer)

    if link._main_mode != _PX4_MAIN_MODE_OFFBOARD:
        _emit_log([f'⚠️  当前模式={link._main_mode}，未在 OFFBOARD'], log_writer)

    t0 = time.monotonic()
    last_log = 0.0
    phi_prev = 0.0
    wind_mag_sum = 0.0
    wind_mag_count = 0
    done_msg = '✅ 圆轨迹任务结束' if path['kind'] == 'circle' else '✅ 蝴蝶结路径任务结束'
    ok = False

    try:
        while time.monotonic() - t0 < duration:
            if link._main_mode != _PX4_MAIN_MODE_OFFBOARD:
                _emit_log(['❌ 已退出 OFFBOARD，中止路径跟随'], log_writer)
                return False

            snap = link.control_snapshot(default_airspeed=default_V)
            V = max(snap['V'], 1.0)
            state = {
                'x': snap['x'], 'y': snap['y'], 'z': snap['z'],
                'vx': snap['vx'], 'vy': snap['vy'], 'vz': snap['vz'],
                'V': V, 'psi': snap['psi'], 'gamma': snap['gamma'],
                'roll': snap['roll'],
            }

            wind_hat = None
            if wind_observer is not None:
                wind_hat = wind_observer.update(state, dt)

            geom = _path_geom(controller, path, theta, snap['z'])
            theta2_nom = _theta2_nom(V, theta, path, theta_dot_max)
            theta2_ctrl = 0.7 * theta2 + 0.3 * theta2_nom

            wind_est = (
                wind_hat if (wind_compensate and wind_hat is not None) else ZERO_WIND
            )
            phi_c, _gamma_c, mu, ex, ey, ez = controller.compute_control_law(
                state, wind_est, theta, theta2_ctrl, geom=geom,
                path_forward_sign=path_sign,
            )

            mu = max(-mu_max, min(mu_max, mu))
            theta2 += mu * dt
            theta2 = path_sign * min(abs(theta2), theta_dot_max)
            theta += theta2 * dt

            if phi_rate_limit > 0:
                dphi = _angle_diff(phi_c, phi_prev)
                max_dphi = phi_rate_limit * dt
                if abs(dphi) > max_dphi:
                    phi_c = phi_prev + math.copysign(max_dphi, dphi)
                phi_prev = phi_c

            link.send_onboard_heartbeat()
            link.send_attitude_setpoint(
                roll=phi_c,
                pitch=alpha_trim,
                yaw=None,
                thrust=thrust,
            )
            link.drain_messages()

            elapsed = time.monotonic() - t0
            if elapsed - last_log >= print_interval:
                wind_mag_avg = None
                if wind_hat is not None:
                    w_mag = wind_magnitude(wind_hat)
                    wind_mag_sum += w_mag
                    wind_mag_count += 1
                    wind_mag_avg = wind_mag_sum / wind_mag_count
                _print_runtime_status(
                    elapsed, phi_c, V, ex, ey, snap, geom, theta, theta2, mu,
                    path=path, flight=flight, log_writer=log_writer,
                    wind_hat=wind_hat, wind_mag_avg=wind_mag_avg,
                )
                last_log = elapsed

            time.sleep(dt)

        ok = True
        _emit_log([done_msg], log_writer)
        return True
    except KeyboardInterrupt:
        _emit_log(['', '⚠️  用户中断'], log_writer)
        return False
    finally:
        if log_writer is not None:
            log_writer.close(elapsed_sec=time.monotonic() - t0, ok=ok)


run_circle_loop = run_path_loop


def main(path_type: Optional[str] = None) -> int:
    cfg, _ = load_config()
    mcfg = cfg['mavlink']
    oc = cfg.get('offboard', {})
    resolved = _resolve_path_type(cfg, path_type)

    link = TakeoffLink(device=mcfg['device'], baud=mcfg['baudrate'])
    if not link.wait_heartbeat(timeout=mcfg.get('heartbeat_timeout', 5)):
        return 1
    link.request_telemetry_streams(rate_hz=int(oc.get('stream_hz', 25)))

    if not enter_offboard(link, cfg):
        return 1
    ok = run_path_loop(link, cfg, path_type=resolved)
    link.exit_offboard_to_hold()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
