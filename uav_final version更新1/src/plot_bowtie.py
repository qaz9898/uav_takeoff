#!/usr/bin/env python3
"""本地预览蝴蝶结路径走向（NED：x=北，y=东）。用法: python3 src/plot_bowtie.py"""
from __future__ import annotations

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

from circle_hitl import build_bow_tie_at_lock, load_config
from path_follower import OptimalPathFollower


def main() -> int:
    cfg, root = load_config()
    lock = {
        'x': 150.0,
        'y': 0.0,
        'z': -30.0,
        'V': float(cfg.get('flight', {}).get('airspeed', 15.0)),
    }
    path = build_bow_tie_at_lock(lock, cfg)
    ctrl = OptimalPathFollower(cfg['controller'], cfg.get('flight', {}))

    thetas = np.linspace(-math.pi, math.pi, 1000)
    xs = path['center_x'] + path['x_sign'] * path['amplitude_x'] * np.cos(thetas)
    ys = path['center_y'] + path['y_sign'] * path['amplitude_y'] * np.sin(2.0 * thetas)

    th0 = path['theta0']
    t2 = path['theta2_0']
    seg = np.linspace(th0, th0 + t2 * 30.0, 80)
    xs_seg = path['center_x'] + path['x_sign'] * path['amplitude_x'] * np.cos(seg)
    ys_seg = path['center_y'] + path['y_sign'] * path['amplitude_y'] * np.sin(2.0 * seg)

    fig, ax = plt.subplots(figsize=(7, 9))
    ax.plot(ys, xs, 'r-', lw=1.5, label='bow-tie')
    ax.plot(ys_seg, xs_seg, 'b-', lw=2.5, label='path sweep (~30s)')
    ax.plot(lock['y'], lock['x'], 'go', ms=10, label='lock')
    ax.plot(ys_seg[0], xs_seg[0], 'bo', ms=8)
    ax.annotate(
        '',
        xy=(ys_seg[5], xs_seg[5]),
        xytext=(ys_seg[0], xs_seg[0]),
        arrowprops=dict(arrowstyle='->', color='blue', lw=2),
    )
    ax.set_xlabel('East y [m]')
    ax.set_ylabel('North x [m]')
    ax.set_title('Bow-tie path preview (NED)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    out = os.path.join(root, 'bowtie_preview.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f'已保存 {out}')
    print(f'θ₀={math.degrees(th0):.1f}°  θ̇={t2:.4f} rad/s  entry={cfg["path"].get("entry_point")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
