#!/usr/bin/env python3  # 指定使用的解释器为 python3
"""
最优路径跟踪控制器 — 论文式 (16)(24)，无风时可令 wind_est 全零。

论文：Optimal Path Following for Small Fixed-Wing UAVs Under Wind Disturbances

坐标约定（与 PX4 LOCAL_POSITION_NED 一致）：
  x 北, y 东, z 地；psi 自北顺时针；+phi 为右滚（右翼下压）。
附录 ω 符号在该约定下与 PX4 滚转正方向相反，故 φ_c = -atan(Vω/g)。
path_forward_sign 传入后 μ 会按路径正向（θ 增大/减小）换算为 θ̇ 的修正量。
"""
from __future__ import annotations  # 启用新版注解功能

import math  # 导入数学库

import numpy as np  # 导入 numpy 库用于数值计算

ZERO_WIND = {  # 无风条件下的风和加速度估计字典
    'wx': 0.0, 'wy': 0.0, 'wz': 0.0,
    'ax': 0.0, 'ay': 0.0, 'az': 0.0,
}


class OptimalPathFollower:  # 最优路径跟踪器类
    """论文 Section V 路径跟踪律（r=1）。"""

    def __init__(self, params, flight_params=None):  # 初始化方法，传入参数
        flight_params = flight_params or {}  # 如果未提供 flight_params 使用空字典
        self.Tp = float(params['prediction_horizon'])  # 预测时域 Tp
        self.r = int(params['control_order'])  # 控制阶数 r
        if self.r == 1:  # 仅支持 r=1
            self.k0 = 3.0 / (self.Tp ** 2)  # 跟踪律参数 k0
            self.k1 = 3.0 / self.Tp         # 跟踪律参数 k1
        else:
            raise NotImplementedError('当前仅实现 control_order r=1')  # 其它控制阶未实现
        self.g = 9.81  # 重力加速度
        self.alpha = float(params.get('alpha', 2.0))  # gamma 指令滤波参数
        self.max_roll = math.radians(float(flight_params.get('max_roll', 35.0)))   # 最大滚转角（弧度）
        self.max_gamma = math.radians(float(flight_params.get('max_pitch', 20.0))) # 最大俯仰角（弧度）

    def circle_geometry(self, theta, radius, center_x, center_y, current_z):  # 生成圆形路径及几何信息
        """圆：x_d=cx+R cosθ, y_d=cy+R sinθ, z_d=当前高度。"""
        xd = center_x + radius * math.cos(theta)        # 圆路径 x 目标点
        yd = center_y + radius * math.sin(theta)        # 圆路径 y 目标点
        zd = current_z                                  # 路径高度等于当前高度
        nabla_x = -radius * math.sin(theta)             # 路径对 θ 的一阶导 x
        nabla_y = radius * math.cos(theta)              # 路径对 θ 的一阶导 y
        nabla_z = 0.0                                   # 路径对 θ 的一阶导 z
        nabla2_x = -radius * math.cos(theta)            # 路径对 θ 的二阶导 x
        nabla2_y = -radius * math.sin(theta)            # 路径对 θ 的二阶导 y
        nabla2_z = 0.0                                  # 路径对 θ 的二阶导 z
        return {  # 返回字典
            'xd': xd, 'yd': yd, 'zd': zd,
            'nabla_x': nabla_x, 'nabla_y': nabla_y, 'nabla_z': nabla_z,
            'nabla2_x': nabla2_x, 'nabla2_y': nabla2_y, 'nabla2_z': nabla2_z,
        }

    @staticmethod
    def bow_tie_ds_dtheta(theta, amplitude_x, amplitude_y, x_sign=1.0, y_sign=1.0):
        """平面蝴蝶结 ds/dθ = ||d(x_d, y_d)/dθ||。"""
        s = math.sin(theta)
        c2 = math.cos(2.0 * theta)
        return math.hypot(
            abs(float(x_sign)) * amplitude_x * s,
            abs(float(y_sign)) * 2.0 * amplitude_y * c2,
        )

    def bow_tie_geometry(
        self, theta, amplitude_x, amplitude_y, center_x, center_y, current_z,
        *, x_sign=1.0, y_sign=1.0,
    ):
        """平面蝴蝶结：x_d=cx+sign_x*A cosθ, y_d=cy+sign_y*B sin(2θ), z_d=当前高度。"""
        x_sign = float(x_sign)
        y_sign = float(y_sign)
        c = math.cos(theta)
        s = math.sin(theta)
        s2 = math.sin(2.0 * theta)
        c2 = math.cos(2.0 * theta)
        xd = center_x + x_sign * amplitude_x * c
        yd = center_y + y_sign * amplitude_y * s2
        zd = current_z
        nabla_x = x_sign * (-amplitude_x * s)
        nabla_y = y_sign * 2.0 * amplitude_y * c2
        nabla_z = 0.0
        nabla2_x = x_sign * (-amplitude_x * c)
        nabla2_y = y_sign * (-4.0 * amplitude_y * s2)
        nabla2_z = 0.0
        return {  # 返回字典
            'xd': xd, 'yd': yd, 'zd': zd,
            'nabla_x': nabla_x, 'nabla_y': nabla_y, 'nabla_z': nabla_z,
            'nabla2_x': nabla2_x, 'nabla2_y': nabla2_y, 'nabla2_z': nabla2_z,
        }

    def _lie_derivatives(self, state, geom, theta2):  # 计算李导数项
        theta2 = max(-10.0, min(10.0, float(theta2)))             # 限制 theta2 在 [-10, 10] 范围
        V = max(float(state['V']), 1.0)                           # 速度下限不小于 1 m/s
        psi = float(state['psi'])                                 # 偏航角
        gamma = float(state['gamma'])                             # 俯仰角
        roll = float(state.get('roll', 0.0))                      # 滚转角，默认 0
        nx, ny, nz = geom['nabla_x'], geom['nabla_y'], geom['nabla_z']    # 取一阶导
        n2x, n2y, n2z = geom['nabla2_x'], geom['nabla2_y'], geom['nabla2_z']  # 取二阶导
        cg, sg = math.cos(gamma), math.sin(gamma)                 # 俯仰角余弦和正弦
        cpsi, spsi = math.cos(psi), math.sin(psi)                 # 偏航角余弦和正弦
        tan_phi = math.tan(roll) if abs(roll) < math.radians(89) else math.tan(math.radians(89))  # 避免 tan(90) 无穷大
        psi_dot = (self.g / V) * tan_phi                          # 偏航角速度

        Lf_hx = V * cpsi * cg - nx * theta2                       # 水平X的李导数
        Lf_hy = V * spsi * cg - ny * theta2                       # 水平Y的李导数
        Lf_hz = V * sg - nz * theta2                              # 垂直Z的李导数

        L2_hx = V * (-spsi * cg * psi_dot) - n2x * theta2 ** 2    # X方向二阶李导数
        L2_hy = V * (cpsi * cg * psi_dot) - n2y * theta2 ** 2     # Y方向二阶李导数
        L2_hz = -n2z * theta2 ** 2                                # Z方向二阶李导数
        return Lf_hx, Lf_hy, Lf_hz, L2_hx, L2_hy, L2_hz           # 返回所有李导数项

    def _appendix_control(self, V, psi, gamma, geom, vx, vy, vz):  # 计算附录中的辅助控制量
        nx, ny, nz = geom['nabla_x'], geom['nabla_y'], geom['nabla_z']  # 取一阶导
        cg, sg = math.cos(gamma), math.sin(gamma)                       # 俯仰角余弦和正弦
        cpsi, spsi = math.cos(psi), math.sin(psi)                       # 偏航角余弦和正弦
        phi_den = (                                                   # 控制分母
            -V ** 2 * sg * cg * nz
            - V ** 2 * cg ** 2 * (cpsi * nx + spsi * ny)
        )
        if abs(phi_den) < 1e-3:            # 分母过小防止除零
            phi_den = 1e-3 if phi_den >= 0 else -1e-3
        omega = (                          # omega 控制项
            (V * spsi * sg * nz + V * cg * ny) * vx
            - (V * cg * nx + V * cpsi * sg * nz) * vy
            + (V * cpsi * sg * ny - V * spsi * sg * nx) * vz
        ) / phi_den
        nu = (                             # nu 控制项
            V * cpsi * cg * nz * vx + V * spsi * cg * nz * vy
            - (V * cpsi * cg * nx + V * spsi * cg * ny) * vz
        ) / phi_den
        mu = (                             # mu 控制项
            V ** 2 * cpsi * cg ** 2 * vx
            + V ** 2 * spsi * cg ** 2 * vy
            + V ** 2 * sg * cg * vz
        ) / phi_den
        return omega, nu, mu               # 返回 omega, nu, mu

    def compute_control_law(
        self,
        state,
        wind_est,
        theta,
        theta2,
        *,
        geom,
        path_forward_sign=1.0,
    ):
        """
        式 (16) 无风：wind_est 全零即可。
        geom 由 circle_geometry / bow_tie_geometry 提供。
        path_forward_sign: +1 表示 θ 增大为沿路径正向，-1 表示 θ 减小为正向。
        返回 phi_c, gamma_c, mu, ex, ey, ez（mu 已含 path_forward_sign，可直接 θ̇+=μ·dt）。
        """
        path_forward_sign = 1.0 if float(path_forward_sign) >= 0.0 else -1.0
        V = max(float(state['V']), 1.0)
        gamma = float(state['gamma'])
        ex = state['x'] - geom['xd']
        ey = state['y'] - geom['yd']
        ez = state['z'] - geom['zd']
        Lf_hx, Lf_hy, Lf_hz, L2_hx, L2_hy, L2_hz = self._lie_derivatives(
            state, geom, theta2,
        )
        vx = self.k0 * ex + self.k1 * Lf_hx + L2_hx + wind_est.get('ax', 0.0)
        vy = self.k0 * ey + self.k1 * Lf_hy + L2_hy + wind_est.get('ay', 0.0)
        vz = self.k0 * ez + self.k1 * Lf_hz + L2_hz + wind_est.get('az', 0.0)
        vx += self.k1 * wind_est.get('wx', 0.0)
        vy += self.k1 * wind_est.get('wy', 0.0)
        vz += self.k1 * wind_est.get('wz', 0.0)

        omega, nu, mu = self._appendix_control(
            V, state['psi'], gamma, geom, vx, vy, vz,
        )
        # 附录 ω → 滚转：在 NED/ PX4 约定下符号与论文原式相反（见模块说明）
        phi_c = -math.atan(V * omega / self.g)
        gamma_c = gamma + nu / self.alpha
        mu = path_forward_sign * mu
        phi_c = float(np.clip(phi_c, -self.max_roll, self.max_roll))
        gamma_c = float(np.clip(gamma_c, -self.max_gamma, self.max_gamma))
        return phi_c, gamma_c, mu, ex, ey, ez
