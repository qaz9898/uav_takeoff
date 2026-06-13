#!/usr/bin/env python3  # 指定使用的解释器为 python3
"""
最优路径跟踪控制器 — 论文式 (16)(24)，无风时可令 wind_est 全零。  # 注释：整体说明

论文：Optimal Path Following for Small Fixed-Wing UAVs Under Wind Disturbances  # 注释：引用论文

坐标约定（与 PX4 LOCAL_POSITION_NED 一致）：  # 注释：坐标约定
  x 北, y 东, z 地；psi 自北顺时针；+phi 为右滚（右翼下压）。  # 注释：坐标、姿态变量约定
附录 ω 符号在该约定下与 PX4 滚转正方向相反，故 φ_c = -atan(Vω/g)。  # 注释：符号约定说明
path_forward_sign 传入后 μ 会按路径正向（θ 增大/减小）换算为 θ̇ 的修正量。  # 注释：路径方向约定
"""
from __future__ import annotations  # 启用新版注解功能

import math  # 导入数学库

import numpy as np  # 导入 numpy 库用于数值计算

ZERO_WIND = {  # 无风条件下的风和加速度估计字典
    'wx': 0.0, 'wy': 0.0, 'wz': 0.0,  # 风速三个分量设为0
    'ax': 0.0, 'ay': 0.0, 'az': 0.0,  # 加速度三个分量设为0
}

class OptimalPathFollower:  # 最优路径跟踪器类
    """论文 Section V 路径跟踪律（r=1）。"""  # 注释：仅支持r=1

    def __init__(self, params, flight_params=None):  # 初始化方法，传入参数
        flight_params = flight_params or {}  # flight_params默认为空字典
        self.Tp = float(params['prediction_horizon'])  # 预测时域 Tp
        self.r = int(params['control_order'])  # 控制阶数 r
        if self.r == 1:  # 只支持阶数1
            self.k0 = 3.0 / (self.Tp ** 2)  # k0参数计算
            self.k1 = 3.0 / self.Tp  # k1参数计算
        else:  # 如果不是r=1抛出异常
            raise NotImplementedError('当前仅实现 control_order r=1')  # 未实现其它阶数
        self.g = 9.81  # 重力加速度
        self.alpha = float(params.get('alpha', 2.0))  # gamma滤波参数
        self.max_roll = math.radians(float(flight_params.get('max_roll', 35.0)))  # 最大滚转角（弧度）
        self.max_gamma = math.radians(float(flight_params.get('max_pitch', 20.0)))  # 最大俯仰角（弧度）

    def circle_geometry(self, theta, radius, center_x, center_y, current_z):  # 生成圆形路径及几何信息
        """圆：x_d=cx+R cosθ, y_d=cy+R sinθ, z_d=当前高度。"""
        xd = center_x + radius * math.cos(theta)
        yd = center_y + radius * math.sin(theta)
        zd = current_z
        nabla_x = -radius * math.sin(theta)  # 对theta一阶导x分量
        nabla_y = radius * math.cos(theta)   # 对theta一阶导y分量
        nabla_z = 0.0  # 对theta一阶导z分量恒为0
        nabla2_x = -radius * math.cos(theta)  # 对theta二阶导x分量
        nabla2_y = -radius * math.sin(theta)  # 对theta二阶导y分量
        nabla2_z = 0.0  # 对theta二阶导z分量恒为0
        return {  # 返回包含所有信息的字典
            'xd': xd, 'yd': yd, 'zd': zd,  # 目标点
            'nabla_x': nabla_x, 'nabla_y': nabla_y, 'nabla_z': nabla_z,  # 一阶导
            'nabla2_x': nabla2_x, 'nabla2_y': nabla2_y, 'nabla2_z': nabla2_z,  # 二阶导
        }

    @staticmethod
    def bow_tie_ds_dtheta(theta, amplitude_x, amplitude_y, x_sign=1.0, y_sign=1.0):  # 蝴蝶结路径的ds/dtheta计算
        """平面蝴蝶结 ds/dθ = ||d(x_d, y_d)/dθ||。"""  # 注释：公式
        s = math.sin(theta)  # 计算sin(theta)
        c2 = math.cos(2.0 * theta)  # 计算cos(2*theta)
        return math.hypot(  # 返回欧氏范数
            abs(float(x_sign)) * amplitude_x * s,  # x方向导数分量
            abs(float(y_sign)) * 2.0 * amplitude_y * c2,  # y方向导数分量
        )

    def bow_tie_geometry(
        self, theta, amplitude_x, amplitude_y, center_x, center_y, current_z,
        *, x_sign=1.0, y_sign=1.0,
    ):  # 生成蝴蝶结路径及几何信息
        """平面蝴蝶结：x_d=cx+sign_x*A cosθ, y_d=cy+sign_y*B sin(2θ), z_d=当前高度。"""  # 注释：描述路径方程
        x_sign = float(x_sign)  # x方向符号
        y_sign = float(y_sign)  # y方向符号
        c = math.cos(theta)  # 计算cos(theta)
        s = math.sin(theta)  # 计算sin(theta)
        s2 = math.sin(2.0 * theta)  # 计算sin(2*theta)
        c2 = math.cos(2.0 * theta)  # 计算cos(2*theta)
        xd = center_x + x_sign * amplitude_x * c  # 计算x方向目标点
        yd = center_y + y_sign * amplitude_y * s2  # 计算y方向目标点
        zd = current_z  # z与当前高度相同
        nabla_x = x_sign * (-amplitude_x * s)  # x对theta一阶导
        nabla_y = y_sign * 2.0 * amplitude_y * c2  # y对theta一阶导
        nabla_z = 0.0  # z对theta一阶导为0
        nabla2_x = x_sign * (-amplitude_x * c)  # x对theta二阶导
        nabla2_y = y_sign * (-4.0 * amplitude_y * s2)  # y对theta二阶导
        nabla2_z = 0.0  # z对theta二阶导为0
        return {  # 返回包含所有信息的字典
            'xd': xd, 'yd': yd, 'zd': zd,  # 目标点
            'nabla_x': nabla_x, 'nabla_y': nabla_y, 'nabla_z': nabla_z,  # 一阶导
            'nabla2_x': nabla2_x, 'nabla2_y': nabla2_y, 'nabla2_z': nabla2_z,  # 二阶导
        }

    def _lie_derivatives(self, state, geom, theta2):  # 计算李导数各项
        theta2 = max(-10.0, min(10.0, float(theta2)))  # 限定theta2范围，防止过大过小
        V = max(float(state['V']), 1.0)  # 速度不小于1m/s
        psi = float(state['psi'])  # 获取偏航角psi
        gamma = float(state['gamma'])  # 获取俯仰角gamma
        roll = float(state.get('roll', 0.0))  # 获取滚转角roll，默认为0
        nx, ny, nz = geom['nabla_x'], geom['nabla_y'], geom['nabla_z']  # 一阶导
        n2x, n2y, n2z = geom['nabla2_x'], geom['nabla2_y'], geom['nabla2_z']  # 二阶导
        cg, sg = math.cos(gamma), math.sin(gamma)  # gamma的余弦和正弦
        cpsi, spsi = math.cos(psi), math.sin(psi)  # psi的余弦和正弦
        tan_phi = math.tan(roll) if abs(roll) < math.radians(89) else math.tan(math.radians(89))  # 防止滚转角过大导致无穷
        psi_dot = (self.g / V) * tan_phi  # 侧向加速度推导的航向角速度

        Lf_hx = V * cpsi * cg - nx * theta2  # x方向一阶李导数
        Lf_hy = V * spsi * cg - ny * theta2  # y方向一阶李导数
        Lf_hz = V * sg - nz * theta2  # z方向一阶李导数

        L2_hx = V * (-spsi * cg * psi_dot) - n2x * theta2 ** 2  # x方向二阶李导数
        L2_hy = V * (cpsi * cg * psi_dot) - n2y * theta2 ** 2  # y方向二阶李导数
        L2_hz = -n2z * theta2 ** 2  # z方向二阶李导数
        return Lf_hx, Lf_hy, Lf_hz, L2_hx, L2_hy, L2_hz  # 返回所有李导数

    def _appendix_control(self, V, psi, gamma, geom, vx, vy, vz):  # 附录控制辅助量
        nx, ny, nz = geom['nabla_x'], geom['nabla_y'], geom['nabla_z']  # 路径导数
        cg, sg = math.cos(gamma), math.sin(gamma)  # gamma余弦和正弦
        cpsi, spsi = math.cos(psi), math.sin(psi)  # psi余弦和正弦
        phi_den = (  # 滚转分母项
            -V ** 2 * sg * cg * nz  # 第1项
            - V ** 2 * cg ** 2 * (cpsi * nx + spsi * ny)  # 第2项
        )
        if abs(phi_den) < 1e-3:  # 防止分母过小
            phi_den = 1e-3 if phi_den >= 0 else -1e-3  # 最小值修正
        omega = (  # ω 控制作动量
            (V * spsi * sg * nz + V * cg * ny) * vx
            - (V * cg * nx + V * cpsi * sg * nz) * vy
            + (V * cpsi * sg * ny - V * spsi * sg * nx) * vz
        ) / phi_den  # 归一化分母
        nu = (  # ν 控制项
            V * cpsi * cg * nz * vx + V * spsi * cg * nz * vy
            - (V * cpsi * cg * nx + V * spsi * cg * ny) * vz
        ) / phi_den  # 归一化
        mu = (  # μ 控制项
            V ** 2 * cpsi * cg ** 2 * vx
            + V ** 2 * spsi * cg ** 2 * vy
            + V ** 2 * sg * cg * vz
        ) / phi_den  # 归一化
        return omega, nu, mu  # 返回三个控制辅助量

    def compute_control_law(
        self,
        state,  # 当前飞行状态字典
        wind_est,  # 风估计（六维字典）
        theta,  # 路径进度参数theta
        theta2,  # 路径参数theta的导数
        *,
        geom,  # 路径几何参数
        path_forward_sign=1.0,  # 路径方向正负号
    ):  # 计算控制指令
        """
        式 (16) 无风：wind_est 全零即可。  # 注释：无风直接用零
        geom 由 circle_geometry / bow_tie_geometry 提供。  # 注释：路径几何输入
        path_forward_sign: +1 表示 θ 增大为沿路径正向，-1 表示 θ 减小为正向。  # 注释：路径正逆
        返回 phi_c, gamma_c, mu, ex, ey, ez（mu 已含 path_forward_sign，可直接 θ̇+=μ·dt）。  # 注释：输出含义
        """
        path_forward_sign = 1.0 if float(path_forward_sign) >= 0.0 else -1.0  # 判断路径正向
        V = max(float(state['V']), 1.0)  # 获取并限制速度
        gamma = float(state['gamma'])  # 获取俯仰角
        ex = state['x'] - geom['xd']  # x偏差
        ey = state['y'] - geom['yd']  # y偏差
        ez = state['z'] - geom['zd']  # z偏差
        Lf_hx, Lf_hy, Lf_hz, L2_hx, L2_hy, L2_hz = self._lie_derivatives(
            state, geom, theta2,  # 计算各方向李导数
        )
        vx = self.k0 * ex + self.k1 * Lf_hx + L2_hx + wind_est.get('ax', 0.0)  # x方向辅助量
        vy = self.k0 * ey + self.k1 * Lf_hy + L2_hy + wind_est.get('ay', 0.0)  # y方向辅助量
        vz = self.k0 * ez + self.k1 * Lf_hz + L2_hz + wind_est.get('az', 0.0)  # z方向辅助量
        vx += self.k1 * wind_est.get('wx', 0.0)  # 考虑x方向风速
        vy += self.k1 * wind_est.get('wy', 0.0)  # 考虑y方向风速
        vz += self.k1 * wind_est.get('wz', 0.0)  # 考虑z方向风速

        omega, nu, mu = self._appendix_control(
            V, state['psi'], gamma, geom, vx, vy, vz,  # 计算附录控制
        )
        # 附录 ω → 滚转：在 NED/ PX4 约定下符号与论文原式相反（见模块说明）  # 注释：符号变号
        phi_c = -math.atan(V * omega / self.g)  # 计算目标滚转角
        gamma_c = gamma + nu / self.alpha  # 计算目标俯仰角
        mu = path_forward_sign * mu  # 正向/反向修正mu
        phi_c = float(np.clip(phi_c, -self.max_roll, self.max_roll))  # 限幅滚转指令
        gamma_c = float(np.clip(gamma_c, -self.max_gamma, self.max_gamma))  # 限幅俯仰指令
        return phi_c, gamma_c, mu, ex, ey, ez  # 返回所有关键控制项及偏差
 
