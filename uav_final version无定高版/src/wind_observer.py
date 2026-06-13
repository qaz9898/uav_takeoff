#!/usr/bin/env python3
"""
论文 Section IV 风扰观测器（式 20–23），三轴独立三阶滑模观测器。

运动学 (1)：
  ẋ = V cos ψ cos γ + w_x
  ẏ = V sin ψ cos γ + w_y
  ż = V sin γ + w_z

单轴（以 x 为例）：
  x̂̇ = V_x + ŵ + l1·L^(1/3)·|x−x̂|^(2/3)·sign(x−x̂)
  ŵ̇ = â + l2·L^(1/2)·|x−x̂|^(1/2)·sign(x−x̂)
  â̇ = l3·L·sign(x−x̂)

默认增益与论文 Section V/VI 一致：l1=2, l2=1.5, l3=1.5, L=1。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional


def _signed_power(z: float, alpha: float) -> float:
    if abs(z) < 1e-12:
        return 0.0
    return math.copysign(abs(z) ** alpha, z)


def _sign(z: float) -> float:
    if z > 0.0:
        return 1.0
    if z < 0.0:
        return -1.0
    return 0.0


def wind_magnitude(wind_hat: Mapping[str, float]) -> float:
    return math.sqrt(
        wind_hat['wx'] ** 2 + wind_hat['wy'] ** 2 + wind_hat['wz'] ** 2
    )


def format_wind_log_line(wind_hat: Mapping[str, float], wind_mag_avg: float) -> str:
    """与 run_wind_monitor / circle_hitl 运行时第 4 行一致。"""
    w_mag = wind_magnitude(wind_hat)
    return (
        f'wind̂=({wind_hat["wx"]:+.2f},{wind_hat["wy"]:+.2f},'
        f'{wind_hat["wz"]:+.2f}) m/s  |wind̂|={w_mag:.2f} m/s  '
        f'|wind̂|_avg={wind_mag_avg:.2f} m/s'
    )


def air_kinematic_velocity(V: float, psi: float, gamma: float) -> tuple[float, float, float]:
    """式 (1) 中空速在 NED 惯性系下的贡献 [m/s]。"""
    cg, sg = math.cos(gamma), math.sin(gamma)
    cpsi, spsi = math.cos(psi), math.sin(psi)
    return V * cpsi * cg, V * spsi * cg, V * sg


@dataclass
class _AxisState:
    pos_hat: float = 0.0
    wind_hat: float = 0.0
    wind_dot_hat: float = 0.0


@dataclass
class WindDisturbanceObserver:
    """论文式 (20)(22)(23) 风扰观测器。"""

    l1: float = 2.0
    l2: float = 1.5
    l3: float = 1.5
    L: float = 1.0
    _x: _AxisState = field(default_factory=_AxisState)
    _y: _AxisState = field(default_factory=_AxisState)
    _z: _AxisState = field(default_factory=_AxisState)
    _initialized: bool = False

    @classmethod
    def from_config(cls, cfg: Optional[Mapping] = None) -> 'WindDisturbanceObserver':
        cfg = cfg or {}
        return cls(
            l1=float(cfg.get('l1', 2.0)),
            l2=float(cfg.get('l2', 1.5)),
            l3=float(cfg.get('l3', 1.5)),
            L=float(cfg.get('L', 1.0)),
        )

    def reset(self, state: Optional[Mapping[str, float]] = None) -> None:
        self._x = _AxisState()
        self._y = _AxisState()
        self._z = _AxisState()
        self._initialized = False
        if state is not None:
            self._sync_position(state)

    def _sync_position(self, state: Mapping[str, float]) -> None:
        self._x.pos_hat = float(state['x'])
        self._y.pos_hat = float(state['y'])
        self._z.pos_hat = float(state['z'])

    def _step_axis(
        self,
        axis: _AxisState,
        pos_meas: float,
        vel_air: float,
        dt: float,
    ) -> None:
        e = pos_meas - axis.pos_hat
        L13 = self.L ** (1.0 / 3.0)
        L12 = math.sqrt(self.L)

        pos_hat_dot = (
            vel_air + axis.wind_hat
            + self.l1 * L13 * _signed_power(e, 2.0 / 3.0)
        )
        wind_hat_dot = (
            axis.wind_dot_hat
            + self.l2 * L12 * _signed_power(e, 0.5)
        )
        wind_dot_hat_dot = self.l3 * self.L * _sign(e)

        axis.pos_hat += pos_hat_dot * dt
        axis.wind_hat += wind_hat_dot * dt
        axis.wind_dot_hat += wind_dot_hat_dot * dt

    def update(self, state: Mapping[str, float], dt: float) -> Dict[str, float]:
        """
        用当前遥测推进观测器。

        state 需含 x,y,z,vx,vy,vz,V,psi,gamma（与 control_snapshot 一致）。
        返回与 path_follower.ZERO_WIND 相同键的字典，供后续风补偿使用。
        """
        dt = max(float(dt), 1e-6)
        if not self._initialized:
            self._sync_position(state)
            self._initialized = True

        V = max(float(state.get('V', 1.0)), 1.0)
        psi = float(state.get('psi', 0.0))
        gamma = float(state.get('gamma', 0.0))
        vx_a, vy_a, vz_a = air_kinematic_velocity(V, psi, gamma)

        self._step_axis(self._x, float(state['x']), vx_a, dt)
        self._step_axis(self._y, float(state['y']), vy_a, dt)
        self._step_axis(self._z, float(state['z']), vz_a, dt)

        return {
            'wx': self._x.wind_hat,
            'wy': self._y.wind_hat,
            'wz': self._z.wind_hat,
            'ax': self._x.wind_dot_hat,
            'ay': self._y.wind_dot_hat,
            'az': self._z.wind_dot_hat,
        }

    def as_dict(self) -> Dict[str, float]:
        return {
            'wx': self._x.wind_hat,
            'wy': self._y.wind_hat,
            'wz': self._z.wind_hat,
            'ax': self._x.wind_dot_hat,
            'ay': self._y.wind_dot_hat,
            'az': self._z.wind_dot_hat,
        }
