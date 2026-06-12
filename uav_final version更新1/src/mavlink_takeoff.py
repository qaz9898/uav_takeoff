#!/usr/bin/env python3
"""
PX4 固定翼 MAVLink 通信层（TakeoffLink）

职责（与论文算法分离，属于「执行层」）：
  - 读：LOCAL_POSITION_NED（与 PX4 一致）、HOME_POSITION、姿态、空速 → control_snapshot()
  - 写：SET_POSITION_TARGET（位置/盘旋 setpoint）、SET_ATTITUDE_TARGET（φ,θ,油门）
  - 模式：解锁、TAKEOFF、OFFBOARD、Hold

Telem2 上 SERIAL_CONTROL 通常进不了 NSH，故用 MAVLink 等价 commander：
  commander arm -f  → MAV_CMD_COMPONENT_ARM_DISARM (param2=21196)
  commander takeoff → 主模式 AUTO.TAKEOFF
  OFFBOARD          → 需先 >1s 预热 setpoint，再发模式切换

stage5 主要用到：
  try_set_offboard_mode / control_snapshot / send_attitude_setpoint / exit_offboard_to_hold
"""
import math
import time
from pymavlink import mavutil

_PX4_FORCE_ARM = 21196.0
_PX4_MAIN_AUTO = 4
_PX4_SUB_TAKEOFF = 2
_PX4_SUB_LOITER = 3
_PX4_MAIN_MODE_OFFBOARD = 6
_PX4_OFFBOARD_CUSTOM_MODE = _PX4_MAIN_MODE_OFFBOARD << 16
# 仅使用 x,y,z（与起飞成功版本.txt 一致）
_POS_TYPE_MASK = 0x1F8
# 固定翼盘旋后切 OFFBOARD：loiter 类型 setpoint（PX4 扩展 type_mask）
_FW_LOITER_TYPE = 12288
# 四元数姿态 + 油门；仅忽略角速率（勿含 128=ATTITUDE_IGNORE）
_ATTITUDE_TYPE_MASK = 0b00000111
_EARTH_RADIUS_M = 6378137.0

_ACK_NAMES = {
    0: 'ACCEPTED',
    1: 'TEMPORARILY_REJECTED',
    2: 'DENIED',
    3: 'UNSUPPORTED',
    4: 'FAILED',
    5: 'IN_PROGRESS',
    6: 'CANCELLED',
}


class TakeoffLink:
    def __init__(self, device='/dev/ttyAMA0', baud=57600):
        self.master = mavutil.mavlink_connection(device, baud=baud, autoreconnect=True)
        self.relative_alt_m = 0.0
        self.alt_amsl_m = 0.0
        self._lat_deg = 0.0
        self._lon_deg = 0.0
        self._ned_x = 0.0
        self._ned_y = 0.0
        self._ned_z = 0.0
        self._yaw_rad = 0.0
        self._roll_rad = 0.0
        self._pitch_rad = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._vz = 0.0
        self._airspeed = 0.0
        self._groundspeed = 0.0
        self._throttle_pct = 0.0
        self._throttle_from_hil = False
        self._home_lat = None
        self._home_lon = None
        self._local_ned_ready = False
        self._local_ned_mono = 0.0
        self._position_ready = False
        self._armed = False
        self._main_mode = None
        self._last_status = ''
        self._boot_mono = time.monotonic()
        self._offboard_use_global = False
        self._offboard_loiter_type = False
        self._path_lock_snap = None  # OFFBOARD 切换前一刻的状态（锁圆心用）

    def wait_heartbeat(self, timeout=5.0):
        """
        逻辑说明：
        此函数用于等待与飞控（Flight Controller, FC）的 MAVLink 心跳包（heartbeat message），
        以确认通信链路已建立、飞控在线。MAVLink 心跳包是一种周期性消息，由飞控自动向外广播，
        内容包含解锁状态、主模式、系统ID等基本状态信息，是判断飞控在线、模式切换的基础。

        工作流程：
        - 本函数阻塞式地等待来自飞控的心跳包（是飞控“那边”周期性主动发出的消息）；
        - 若在超时时间 timeout（如5秒）内未收到，则打印错误提示、返回 False；
        - 若收到心跳包，会解析当前解锁状态（ARMED/未解锁），记录到 self._armed，并打印连接成功信息；
        """
        print('⏳ 等待飞控心跳...')
        msg = self.master.wait_heartbeat(timeout=timeout)
        # 如果没收到心跳包，返回失败
        if msg is None:
            print('❌ 飞控心跳超时')
            return False
        # 解析MAVLink心跳包中的解锁状态（base_mode 按位与 ARM 标志位）
        self._armed = bool(
            msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        )
        print(f'✅ 飞控连接成功（{"已解锁" if self._armed else "未解锁"}）')
        return True

    def request_data_stream(self, rate=10):
        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            rate,
            1,
        )

    def request_message_interval(self, msg_id, rate_hz):
        """PX4：MAV_CMD_SET_MESSAGE_INTERVAL 请求指定消息频率 [Hz]。"""
        if rate_hz <= 0:
            interval_us = -1
        else:
            interval_us = int(1_000_000 / rate_hz)
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            float(msg_id),
            float(interval_us),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def request_telemetry_streams(self, rate_hz=25):
        """请求路径控制所需的关键 MAVLink 消息（PX4 优先用 interval）。"""
        hz = max(int(rate_hz), 1)
        self.request_data_stream(rate=hz)
        for msg_id in (
            mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED,
            mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
            mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD,
        ):
            self.request_message_interval(msg_id, hz)

    def _from_fc(self, msg):
        return msg.get_srcSystem() == self.master.target_system

    def _handle_message(self, msg):
        t = msg.get_type()
        if t == 'LOCAL_POSITION_NED' and self._from_fc(msg):
            self._ned_x = float(msg.x)
            self._ned_y = float(msg.y)
            self._ned_z = float(msg.z)
            self._vx = float(msg.vx)
            self._vy = float(msg.vy)
            self._vz = float(msg.vz)
            self._local_ned_ready = True
            self._local_ned_mono = time.monotonic()
            self._position_ready = True
        elif t == 'GLOBAL_POSITION_INT':
            self._lat_deg = msg.lat * 1e-7
            self._lon_deg = msg.lon * 1e-7
            self.relative_alt_m = msg.relative_alt * 1e-3
            self.alt_amsl_m = msg.alt * 1e-3
            if not self._local_ned_ready and self._home_lat is not None:
                self._ned_x, self._ned_y, self._ned_z = self._latlon_to_ned(
                    self._lat_deg,
                    self._lon_deg,
                    self.relative_alt_m,
                    self._home_lat,
                    self._home_lon,
                )
                self._vx = msg.vx * 1e-2
                self._vy = msg.vy * 1e-2
                self._vz = msg.vz * 1e-2
                self._position_ready = True
        elif t == 'VFR_HUD' and self._from_fc(msg):
            self._airspeed = msg.airspeed
            self._groundspeed = msg.groundspeed
            if not self._throttle_from_hil:
                self._throttle_pct = float(getattr(msg, 'throttle', 0.0))
        elif t == 'HIL_ACTUATOR_CONTROLS' and self._from_fc(msg):
            self._throttle_from_hil = True
            self._throttle_pct = max(0.0, float(msg.controls[3])) * 100.0
        elif t == 'HOME_POSITION':
            self._home_lat = msg.latitude * 1e-7
            self._home_lon = msg.longitude * 1e-7
        elif t == 'ATTITUDE' and self._from_fc(msg):
            self._roll_rad = msg.roll
            self._pitch_rad = msg.pitch
            self._yaw_rad = msg.yaw
        elif t == 'HEARTBEAT' and self._from_fc(msg):
            self._armed = bool(
                msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            )
            self._main_mode = (msg.custom_mode >> 16) & 0xFF
        elif t == 'STATUSTEXT' and self._from_fc(msg):
            self._last_status = msg.text
            print(f'📢 {msg.text}')
        elif t == 'COMMAND_ACK' and self._from_fc(msg):
            name = _ACK_NAMES.get(msg.result, str(msg.result))
            print(f'📋 ACK cmd={msg.command} result={msg.result} ({name})')

    def drain_messages(self, limit=50):
        for _ in range(limit):
            msg = self.master.recv_match(blocking=False)
            if msg is None:
                break
            self._handle_message(msg)

    def drain_all_messages(self, max_msgs=500):
        """读空当前接收队列，保证用到最新 LOCAL_POSITION_NED。"""
        for _ in range(max_msgs):
            msg = self.master.recv_match(blocking=False)
            if msg is None:
                break
            self._handle_message(msg)

    def _wait_messages(self, duration):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            msg = self.master.recv_match(blocking=True, timeout=0.2)
            if msg is not None:
                self._handle_message(msg)
            else:
                self.drain_messages()

    def _time_boot_ms(self):
        elapsed_ms = int((time.monotonic() - self._boot_mono) * 1000)
        return elapsed_ms & 0xFFFFFFFF

    @staticmethod
    def _is_offboard(custom_mode):
        return ((custom_mode >> 16) & 0xFF) == _PX4_MAIN_MODE_OFFBOARD

    @staticmethod
    def _euler_to_quaternion(roll, pitch, yaw):
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        return [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]

    def wait_position(self, timeout=15.0):
        print('⏳ 等待 LOCAL_POSITION_NED / HOME_POSITION / GPS...')
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            msg = self.master.recv_match(blocking=True, timeout=0.3)
            if msg is not None:
                self._handle_message(msg)
            if self._position_ready:
                src = 'LOCAL_POSITION_NED' if self._local_ned_ready else 'GLOBAL+HOME'
                print(
                    f'✅ 位置就绪 ({src}) '
                    f'NED=({self._ned_x:.1f}, {self._ned_y:.1f}, {self._ned_z:.1f})'
                )
                return True
        print('❌ 超时未收到位置')
        return False

    def control_snapshot(self, default_airspeed=15.0, min_course_speed=0.5):
        """
        路径控制用状态快照（stage5 每拍调用）。

        返回 dict 键与 path_follower.state 对齐：
          x,y,z [m] NED；V [m/s]；psi [rad] 航向；gamma [rad] 航迹角；roll [rad]
        V 优先 airspeed，其次 groundspeed，再次 default_airspeed（配置 flight.airspeed）。
        """
        self.drain_all_messages()
        V = self._airspeed
        if V < 2.0:
            V = self._groundspeed
        if V < 2.0:
            V = default_airspeed
        v_hor = math.hypot(self._vx, self._vy)
        if v_hor >= min_course_speed:
            gamma = math.atan2(-self._vz, v_hor)
            course = math.atan2(self._vy, self._vx)
        else:
            gamma = self._pitch_rad
            course = self._yaw_rad
        return {
            'x': self._ned_x,
            'y': self._ned_y,
            'z': self._ned_z,
            'vx': self._vx,
            'vy': self._vy,
            'vz': self._vz,
            'V': V,
            'psi': self._yaw_rad,
            'course': course,
            'gamma': gamma,
            'roll': self._roll_rad,
            'main_mode': self._main_mode,
            'relative_alt_m': (
                -self._ned_z if self._local_ned_ready else self.relative_alt_m
            ),
            'throttle_pct': self._throttle_pct,
            'pos_age_ms': (
                (time.monotonic() - self._local_ned_mono) * 1000.0
                if self._local_ned_ready else -1.0
            ),
        }

    def send_airspeed_command(self, speed_mps, throttle_pct=-1.0):
        """
        MAV_CMD_DO_CHANGE_SPEED → PX4/TECS 空速目标 [m/s]。
        固定翼 OFFBOARD 姿态模式：这是 companion 侧有效的控速手段。
        """
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            0,
            0.0,
            float(speed_mps),
            float(throttle_pct),
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def send_airspeed_if_changed(
        self,
        speed_mps,
        *,
        last_sent,
        min_interval_sec=0.4,
        min_delta_mps=0.25,
    ):
        """限频发送空速指令，避免 COMMAND_LONG 刷屏。"""
        now = time.monotonic()
        if (
            last_sent['v'] is None
            or now - last_sent['t'] >= min_interval_sec
            or abs(speed_mps - last_sent['v']) >= min_delta_mps
        ):
            self.send_airspeed_command(speed_mps)
            last_sent['t'] = now
            last_sent['v'] = speed_mps
            return True
        return False

    def send_attitude_setpoint(self, roll=0.0, pitch=0.0, yaw=None, thrust=0.55):
        """
        OFFBOARD 姿态 setpoint → MAVLink SET_ATTITUDE_TARGET。

        注意：PX4 固定翼 OFFBOARD 姿态模式下 thrust 多由 TECS 内部决定，
        改 thrust 往往看不到空速变化；控速请用 send_airspeed_command()。
        """
        if yaw is None:
            yaw = self._yaw_rad
        q = self._euler_to_quaternion(roll, pitch, yaw)
        self.master.mav.set_attitude_target_send(
            self._time_boot_ms(),
            self.master.target_system,
            self.master.target_component,
            _ATTITUDE_TYPE_MASK,
            q,
            0.0,
            0.0,
            0.0,
            thrust,
        )

    def run_offboard_attitude_hold(
        self,
        duration_sec=30.0,
        stream_hz=25.0,
        roll=0.0,
        pitch=0.0,
        thrust=0.55,
        print_interval_sec=1.0,
        handoff_sec=2.0,
    ):
        """
        阶段 1：已在 OFFBOARD 时发姿态 setpoint。
        handoff_sec 内同时发位置保持，避免位置→姿态切换瞬间丢信号。
        """
        print('=' * 50)
        print(' 阶段 1：OFFBOARD 姿态保持')
        print('=' * 50)
        print(
            f'时长 {duration_sec:.0f}s @ {stream_hz:.0f}Hz | '
            f'φ={math.degrees(roll):.1f}° θ={math.degrees(pitch):.1f}° thrust={thrust:.2f}'
        )

        interval = 1.0 / stream_hz
        t0 = time.monotonic()
        last_log = 0.0
        drops = 0
        loiter = self._offboard_loiter_type
        if handoff_sec > 0:
            print(f'⏳ 过渡 {handoff_sec:.0f}s：位置+姿态 setpoint 并行（防 No offboard signal）')

        while time.monotonic() - t0 < duration_sec:
            elapsed = time.monotonic() - t0
            if self._main_mode != _PX4_MAIN_MODE_OFFBOARD:
                drops += 1
                if drops == 1:
                    print('⚠️  已离开 OFFBOARD，继续发 setpoint 尝试维持...')

            self.send_onboard_heartbeat()
            if elapsed < handoff_sec:
                self._send_offboard_setpoint(
                    self._ned_x, self._ned_y, self._ned_z, loiter
                )
            self.send_attitude_setpoint(roll=roll, pitch=pitch, thrust=thrust)
            self.drain_messages()

            if elapsed - last_log >= print_interval_sec:
                print(
                    f'[{elapsed:5.1f}s] mode={self._main_mode} | '
                    f'φ={math.degrees(self._roll_rad):5.1f}° '
                    f'θ={math.degrees(self._pitch_rad):5.1f}° | '
                    f'alt={self.relative_alt_m:5.1f}m'
                )
                last_log = elapsed
            time.sleep(interval)

        in_ob = self._main_mode == _PX4_MAIN_MODE_OFFBOARD
        print(f'✅ 姿态保持结束（末模式={self._main_mode}，{"仍在OFFBOARD" if in_ob else "已退出OFFBOARD"}）')
        return in_ob

    def _run_offboard_attitude_segment(
        self,
        duration_sec,
        stream_hz,
        roll,
        pitch,
        thrust,
        print_interval_sec,
        handoff_sec,
        label,
    ):
        """单段 OFFBOARD 姿态 setpoint（内部用）"""
        interval = 1.0 / stream_hz
        loiter = self._offboard_loiter_type
        t0 = time.monotonic()
        last_log = 0.0
        lost = False

        print(
            f'\n▶ {label} | φ={math.degrees(roll):+.1f}° θ={math.degrees(pitch):+.1f}° '
            f'| {duration_sec:.0f}s @ {stream_hz:.0f}Hz'
        )
        if handoff_sec > 0:
            print(f'  过渡 {handoff_sec:.0f}s：位置+姿态并行')

        while time.monotonic() - t0 < duration_sec:
            elapsed = time.monotonic() - t0
            if self._main_mode != _PX4_MAIN_MODE_OFFBOARD:
                if not lost:
                    print('  ⚠️  已离开 OFFBOARD')
                    lost = True

            self.send_onboard_heartbeat()
            if elapsed < handoff_sec:
                self._send_offboard_setpoint(
                    self._ned_x, self._ned_y, self._ned_z, loiter
                )
            self.send_attitude_setpoint(roll=roll, pitch=pitch, thrust=thrust)
            self.drain_messages()

            if elapsed - last_log >= print_interval_sec:
                print(
                    f'  [{elapsed:5.1f}s] mode={self._main_mode} | '
                    f'φ={math.degrees(self._roll_rad):+6.1f}° '
                    f'θ={math.degrees(self._pitch_rad):+6.1f}° | '
                    f'alt={self.relative_alt_m:5.1f}m'
                )
                last_log = elapsed
            time.sleep(interval)

        return self._main_mode == _PX4_MAIN_MODE_OFFBOARD

    def run_open_loop_attitude_test(
        self,
        roll_amp_deg=15.0,
        pitch_amp_deg=5.0,
        step_duration_sec=10.0,
        pause_sec=5.0,
        settle_sec=5.0,
        stream_hz=25.0,
        thrust=0.55,
        handoff_sec=2.0,
        print_interval_sec=1.0,
    ):
        """
        阶段 2：开环姿态验证
        中立 → +roll → 中立 → -roll → 中立 → +pitch → 中立 → -pitch → 中立
        """
        print('=' * 50)
        print(' 阶段 2：开环姿态验证')
        print('=' * 50)
        print(
            f'滚转 ±{roll_amp_deg:.0f}° | 俯仰 ±{pitch_amp_deg:.0f}° | '
            f'每步 {step_duration_sec:.0f}s | thrust={thrust:.2f}'
        )

        roll_a = math.radians(roll_amp_deg)
        pitch_a = math.radians(pitch_amp_deg)
        steps = [
            ('中立稳定', 0.0, 0.0, settle_sec, handoff_sec),
            (f'右滚 +{roll_amp_deg:.0f}°', roll_a, 0.0, step_duration_sec, 0.0),
            ('回正', 0.0, 0.0, pause_sec, 0.0),
            (f'左滚 -{roll_amp_deg:.0f}°', -roll_a, 0.0, step_duration_sec, 0.0),
            ('回正', 0.0, 0.0, pause_sec, 0.0),
            (f'抬头 +{pitch_amp_deg:.0f}°', 0.0, pitch_a, step_duration_sec, 0.0),
            ('回正', 0.0, 0.0, pause_sec, 0.0),
            (f'低头 -{pitch_amp_deg:.0f}°', 0.0, -pitch_a, step_duration_sec, 0.0),
            ('回正', 0.0, 0.0, pause_sec, 0.0),
        ]

        all_ok = True
        for label, roll, pitch, dur, handoff in steps:
            ok = self._run_offboard_attitude_segment(
                duration_sec=dur,
                stream_hz=stream_hz,
                roll=roll,
                pitch=pitch,
                thrust=thrust,
                print_interval_sec=print_interval_sec,
                handoff_sec=handoff,
                label=label,
            )
            if not ok:
                all_ok = False

        print(
            f'\n{"✅ 阶段 2 完成（全程 OFFBOARD）" if all_ok else "⚠️  阶段 2 结束但曾离开 OFFBOARD"}'
        )
        return all_ok

    @staticmethod
    def _latlon_to_ned(lat_deg, lon_deg, relative_alt_m, home_lat, home_lon):
        if home_lat is None:
            return 0.0, 0.0, -relative_alt_m
        home_lat_rad = math.radians(home_lat)
        d_lat = math.radians(lat_deg - home_lat)
        d_lon = math.radians(lon_deg - home_lon)
        north = d_lat * _EARTH_RADIUS_M
        east = d_lon * _EARTH_RADIUS_M * math.cos(home_lat_rad)
        return north, east, -relative_alt_m

    def send_onboard_heartbeat(self):
        """告知飞控存在外部控制器（与起飞成功版本一致）"""
        self.master.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0,
            0,
            0,
        )

    @staticmethod
    def _ned_to_latlon(x, y, z, home_lat, home_lon):
        if home_lat is None:
            return 0.0, 0.0, -z
        home_lat_rad = math.radians(home_lat)
        lat_deg = home_lat + math.degrees(x / _EARTH_RADIUS_M)
        lon_deg = home_lon + math.degrees(
            y / (_EARTH_RADIUS_M * math.cos(home_lat_rad))
        )
        return lat_deg, lon_deg, -z

    def send_position_local(self, x, y, z, loiter_type=False):
        type_mask = _POS_TYPE_MASK | (_FW_LOITER_TYPE if loiter_type else 0)
        self.master.mav.set_position_target_local_ned_send(
            self._time_boot_ms(),
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            x,
            y,
            z,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def send_position_global(self, lat_deg, lon_deg, alt_m, loiter_type=False):
        type_mask = _POS_TYPE_MASK | (_FW_LOITER_TYPE if loiter_type else 0)
        self.master.mav.set_position_target_global_int_send(
            self._time_boot_ms(),
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            type_mask,
            int(lat_deg * 1e7),
            int(lon_deg * 1e7),
            alt_m,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def send_position_hold_local(self, loiter_type=False):
        """固定翼 OFFBOARD：LOCAL_NED 位置保持（当前位置）"""
        self.send_position_local(
            self._ned_x, self._ned_y, self._ned_z, loiter_type=loiter_type
        )

    def send_position_hold_global(self, loiter_type=False):
        """固定翼 OFFBOARD：GLOBAL 位置保持（当前经纬度）"""
        if not self._position_ready:
            return False
        self.send_position_global(
            self._lat_deg,
            self._lon_deg,
            self.relative_alt_m,
            loiter_type=loiter_type,
        )
        return True

    def _send_offboard_setpoint(
        self, x, y, z, loiter_type=False, use_global=None,
    ):
        self.send_onboard_heartbeat()
        if use_global is None:
            use_global = self._offboard_use_global
        if use_global:
            lat, lon, alt = self._ned_to_latlon(
                x, y, z, self._home_lat, self._home_lon
            )
            self.send_position_global(lat, lon, alt, loiter_type=loiter_type)
        else:
            self.send_position_local(x, y, z, loiter_type=loiter_type)

    def _stream_offboard_setpoints(self, use_global=False, loiter_type=False):
        self.send_onboard_heartbeat()
        if use_global:
            self.send_position_hold_global(loiter_type=loiter_type)
        else:
            self.send_position_hold_local(loiter_type=loiter_type)

    def _stream_offboard_cruise_setpoint(
        self,
        origin_x,
        origin_y,
        origin_z,
        course,
        s_travel,
        cruise_speed_mps,
        lookahead_m,
        dt,
        use_global=False,
        loiter_type=False,
    ):
        """平稳前推 setpoint（沿航向前瞻，勿钉原点）。"""
        s_travel += cruise_speed_mps * dt
        base_x = origin_x + s_travel * math.cos(course)
        base_y = origin_y + s_travel * math.sin(course)
        tx = base_x + lookahead_m * math.cos(course)
        ty = base_y + lookahead_m * math.sin(course)
        self._send_offboard_setpoint(
            tx, ty, origin_z, loiter_type=loiter_type, use_global=use_global,
        )
        return s_travel

    def _send_offboard_mode_cmd(self, legacy=False):
        if legacy:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,
                1.0,
                6.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            print('📤 OFFBOARD 模式指令（legacy param1=1, param2=6）')
            return
        mapping = self.master.mode_mapping()
        if mapping and 'OFFBOARD' in mapping:
            self.master.set_mode('OFFBOARD')
            print('📤 OFFBOARD 模式指令（set_mode OFFBOARD）')
            return
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            float(mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
            float(_PX4_OFFBOARD_CUSTOM_MODE),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            _PX4_OFFBOARD_CUSTOM_MODE,
        )
        print('📤 OFFBOARD 模式指令（custom_mode=6<<16）')

    def try_set_offboard_mode(
        self,
        timeout=15.0,
        warmup_sec=3.0,
        stream_hz=10.0,
        mode_cmd_hz=5.0,
        hold_after_sec=5.0,
        warmup_style='hold',
        cruise_speed_mps=12.0,
        lookahead_m=40.0,
    ):
        """
        切换 OFFBOARD：先预热 setpoint（PX4 硬性要求 >1s @ >2Hz），再发模式指令。
        warmup_style: hold=当前位置保持 | cruise=沿航向平稳前推（固定翼推荐）。
        QGC 里 commander mode offboard 不会发 setpoint，单独用必然无效。
        """
        print('=' * 50)
        print(' OFFBOARD 切换测试')
        print('=' * 50)
        print('ℹ️  PX4 要求：先连续收到 setpoint（>2Hz，>1s）才能进 OFFBOARD')
        print('ℹ️  QGC 控制台 commander mode offboard 不会发 setpoint，需脚本/伴机发')

        if not self._position_ready:
            print('⏳ 等待 GPS 位置...')
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                msg = self.master.recv_match(blocking=True, timeout=0.3)
                if msg is not None:
                    self._handle_message(msg)
                if self._position_ready:
                    break
            if not self._position_ready:
                print('❌ 无 GPS 位置，固定翼 OFFBOARD 无法切换')
                return False

        print(
            f'📍 当前位置 lat={self._lat_deg:.6f} lon={self._lon_deg:.6f} '
            f'alt={self.relative_alt_m:.1f}m'
        )

        sp_interval = 1.0 / stream_hz
        mode_interval = 1.0 / mode_cmd_hz
        use_cruise_warmup = str(warmup_style).lower() == 'cruise'
        if use_cruise_warmup:
            snap = self.control_snapshot()
            course = snap['course'] if snap['V'] > 3.0 else snap['psi']
            print(
                f'ℹ️  预热方式：平稳前推 {cruise_speed_mps:.0f} m/s | '
                f'前瞻 {lookahead_m:.0f} m | 航向 {math.degrees(course):.0f}°'
            )
        else:
            print('ℹ️  预热方式：当前位置保持（钉原点）')
        strategies = (
            ('LOCAL_NED', False, False),
            ('LOCAL_NED+LOITER', False, True),
            ('GLOBAL_REL_ALT', True, False),
            ('GLOBAL+LOITER', True, True),
        )

        for label, use_global, loiter_type in strategies:
            print(f'\n--- 策略: {label} ---')
            print(f'⏳ 预热 setpoint {warmup_sec:.0f}s（仅发目标点，不切模式）...')
            origin_x = self._ned_x
            origin_y = self._ned_y
            origin_z = self._ned_z
            s_travel = 0.0
            warmup_deadline = time.monotonic() + warmup_sec
            while time.monotonic() < warmup_deadline:
                if use_cruise_warmup:
                    s_travel = self._stream_offboard_cruise_setpoint(
                        origin_x,
                        origin_y,
                        origin_z,
                        course,
                        s_travel,
                        cruise_speed_mps,
                        lookahead_m,
                        sp_interval,
                        use_global=use_global,
                        loiter_type=loiter_type,
                    )
                else:
                    self._stream_offboard_setpoints(
                        use_global=use_global, loiter_type=loiter_type,
                    )
                self.drain_messages()
                time.sleep(sp_interval)

            self.drain_messages()
            self._path_lock_snap = self.control_snapshot()
            print('📌 已采样切入瞬间 GPS/航向（OFFBOARD 模式指令发出前）')

            print('🚀 预热完成，开始发 OFFBOARD 模式指令...')
            deadline = time.monotonic() + timeout
            last_sp = 0.0
            last_mode = 0.0
            legacy_sent = False
            ack_ok = False

            while time.monotonic() < deadline:
                now = time.monotonic()
                if now - last_sp >= sp_interval:
                    if use_cruise_warmup:
                        s_travel = self._stream_offboard_cruise_setpoint(
                            origin_x,
                            origin_y,
                            origin_z,
                            course,
                            s_travel,
                            cruise_speed_mps,
                            lookahead_m,
                            sp_interval,
                            use_global=use_global,
                            loiter_type=loiter_type,
                        )
                    else:
                        self._stream_offboard_setpoints(
                            use_global=use_global, loiter_type=loiter_type,
                        )
                    last_sp = now
                if now - last_mode >= mode_interval:
                    self._send_offboard_mode_cmd(legacy=False)
                    last_mode = now
                if not legacy_sent and (deadline - now) < timeout * 0.4:
                    self._send_offboard_mode_cmd(legacy=True)
                    legacy_sent = True

                msg = self.master.recv_match(blocking=True, timeout=0.02)
                if msg is not None:
                    self._handle_message(msg)
                    if msg.get_type() == 'HEARTBEAT' and self._from_fc(msg):
                        if self._is_offboard(msg.custom_mode):
                            print(f'✅ OFFBOARD 成功（策略 {label}）')
                            self._offboard_use_global = use_global
                            self._offboard_loiter_type = loiter_type
                            return True
                    elif msg.get_type() == 'COMMAND_ACK' and self._from_fc(msg):
                        if (
                            msg.command == mavutil.mavlink.MAV_CMD_DO_SET_MODE
                            and msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED
                        ):
                            ack_ok = True

            if ack_ok:
                print(f'⚠️  策略 {label}：ACK 成功但 HEARTBEAT 主模式仍为 {self._main_mode}')

        print(f'❌ 全部策略失败（当前主模式={self._main_mode}，期望={_PX4_MAIN_MODE_OFFBOARD}）')
        if self._last_status:
            print(f'   飞控: {self._last_status}')
        print('💡 请在 QGC 参数检查：COM_OBL_RC_ACT、COM_OF_LOSS_T、COM_RC_IN_MODE')
        return False

    def run_s_curve_offboard(
        self,
        length_m=250.0,
        width_m=50.0,
        cruise_speed_mps=12.0,
        lookahead_m=40.0,
        stream_hz=10.0,
        settle_sec=2.0,
        hold_end_sec=3.0,
    ):
        """板外模式下沿 S 形路径飞行（先左弯再右弯）"""
        from offboard_s_curve import s_curve_ned_point, s_curve_progress

        if not self._position_ready:
            print('❌ 无位置信息，无法执行 S 形路径')
            return False
        if self._main_mode != _PX4_MAIN_MODE_OFFBOARD:
            print(f'⚠️  当前主模式={self._main_mode}，未在 OFFBOARD，仍将发送 setpoint')

        origin_x = self._ned_x
        origin_y = self._ned_y
        origin_z = self._ned_z
        yaw = self._yaw_rad
        loiter = self._offboard_loiter_type
        interval = 1.0 / stream_hz

        print('=' * 50)
        print(' OFFBOARD S 形路径')
        print('=' * 50)
        print(
            f'📐 总长 {length_m:.0f}m | 横向幅值 {width_m:.0f}m | '
            f'地速 {cruise_speed_mps:.0f}m/s | 前瞻 {lookahead_m:.0f}m'
        )
        print(f'🧭 起始航向 {math.degrees(yaw):.0f}° | 坐标 {"GLOBAL" if self._offboard_use_global else "LOCAL_NED"}')

        if settle_sec > 0:
            print(f'⏳ 当前点稳定 {settle_sec:.0f}s...')
            deadline = time.monotonic() + settle_sec
            while time.monotonic() < deadline:
                self._send_offboard_setpoint(origin_x, origin_y, origin_z, loiter)
                self.drain_messages()
                time.sleep(interval)

        s_travel = 0.0
        t0 = time.monotonic()
        last_log = 0.0
        print('🛫 开始 S 形：左弯 → 右弯 → 回正')

        while s_travel < length_m:
            if self._main_mode != _PX4_MAIN_MODE_OFFBOARD:
                print('❌ 已退出 OFFBOARD，中止 S 形')
                return False

            dt = interval
            s_travel = min(length_m, s_travel + cruise_speed_mps * dt)
            s_target = min(length_m, s_travel + lookahead_m)
            tx, ty, tz = s_curve_ned_point(
                origin_x, origin_y, origin_z, yaw, s_target, length_m, width_m
            )
            self._send_offboard_setpoint(tx, ty, tz, loiter)
            self.drain_messages()

            elapsed = time.monotonic() - t0
            if elapsed - last_log >= 2.0:
                prog, lateral = s_curve_progress(s_travel, length_m, width_m)
                side = '左' if lateral < -1.0 else ('右' if lateral > 1.0 else '中')
                print(
                    f'  进度 {prog * 100:.0f}% | s={s_travel:.0f}m | '
                    f'横向 {lateral:+.0f}m ({side}) | {elapsed:.0f}s'
                )
                last_log = elapsed
            time.sleep(interval)

        end_x, end_y, end_z = s_curve_ned_point(
            origin_x, origin_y, origin_z, yaw, length_m, length_m, width_m
        )
        print(f'✅ S 形完成，末端保持 {hold_end_sec:.0f}s...')
        hold_deadline = time.monotonic() + hold_end_sec
        while time.monotonic() < hold_deadline:
            self._send_offboard_setpoint(end_x, end_y, end_z, loiter)
            self.drain_messages()
            time.sleep(interval)

        print('✅ S 形板外任务结束')
        return True

    def run_forward_offboard_cruise(
        self,
        duration_sec,
        cruise_speed_mps=12.0,
        lookahead_m=40.0,
        stream_hz=10.0,
        command_airspeed_mps=None,
        airspeed_cmd_interval_sec=5.0,
    ):
        """OFFBOARD 下沿当前航向前推位置 setpoint（固定翼平稳前飞，勿钉原点）。"""
        if not self._position_ready:
            print('❌ 无位置信息，无法前推 setpoint')
            return False
        if self._main_mode != _PX4_MAIN_MODE_OFFBOARD:
            print(f'⚠️  当前主模式={self._main_mode}，未在 OFFBOARD')

        loiter = self._offboard_loiter_type
        interval = 1.0 / stream_hz
        origin_x = self._ned_x
        origin_y = self._ned_y
        origin_z = self._ned_z
        snap = self.control_snapshot()
        course = snap['course'] if snap['V'] > 3.0 else snap['psi']
        print(
            f'⏳ 平稳前飞 {duration_sec:.0f}s | '
            f'{cruise_speed_mps:.0f} m/s | 前瞻 {lookahead_m:.0f} m | '
            f'航向 {math.degrees(course):.0f}°'
        )
        s_travel = 0.0
        t0 = time.monotonic()
        last_log = 0.0
        last_asp_cmd = -airspeed_cmd_interval_sec
        if command_airspeed_mps is not None:
            self.send_airspeed_command(command_airspeed_mps)
            last_asp_cmd = 0.0
            print(f'📤 空速指令 → {command_airspeed_mps:.0f} m/s')
        while time.monotonic() - t0 < duration_sec:
            if self._main_mode != _PX4_MAIN_MODE_OFFBOARD:
                print('❌ 已退出 OFFBOARD，中止前飞')
                return False
            s_travel += cruise_speed_mps * interval
            base_x = origin_x + s_travel * math.cos(course)
            base_y = origin_y + s_travel * math.sin(course)
            tx = base_x + lookahead_m * math.cos(course)
            ty = base_y + lookahead_m * math.sin(course)
            # 高度锁定：NED z 固定为切入时刻，避免只推 xy 导致爬升/俯冲振荡
            self._send_offboard_setpoint(tx, ty, origin_z, loiter)
            self.drain_messages()
            elapsed = time.monotonic() - t0
            if (
                command_airspeed_mps is not None
                and elapsed - last_asp_cmd >= airspeed_cmd_interval_sec
            ):
                self.send_airspeed_command(command_airspeed_mps)
                last_asp_cmd = elapsed
            if elapsed - last_log >= 1.0:
                snap = self.control_snapshot()
                print(
                    f'  [{elapsed:4.0f}s] s={s_travel:.0f} m '
                    f'alt={self.relative_alt_m:.1f} m V≈{snap["V"]:.1f} m/s'
                )
                last_log = elapsed
            time.sleep(interval)
        return True

    @staticmethod
    def _angle_diff(a, b):
        return (a - b + math.pi) % (2.0 * math.pi) - math.pi

    def run_straight_attitude_cruise(
        self,
        duration_sec,
        target_airspeed_mps=15.0,
        stream_hz=25.0,
        thrust=0.38,
        alpha_trim=0.05,
        k_airspeed_pitch=0.012,
        k_course_roll=0.35,
        max_roll_rad=0.21,
        z_hold_ned=None,
        k_alt_pitch=0.003,
        pitch_max_up=0.12,
        pitch_max_dn=0.14,
        command_airspeed_mps=None,
        airspeed_cmd_interval_sec=5.0,
        course_rad=None,
    ):
        """
        OFFBOARD 姿态直飞：φ≈0 + 航向保持 + 空速/高度闭环。
        比纯位置前推更适合固定翼匀速直线（侧风时用小幅滚转修航向）。
        """
        if self._main_mode != _PX4_MAIN_MODE_OFFBOARD:
            print(f'⚠️  当前主模式={self._main_mode}，未在 OFFBOARD')

        snap0 = self.control_snapshot(default_airspeed=target_airspeed_mps)
        course = course_rad if course_rad is not None else (
            snap0['course'] if snap0['V'] > 3.0 else snap0['psi']
        )
        if z_hold_ned is None:
            z_hold_ned = self._ned_z

        interval = 1.0 / stream_hz
        print(
            f'⏳ 姿态直飞 {duration_sec:.0f}s | 目标空速 {target_airspeed_mps:.0f} m/s | '
            f'航向 {math.degrees(course):.0f}° | thrust={thrust:.2f}'
        )

        t0 = time.monotonic()
        last_log = 0.0
        last_asp_cmd = -airspeed_cmd_interval_sec
        if command_airspeed_mps is not None:
            self.send_airspeed_command(command_airspeed_mps)
            last_asp_cmd = 0.0
            print(f'📤 空速指令 → {command_airspeed_mps:.0f} m/s')

        while time.monotonic() - t0 < duration_sec:
            if self._main_mode != _PX4_MAIN_MODE_OFFBOARD:
                print('❌ 已退出 OFFBOARD，中止直飞')
                return False

            snap = self.control_snapshot(default_airspeed=target_airspeed_mps)
            V = snap['V']
            hdg = snap['course'] if V > 3.0 else snap['psi']

            roll = k_course_roll * self._angle_diff(course, hdg)
            roll = max(-max_roll_rad, min(max_roll_rad, roll))

            pitch = alpha_trim + k_airspeed_pitch * (target_airspeed_mps - V)
            ez = self._ned_z - z_hold_ned
            pitch += k_alt_pitch * ez
            if -ez > 5.0:
                pitch = min(pitch, 0.0)
            pitch = max(-pitch_max_dn, min(pitch_max_up, pitch))

            thr = thrust
            if V > target_airspeed_mps + 2.0:
                thr = min(thr, thrust * (target_airspeed_mps / V) ** 2)

            self.send_onboard_heartbeat()
            self.send_attitude_setpoint(roll=roll, pitch=pitch, yaw=None, thrust=thr)
            self.drain_messages()

            elapsed = time.monotonic() - t0
            if (
                command_airspeed_mps is not None
                and elapsed - last_asp_cmd >= airspeed_cmd_interval_sec
            ):
                self.send_airspeed_command(command_airspeed_mps)
                last_asp_cmd = elapsed
            if elapsed - last_log >= 1.0:
                print(
                    f'  [{elapsed:4.0f}s] alt={self.relative_alt_m:.1f} m '
                    f'V≈{V:.1f} φ={math.degrees(roll):+.1f}° θ={math.degrees(pitch):+.1f}°'
                )
                last_log = elapsed
            time.sleep(interval)
        return True

    def run_takeoff_climb_offboard_cruise(self, tc, oc):
        """
        阶段 5 标准起飞链：
        Hold → 解锁 → Takeoff → 爬升到 altitude_m → OFFBOARD → 平稳前飞 hold_after_sec。
        """
        forward_sec = float(oc.get('hold_after_sec', 5.0))
        stream_hz = float(oc.get('stream_hz', 10.0))
        cruise_speed_mps = float(oc.get('cruise_speed_mps', 12.0))
        lookahead_m = float(oc.get('lookahead_m', 40.0))
        print(
            '🚀 起飞链：Hold→解锁→Takeoff→'
            f'爬升 {tc.get("altitude_m", 30):.0f} m→OFFBOARD→前飞 {forward_sec:.0f} s'
        )
        climb = self.run_commander_takeoff(
            cmd_interval=tc.get('cmd_interval_sec', 2.0),
            hold_settle_sec=tc.get('hold_settle_sec', 1.0),
            try_offboard=False,
            monitor_climb=tc.get('monitor_climb', True),
            altitude_m=tc.get('altitude_m', 30.0),
            altitude_ratio=tc.get('altitude_ratio', 0.9),
            timeout_sec=tc.get('timeout_sec', 180.0),
        )
        if climb is None:
            return False
        print(f'✅ 爬升完成 {climb:.1f} m，切入 OFFBOARD')
        if not self.try_set_offboard_mode(
            warmup_sec=oc.get('warmup_sec', 3.0),
            timeout=oc.get('confirm_timeout', 15.0),
            hold_after_sec=0.0,
            stream_hz=stream_hz,
            warmup_style=oc.get('warmup_style', 'hold'),
            cruise_speed_mps=cruise_speed_mps,
            lookahead_m=lookahead_m,
        ):
            return False
        if forward_sec > 0:
            if not self.run_forward_offboard_cruise(
                forward_sec,
                cruise_speed_mps=cruise_speed_mps,
                lookahead_m=lookahead_m,
                stream_hz=stream_hz,
            ):
                return False
        return True

    def set_mode_hold(self):
        """先切 Hold（QGC「保持」≈ LOITER / HOLD）"""
        mapping = self.master.mode_mapping()
        for name in ('HOLD', 'LOITER', 'AUTO.LOITER', 'POSCTL'):
            if mapping and name in mapping:
                self.master.set_mode(name)
                print(f'📤 模式 → {name}（Hold/保持）')
                return True
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            float(mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
            float(_PX4_MAIN_AUTO),
            float(_PX4_SUB_LOITER),
            0.0,
            0.0,
            0.0,
            0.0,
        )
        print('📤 模式 → AUTO.LOITER（Hold fallback）')
        return True

    def exit_offboard_to_hold(self, confirm_timeout=8.0, retry_interval=1.0):
        """板外任务结束后停止 setpoint 并切回 Hold（LOITER）"""
        print('🔄 板外结束，切回 Hold 模式...')
        deadline = time.monotonic() + confirm_timeout
        attempts = 0
        while time.monotonic() < deadline:
            self.set_mode_hold()
            attempts += 1
            wait_until = time.monotonic() + retry_interval
            while time.monotonic() < wait_until:
                msg = self.master.recv_match(blocking=True, timeout=0.2)
                if msg is not None:
                    self._handle_message(msg)
                if self._main_mode != _PX4_MAIN_MODE_OFFBOARD:
                    print('✅ 已回到 Hold（QGC 应显示保持/Loiter）')
                    return True
        print(f'⚠️  Hold 切换未确认（当前主模式={self._main_mode}，已发 {attempts} 次）')
        return False

    def commander_arm_force(self):
        """等价 commander arm -f"""
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1.0,
            _PX4_FORCE_ARM,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        print('📤 commander arm -f  → MAVLink 强制解锁 (param2=21196)')

    def commander_takeoff_mode(self):
        """等价 commander takeoff：进入 TAKEOFF 模式"""
        mapping = self.master.mode_mapping()
        if mapping and 'TAKEOFF' in mapping:
            self.master.set_mode('TAKEOFF')
            print('📤 commander takeoff → 模式 TAKEOFF')
            return
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            float(mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
            float(_PX4_MAIN_AUTO),
            float(_PX4_SUB_TAKEOFF),
            0.0,
            0.0,
            0.0,
            0.0,
        )
        print('📤 commander takeoff → AUTO.TAKEOFF (fallback)')

    def wait_armed(self, timeout=10.0):
        if self._armed:
            return True
        deadline = time.monotonic() + timeout
        retries = 0
        while time.monotonic() < deadline:
            if self._armed:
                print('✅ QGC 应显示 Armed')
                return True
            if retries < 8:
                self.commander_arm_force()
                retries += 1
            msg = self.master.recv_match(blocking=True, timeout=0.4)
            if msg is not None:
                self._handle_message(msg)
        print('❌ 解锁失败')
        if self._last_status:
            print(f'   飞控: {self._last_status}')
        return False

    def run_commander_takeoff(
        self,
        cmd_interval=2.0,
        hold_settle_sec=1.0,
        try_offboard=False,
        offboard_delay_sec=5.0,
        offboard_warmup_sec=3.0,
        offboard_timeout=15.0,
        offboard_hold_sec=5.0,
        run_s_curve=False,
        s_curve_cfg=None,
        return_to_hold_after=True,
        monitor_climb=True,
        altitude_m=30.0,
        altitude_ratio=0.9,
        timeout_sec=180.0,
        poll=0.5,
    ):
        print('=' * 50)
        print(' PX4 一键起飞（commander 等价 MAVLink）')
        print('=' * 50)
        print('ℹ️  Telem2 无法像 QGC USB 控制台那样走 Shell，已改用 MAVLink')

        self.drain_messages()
        start_rel = self.relative_alt_m if self._position_ready else 0.0

        self.set_mode_hold()
        print(f'⏳ Hold 模式稳定 {hold_settle_sec:.0f}s...')
        self._wait_messages(hold_settle_sec)

        self.commander_arm_force()
        if not self.wait_armed(timeout=10.0):
            return None

        print(f'⏳ 间隔 {cmd_interval:.0f}s（对应两条 commander 指令间隔）')
        self._wait_messages(cmd_interval)

        self.commander_takeoff_mode()
        self._wait_messages(1.0)

        if try_offboard:
            print(f'⏳ takeoff 后等待 {offboard_delay_sec:.0f}s，再切 OFFBOARD...')
            self._wait_messages(offboard_delay_sec)
            if not self.try_set_offboard_mode(
                warmup_sec=offboard_warmup_sec,
                timeout=offboard_timeout,
                hold_after_sec=0.0,
                stream_hz=10.0,
            ):
                return False
            result = True
            if run_s_curve:
                sc = s_curve_cfg or {}
                result = self.run_s_curve_offboard(
                    length_m=sc.get('length_m', 250.0),
                    width_m=sc.get('width_m', 50.0),
                    cruise_speed_mps=sc.get('cruise_speed_mps', 12.0),
                    lookahead_m=sc.get('lookahead_m', 40.0),
                    stream_hz=sc.get('stream_hz', 10.0),
                    settle_sec=sc.get('settle_sec', 2.0),
                    hold_end_sec=sc.get('hold_end_sec', 3.0),
                )
            elif offboard_hold_sec > 0:
                result = self.run_forward_offboard_cruise(
                    offboard_hold_sec,
                    cruise_speed_mps=12.0,
                    lookahead_m=40.0,
                    stream_hz=10.0,
                )
            if result and return_to_hold_after:
                self.exit_offboard_to_hold()
            return result

        if not monitor_climb:
            return True

        goal = start_rel + altitude_m
        threshold = start_rel + altitude_m * altitude_ratio
        t0 = time.monotonic()
        current = start_rel
        print(f'📈 监控爬升 → {goal:.1f} m')

        while current < threshold:
            if time.monotonic() - t0 > timeout_sec:
                print(f'❌ 超时 高度 {current:.1f} m')
                return None
            time.sleep(poll)
            self.drain_messages()
            if self._position_ready:
                current = self.relative_alt_m
            print(f'  {current:.1f} / {goal:.1f} m ({time.monotonic() - t0:.0f}s)')

        print(f'✅ 完成 {current:.1f} m')
        return current
