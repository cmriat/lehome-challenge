#!/usr/bin/env python3
"""
Real-time joint monitor for SO101 Leader and Follower (simulation).

Reads all data from shared state file written by the simulation process.
No serial port access needed — can run safely alongside the simulation.

Usage:
    # Monitor while simulation is running
    python scripts/monitor_joints.py

    # Also directly read leader serial ports (only when simulation is NOT running)
    python scripts/monitor_joints.py --serial --left_port /dev/ttyACM0 --right_port /dev/ttyACM1
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

FOLLOWER_STATE_FILE = "/tmp/lehome_follower_state.json"

# From lehome.assets.robots.lerobot
MOTOR_LIMITS = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}

USD_JOINT_LIMITS = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def norm_to_degree(name: str, norm_val: float) -> float:
    """Convert normalized motor value to degrees."""
    m_min, m_max = MOTOR_LIMITS[name]
    j_min, j_max = USD_JOINT_LIMITS[name]
    return (norm_val - m_min) / (m_max - m_min) * (j_max - j_min) + j_min


def rad_to_degree(rad: float) -> float:
    return rad / math.pi * 180.0


# --- Serial mode (direct port access, only when simulation is NOT running) ---

_left_device = None
_right_device = None


def init_device(port: str):
    sys.path.insert(0, str(Path(__file__).parent.parent / "source" / "lehome"))
    from lehome.devices.lerobot.common.motors import (
        FeetechMotorsBus, Motor, MotorNormMode, MotorCalibration,
    )

    if "ACM0" in port or port.endswith("0"):
        calib_file = "left_so101_leader.json"
    else:
        calib_file = "right_so101_leader.json"

    calibration_path = Path(__file__).parent.parent / "source" / "lehome" / "lehome" / "devices" / "lerobot" / ".cache" / calib_file
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    with open(calibration_path, "r") as f:
        json_data = json.load(f)
    calibration = {}
    for motor_name, motor_data in json_data.items():
        calibration[motor_name] = MotorCalibration(
            id=int(motor_data["id"]),
            drive_mode=int(motor_data["drive_mode"]),
            homing_offset=int(motor_data["homing_offset"]),
            range_min=int(motor_data["range_min"]),
            range_max=int(motor_data["range_max"]),
        )

    bus = FeetechMotorsBus(
        port=port,
        motors={
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        },
        calibration=calibration,
    )
    bus.connect()
    return bus


def read_leader_serial(port: str) -> tuple[dict, str]:
    global _left_device, _right_device
    try:
        if "ACM0" in port:
            if _left_device is None:
                _left_device = init_device(port)
            bus = _left_device
        else:
            if _right_device is None:
                _right_device = init_device(port)
            bus = _right_device
        positions = bus.sync_read("Present_Position", normalize=True)
        return positions, ""
    except Exception as e:
        return {}, str(e)


# --- Shared file mode (read from simulation output) ---

def read_shared_state() -> dict:
    try:
        if os.path.exists(FOLLOWER_STATE_FILE):
            with open(FOLLOWER_STATE_FILE, "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        return {"error": str(e)}


# --- Display ---

def format_values_degree(state: dict, convert_from: str = "norm") -> str:
    """Format joint values as degrees.

    Args:
        convert_from: "norm" (leader normalized), "rad" (follower radians).
    """
    if not state:
        return "No data"
    if "error" in state:
        return f"ERROR: {state['error']}"

    parts = []
    for name in JOINT_NAMES:
        if name in state:
            val = state[name]
            if convert_from == "norm":
                deg = norm_to_degree(name, val)
            elif convert_from == "rad":
                deg = rad_to_degree(val)
            else:
                deg = val
            short = name[:4] if name != "gripper" else "grip"
            parts.append(f"{short}:{deg:>7.1f}°")
    return " ".join(parts)


def print_section(title: str, rows: list[tuple[str, str]], width: int = 78):
    print("┌" + "─" * width + "┐")
    print(f"│ {title:<{width - 2}} │")
    print("├" + "─" * width + "┤")
    for label, val in rows:
        line = f"{label}: {val}"
        print(f"│ {line:<{width - 2}} │")
    print("└" + "─" * width + "┘")


def cleanup():
    global _left_device, _right_device
    for device in [_left_device, _right_device]:
        if device is not None:
            try:
                device.disconnect()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Monitor SO101 joint positions (all values in degrees)")
    parser.add_argument("--serial", action="store_true",
                        help="Read leader directly from serial ports (only when simulation is NOT running)")
    parser.add_argument("--left_port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--right_port", type=str, default="/dev/ttyACM1")
    parser.add_argument("--rate", type=float, default=0.1, help="Update rate in seconds")
    args = parser.parse_args()

    mode = "serial" if args.serial else "file"
    print("\n" + "=" * 80)
    print("SO101 Joint Monitor (all values in degrees) - Press Ctrl+C to exit")
    print(f"Mode: {'Serial (direct port access)' if mode == 'serial' else 'File (read from simulation shared state)'}")
    print("=" * 80)

    try:
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')
            print(f"\n{'='*80}")
            print(f"SO101 Joint Monitor - {time.strftime('%H:%M:%S')}  [mode: {mode}]")
            print(f"{'='*80}\n")

            if mode == "serial":
                # Direct serial read (no simulation running)
                rows = []
                left_state, left_err = read_leader_serial(args.left_port)
                if left_err:
                    rows.append(("LEFT  ARM", f"ERROR: {left_err}"))
                else:
                    rows.append(("LEFT  ARM", format_values_degree(left_state, "norm")))

                right_state, right_err = read_leader_serial(args.right_port)
                if right_err:
                    rows.append(("RIGHT ARM", f"ERROR: {right_err}"))
                else:
                    rows.append(("RIGHT ARM", format_values_degree(right_state, "norm")))

                print_section("LEADER (serial, normalized → degrees)", rows)

            else:
                # File mode: read from simulation shared state
                shared = read_shared_state()
                if not shared:
                    print("  Waiting for simulation to write state file...")
                    print(f"  (expected: {FOLLOWER_STATE_FILE})")
                    time.sleep(args.rate)
                    continue

                ts = shared.get("timestamp", 0)
                age = time.time() - ts
                stale = " [STALE!]" if age > 2.0 else ""
                print(f"  Data age: {age:.1f}s{stale}\n")

                # --- Leader ---
                leader_rows = []
                is_bi = "leader_left_arm" in shared or "leader_right_arm" in shared
                is_single = "leader_arm" in shared

                if is_bi:
                    left_l = shared.get("leader_left_arm", {})
                    right_l = shared.get("leader_right_arm", {})
                    leader_rows.append(("LEFT  ARM", format_values_degree(left_l, "norm")))
                    leader_rows.append(("RIGHT ARM", format_values_degree(right_l, "norm")))
                elif is_single:
                    arm_l = shared.get("leader_arm", {})
                    leader_rows.append(("ARM      ", format_values_degree(arm_l, "norm")))
                else:
                    leader_rows.append(("STATUS   ", "No leader data (press B to start teleoperation)"))

                print_section("LEADER (normalized → degrees)", leader_rows)

                # --- Follower ---
                print()
                follower_rows = []
                if "left_arm" in shared:
                    left_f = shared.get("left_arm", {})
                    right_f = shared.get("right_arm", {})
                    follower_rows.append(("LEFT  ARM", format_values_degree(left_f, "rad")))
                    follower_rows.append(("RIGHT ARM", format_values_degree(right_f, "rad")))
                elif "arm" in shared:
                    arm_f = shared.get("arm", {})
                    follower_rows.append(("ARM      ", format_values_degree(arm_f, "rad")))
                else:
                    follower_rows.append(("STATUS   ", "No follower data"))

                print_section("FOLLOWER (radians → degrees)", follower_rows)

                # --- Diff ---
                if is_bi and "left_arm" in shared:
                    print()
                    diff_rows = []
                    for side, lk, fk in [("LEFT ", "leader_left_arm", "left_arm"),
                                          ("RIGHT", "leader_right_arm", "right_arm")]:
                        ld = shared.get(lk, {})
                        fd = shared.get(fk, {})
                        parts = []
                        for name in JOINT_NAMES:
                            if name in ld and name in fd:
                                l_deg = norm_to_degree(name, ld[name])
                                f_deg = rad_to_degree(fd[name])
                                diff = l_deg - f_deg
                                short = name[:4] if name != "gripper" else "grip"
                                parts.append(f"{short}:{diff:>+7.1f}°")
                        diff_rows.append((f"{side} ARM", " ".join(parts) if parts else "N/A"))
                    print_section("DIFF (leader - follower)", diff_rows)

            print("\n" + "─" * 80)
            print("TIPS:")
            print("  - All values in degrees for direct comparison")
            print("  - Rest pose: shld_lift≈-100°, elbow≈90°, wrist_flex≈50°")
            print("  - DIFF should be near 0° if mapping is correct")
            print("─" * 80)

            time.sleep(args.rate)

    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
