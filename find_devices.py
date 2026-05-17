#!/usr/bin/env python3
"""
Resolves /dev/ttyACMX and /dev/videoX paths for known hardware by USB serial number.
Run this before lerobot-record to get the correct port assignments.
"""

import subprocess
import sys
from pathlib import Path

# --- Known device registry ---
# Map logical name -> USB serial ID (from ID_SERIAL_SHORT or ID_SERIAL)

SERIAL_DEVICES = {
    "leader_left":  "129A6D4B503059384C2E3120FF09113F",
    "leader_right": "55AE2891503059384C2E3120FF072140",
    "follower_left":  "0DB31F42503059384C2E3120FF07192F",
    "follower_right": "00D227A3503059384C2E3120FF06181D",
}

CAMERA_DEVICES = {
    "cam_left_wrist": "170428-_Integrated_Webcam_HD",           # USB 2.0 Camera (left arm wrist)
    "cam_right_wrist": "Suyin_HD_Camera_200910120001",          # HD Camera / Suyin (right arm wrist)
    "cam_top": "Sonix_Technology_Co.__Ltd._USB_2.0_Camera_AY2H91100TN",  # built-in/top cam
}


def udevadm_props(dev: str) -> dict[str, str]:
    try:
        out = subprocess.check_output(
            ["udevadm", "info", "-q", "property", "-n", dev],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except subprocess.CalledProcessError:
        return {}
    props = {}
    for line in out.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            props[k.strip()] = v.strip()
    return props


def find_serial_ports() -> dict[str, str]:
    results = {}
    candidates = sorted(Path("/dev").glob("ttyACM*"), key=lambda p: p.name)
    for dev in candidates:
        props = udevadm_props(str(dev))
        serial_short = props.get("ID_SERIAL_SHORT", "")
        serial_full = props.get("ID_SERIAL", "")
        for name, target_id in SERIAL_DEVICES.items():
            if target_id in (serial_short, serial_full):
                results[name] = str(dev)
    return results


def find_cameras() -> dict[str, str]:
    results = {}
    # Only check even-numbered video nodes (capture nodes; odd = metadata)
    candidates = sorted(
        [p for p in Path("/dev").glob("video*") if int(p.name[5:]) % 2 == 0],
        key=lambda p: int(p.name[5:]),
    )
    for dev in candidates:
        props = udevadm_props(str(dev))
        serial_short = props.get("ID_SERIAL_SHORT", "")
        serial_full = props.get("ID_SERIAL", "")
        for name, target_id in CAMERA_DEVICES.items():
            if target_id in (serial_short, serial_full):
                results[name] = str(dev)
    return results


def main() -> None:
    serial_ports = find_serial_ports()
    cameras = find_cameras()

    missing = []
    for name in SERIAL_DEVICES:
        if name not in serial_ports:
            missing.append(name)
    for name in CAMERA_DEVICES:
        if name not in cameras:
            missing.append(name)

    print("=== Device Resolution ===\n")
    for name, dev in sorted(serial_ports.items()):
        print(f"  {name:<20} -> {dev}")
    for name, dev in sorted(cameras.items()):
        print(f"  {name:<20} -> {dev}")

    if missing:
        print(f"\nWARNING: Could not find: {', '.join(missing)}")
        sys.exit(1)

    lw = serial_ports.get("leader_left", "MISSING")
    lr = serial_ports.get("leader_right", "MISSING")
    fw = serial_ports.get("follower_left", "MISSING")
    fr = serial_ports.get("follower_right", "MISSING")
    cam_lw = cameras.get("cam_left_wrist", "MISSING")
    cam_rw = cameras.get("cam_right_wrist", "MISSING")
    cam_top = cameras.get("cam_top", "MISSING")

    print("\n=== lerobot-record command ===\n")
    print(f"""lerobot-record \\
  --robot.type=bi_omx_follower \\
  --robot.left_arm_config.port={fw} \\
  --robot.right_arm_config.port={fr} \\
  --robot.id=bimanual_omx_follower \\
  --robot.left_arm_config.cameras='{{
    wrist: {{"type": "opencv", "index_or_path": "{cam_lw}", "width": 640, "height": 480, "fps": 25, "fourcc": "MJPG", "backend": "V4L2"}},
    top:   {{"type": "opencv", "index_or_path": "{cam_top}", "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG", "backend": "V4L2"}}}}' \\
  --robot.right_arm_config.cameras='{{
    wrist: {{"type": "opencv", "index_or_path": "{cam_rw}", "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG", "backend": "V4L2"}}}}' \\
  --teleop.type=bi_omx_leader \\
  --teleop.left_arm_config.port={lw} \\
  --teleop.right_arm_config.port={lr} \\
  --teleop.id=bimanual_omx_leader \\
  --display_data=true \\
  --dataset.push_to_hub=False \\
  --dataset.streaming_encoding=true \\
  --dataset.encoder_threads=1""")


if __name__ == "__main__":
    main()
