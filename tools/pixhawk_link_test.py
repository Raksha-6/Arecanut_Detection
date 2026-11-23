"""
pixhawk_link_test.py  -  Pi 5 <-> Pixhawk (ArduPilot) USB link test

1. Finds the Pixhawk's USB serial port automatically
2. Waits for a HEARTBEAT (proves the link is alive)
3. Requests telemetry and prints mode, armed state, attitude, GPS, altitude, battery
   for 15 seconds (proves we can READ the data the flight logger will need)

Run:
    ~/yoloenv/bin/python ~/koleroga/pixhawk_link_test.py
"""
import glob
import sys
import time

from pymavlink import mavutil

# ---------- 1. find the port ----------
ports = sorted(glob.glob("/dev/serial/by-id/*")) or sorted(glob.glob("/dev/ttyACM*"))
if not ports:
    sys.exit("No Pixhawk port found. Check the USB cable (must be a DATA cable) and wait 10 s after plugging in.")
port = ports[0]
print(f"Using port: {port}")

# ---------- 2. heartbeat ----------
m = mavutil.mavlink_connection(port, baud=115200)
print("Waiting for heartbeat (max 15 s)...")
hb = m.wait_heartbeat(timeout=15)
if hb is None:
    sys.exit("NO HEARTBEAT. Wait 15 s after plugging in (bootloader), check the port, then retry.")
print(f"HEARTBEAT OK  system={m.target_system} component={m.target_component} "
      f"autopilot={hb.autopilot} type={hb.type}")

# ---------- 3. ask ArduPilot to stream telemetry at 4 Hz ----------
m.mav.request_data_stream_send(m.target_system, m.target_component,
                               mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1)

wanted = ["HEARTBEAT", "ATTITUDE", "GPS_RAW_INT", "GLOBAL_POSITION_INT", "SYS_STATUS", "VFR_HUD"]
seen = {k: 0 for k in wanted}
last_print = 0
end = time.time() + 15
state = {}
while time.time() < end:
    msg = m.recv_match(type=wanted, blocking=True, timeout=1)
    if msg is None:
        continue
    t = msg.get_type()
    seen[t] += 1
    if t == "HEARTBEAT":
        state["mode"] = m.flightmode
        state["armed"] = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
    elif t == "ATTITUDE":
        state["roll"] = round(msg.roll * 57.3, 1)
        state["pitch"] = round(msg.pitch * 57.3, 1)
        state["yaw"] = round(msg.yaw * 57.3, 1)
    elif t == "GPS_RAW_INT":
        state["gps_fix"] = msg.fix_type          # 0-1 no fix, 2 = 2D, 3 = 3D, 4+ = DGPS/RTK
        state["sats"] = msg.satellites_visible
    elif t == "GLOBAL_POSITION_INT":
        state["lat"] = msg.lat / 1e7
        state["lon"] = msg.lon / 1e7
        state["rel_alt_m"] = round(msg.relative_alt / 1000, 2)
        state["heading"] = msg.hdg / 100 if msg.hdg != 65535 else None
    elif t == "SYS_STATUS":
        state["batt_V"] = round(msg.voltage_battery / 1000, 2)
    if time.time() - last_print > 1:
        print(state)
        last_print = time.time()

print("\nMessages received in 15 s:")
for k, v in seen.items():
    print(f"  {k:<20} {v:>4}  {'OK' if v else 'MISSING'}")
missing = [k for k, v in seen.items() if v == 0]
print("\nRESULT:", "ALL TELEMETRY OK" if not missing else f"missing {missing}")
