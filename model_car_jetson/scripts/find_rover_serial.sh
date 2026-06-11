#!/usr/bin/env bash
set -euo pipefail

echo "[serial] scanning tty candidates from sysfs/udev"
echo

best=""
for sys_path in /sys/class/tty/ttyUSB* /sys/class/tty/ttyACM* /sys/class/tty/ttyTHS* /sys/class/tty/ttyS*; do
  [[ -e "${sys_path}" ]] || continue
  name="$(basename "${sys_path}")"
  props="$(udevadm info -q property -p "${sys_path}" 2>/dev/null || true)"
  devname="$(printf '%s\n' "${props}" | awk -F= '$1 == "DEVNAME" { print $2; exit }')"
  model="$(printf '%s\n' "${props}" | awk -F= '$1 == "ID_MODEL" { print $2; exit }')"
  vendor="$(printf '%s\n' "${props}" | awk -F= '$1 == "ID_VENDOR" { print $2; exit }')"
  serial_short="$(printf '%s\n' "${props}" | awk -F= '$1 == "ID_SERIAL_SHORT" { print $2; exit }')"
  devlinks="$(printf '%s\n' "${props}" | awk -F= '$1 == "DEVLINKS" { print $2; exit }')"
  driver="$(awk -F= '$1 == "DRIVER" { print $2; exit }' "${sys_path}/device/uevent" 2>/dev/null || true)"
  device_path="$(readlink -f "${sys_path}/device" 2>/dev/null || true)"
  [[ -n "${devname}" ]] || devname="/dev/${name}"

  present="missing-in-current-dev-namespace"
  [[ -e "${devname}" ]] && present="present"

  printf '[serial] %-8s dev=%-14s driver=%-12s node=%s\n' "${name}" "${devname}" "${driver:-unknown}" "${present}"
  [[ -n "${vendor}" ]] && printf '         vendor=%s\n' "${vendor}"
  [[ -n "${model}" ]] && printf '         model=%s\n' "${model}"
  [[ -n "${serial_short}" ]] && printf '         serial=%s\n' "${serial_short}"
  [[ -n "${devlinks}" ]] && printf '         links=%s\n' "${devlinks}"
  [[ -n "${device_path}" ]] && printf '         sys=%s\n' "${device_path}"

  if [[ -z "${best}" ]]; then
    case "${name}:${driver}:${model}" in
      ttyUSB*:cp210x:*|ttyUSB*:*CP210*|ttyACM*:*)
        best="${devname}"
        ;;
    esac
  fi
  echo
done

if [[ -n "${best}" ]]; then
  echo "[serial] recommended rover serial: ${best}"
  echo "[serial] lane-drive example:"
  echo "         ./scripts/run_lane_drive_csi.sh 0 --web --no-display --serial ${best}"
else
  echo "[serial] no USB/ACM rover-style serial device found."
  echo "[serial] If the rover is wired to Jetson header UART, try ttyTHS1 or ttyTHS2 only after confirming the wiring."
fi
