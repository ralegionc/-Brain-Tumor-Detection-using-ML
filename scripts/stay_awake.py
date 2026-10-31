"""Ask Windows not to idle-sleep while a given process (the training queue) is alive.
Uses the same per-process request media players make; no power settings are changed,
and the request is dropped automatically when this script exits."""
import ctypes
import sys
import time

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
SYNCHRONIZE, STILL_ACTIVE = 0x00100000, 259
k32 = ctypes.windll.kernel32


def alive(pid):
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return code.value == STILL_ACTIVE


pid = int(sys.argv[1])
k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
print("holding awake while pid", pid, "runs", flush=True)
while alive(pid):
    time.sleep(60)
k32.SetThreadExecutionState(ES_CONTINUOUS)
print("released", flush=True)
