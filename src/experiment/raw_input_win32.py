"""Capture Windows Raw Input mouse messages without pointer acceleration.

WM_INPUT delivery intervals are application/OS timestamps.  They must not be
interpreted as a direct measurement of the mouse's USB polling rate because
the driver, Windows message delivery, and the Python callback can batch data.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import queue
import threading
import time


WM_INPUT = 0x00FF
WM_CLOSE = 0x0010
WM_DESTROY = 0x0002
RID_INPUT = 0x10000003
RIDI_DEVICENAME = 0x20000007
RIM_TYPEMOUSE = 0
RIDEV_INPUTSINK = 0x00000100
HID_USAGE_PAGE_GENERIC = 0x01
HID_USAGE_GENERIC_MOUSE = 0x02

RI_MOUSE_LEFT_BUTTON_DOWN = 0x0001
RI_MOUSE_LEFT_BUTTON_UP = 0x0002
RI_MOUSE_RIGHT_BUTTON_DOWN = 0x0004
RI_MOUSE_RIGHT_BUTTON_UP = 0x0008
RI_MOUSE_MIDDLE_BUTTON_DOWN = 0x0010
RI_MOUSE_MIDDLE_BUTTON_UP = 0x0020
RI_MOUSE_WHEEL = 0x0400


@dataclass(frozen=True)
class RawMouseEvent:
    timestamp_ns: int
    device_handle: int
    device_name: str
    raw_flags: int
    button_flags: int
    button_data: int
    dx: int
    dy: int
    os_cursor_x_px: int | None
    os_cursor_y_px: int | None


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND),
    ]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM),
    ]


class _BUTTON_DATA(ctypes.Structure):
    _fields_ = [("usButtonFlags", wintypes.USHORT), ("usButtonData", wintypes.USHORT)]


class _BUTTON_UNION(ctypes.Union):
    _anonymous_ = ("buttons",)
    _fields_ = [("ulButtons", wintypes.ULONG), ("buttons", _BUTTON_DATA)]


class RAWMOUSE(ctypes.Structure):
    _anonymous_ = ("button_union",)
    _fields_ = [
        ("usFlags", wintypes.USHORT),
        ("button_union", _BUTTON_UNION),
        ("ulRawButtons", wintypes.ULONG),
        ("lLastX", wintypes.LONG),
        ("lLastY", wintypes.LONG),
        ("ulExtraInformation", wintypes.ULONG),
    ]


class _RAW_UNION(ctypes.Union):
    _fields_ = [("mouse", RAWMOUSE)]


class RAWINPUT(ctypes.Structure):
    _anonymous_ = ("data",)
    _fields_ = [("header", RAWINPUTHEADER), ("data", _RAW_UNION)]


WNDPROC = ctypes.WINFUNCTYPE(
    ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
)


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.UINT), ("style", wintypes.UINT), ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR),
        ("hIconSm", wintypes.HICON),
    ]


class RawInputMouse:
    def __init__(self):
        self._events: queue.SimpleQueue[RawMouseEvent] = queue.SimpleQueue()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._hwnd = None
        self._error: BaseException | None = None
        self._wndproc_callback = None
        self._device_name_cache: dict[int, str] = {}

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._message_loop, name="raw-input", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5):
            raise RuntimeError("Raw Input window did not start")
        if self._error is not None:
            raise RuntimeError("Raw Input initialization failed") from self._error

    def drain(self) -> list[RawMouseEvent]:
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def stop(self):
        if self._thread is None:
            return
        if self._hwnd:
            ctypes.windll.user32.PostMessageW(self._hwnd, WM_CLOSE, 0, 0)
        self._thread.join(timeout=5)
        self._thread = None

    def _message_loop(self):
        try:
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
            kernel32.GetModuleHandleW.restype = wintypes.HMODULE
            user32.RegisterClassExW.argtypes = [ctypes.POINTER(WNDCLASSEXW)]
            user32.RegisterClassExW.restype = wintypes.ATOM
            user32.CreateWindowExW.argtypes = [
                wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
            ]
            user32.CreateWindowExW.restype = wintypes.HWND
            user32.DefWindowProcW.argtypes = [
                wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
            ]
            user32.DefWindowProcW.restype = ctypes.c_ssize_t
            user32.RegisterRawInputDevices.argtypes = [
                ctypes.POINTER(RAWINPUTDEVICE), wintypes.UINT, wintypes.UINT
            ]
            user32.RegisterRawInputDevices.restype = wintypes.BOOL
            user32.GetRawInputData.argtypes = [
                wintypes.HANDLE, wintypes.UINT, wintypes.LPVOID,
                ctypes.POINTER(wintypes.UINT), wintypes.UINT,
            ]
            user32.GetRawInputData.restype = wintypes.UINT
            user32.GetRawInputDeviceInfoW.argtypes = [
                wintypes.HANDLE, wintypes.UINT, wintypes.LPVOID,
                ctypes.POINTER(wintypes.UINT),
            ]
            user32.GetRawInputDeviceInfoW.restype = wintypes.UINT
            user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
            user32.GetCursorPos.restype = wintypes.BOOL
            user32.PostMessageW.argtypes = [
                wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
            ]
            user32.PostMessageW.restype = wintypes.BOOL
            hinstance = kernel32.GetModuleHandleW(None)
            class_name = f"PnCRawInput_{id(self)}"

            @WNDPROC
            def wndproc(hwnd, message, wparam, lparam):
                if message == WM_INPUT:
                    size = wintypes.UINT(0)
                    header_size = ctypes.sizeof(RAWINPUTHEADER)
                    user32.GetRawInputData(lparam, RID_INPUT, None, ctypes.byref(size), header_size)
                    if size.value:
                        buffer = ctypes.create_string_buffer(size.value)
                        copied = user32.GetRawInputData(
                            lparam, RID_INPUT, buffer, ctypes.byref(size), header_size
                        )
                        if copied == size.value:
                            raw = ctypes.cast(buffer, ctypes.POINTER(RAWINPUT)).contents
                            if raw.header.dwType == RIM_TYPEMOUSE:
                                mouse = raw.mouse
                                device_handle = int(raw.header.hDevice or 0)
                                device_name = self._get_device_name(user32, device_handle)
                                point = wintypes.POINT()
                                has_cursor = bool(user32.GetCursorPos(ctypes.byref(point)))
                                self._events.put(
                                    RawMouseEvent(
                                        timestamp_ns=time.perf_counter_ns(),
                                        device_handle=device_handle,
                                        device_name=device_name,
                                        raw_flags=int(mouse.usFlags),
                                        button_flags=int(mouse.usButtonFlags),
                                        button_data=int(mouse.usButtonData),
                                        dx=int(mouse.lLastX),
                                        dy=int(mouse.lLastY),
                                        os_cursor_x_px=int(point.x) if has_cursor else None,
                                        os_cursor_y_px=int(point.y) if has_cursor else None,
                                    )
                                )
                    return 0
                if message == WM_CLOSE:
                    user32.DestroyWindow(hwnd)
                    return 0
                if message == WM_DESTROY:
                    user32.PostQuitMessage(0)
                    return 0
                return user32.DefWindowProcW(hwnd, message, wparam, lparam)

            self._wndproc_callback = wndproc
            window_class = WNDCLASSEXW()
            window_class.cbSize = ctypes.sizeof(WNDCLASSEXW)
            window_class.lpfnWndProc = wndproc
            window_class.hInstance = hinstance
            window_class.lpszClassName = class_name
            atom = user32.RegisterClassExW(ctypes.byref(window_class))
            if not atom:
                raise ctypes.WinError()

            # HWND_MESSAGE (-3) creates an invisible message-only window.
            hwnd = user32.CreateWindowExW(
                0, class_name, class_name, 0, 0, 0, 0, 0,
                ctypes.c_void_p(-3), None, hinstance, None,
            )
            if not hwnd:
                raise ctypes.WinError()
            self._hwnd = hwnd
            device = RAWINPUTDEVICE(
                HID_USAGE_PAGE_GENERIC, HID_USAGE_GENERIC_MOUSE, RIDEV_INPUTSINK, hwnd
            )
            if not user32.RegisterRawInputDevices(
                ctypes.byref(device), 1, ctypes.sizeof(RAWINPUTDEVICE)
            ):
                raise ctypes.WinError()

            self._ready.set()
            message = wintypes.MSG()
            while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
                user32.TranslateMessage(ctypes.byref(message))
                user32.DispatchMessageW(ctypes.byref(message))
            user32.UnregisterClassW(class_name, hinstance)
        except BaseException as exc:
            self._error = exc
            self._ready.set()
        finally:
            self._hwnd = None

    def _get_device_name(self, user32, device_handle: int) -> str:
        if not device_handle:
            return ""
        if device_handle in self._device_name_cache:
            return self._device_name_cache[device_handle]
        size = wintypes.UINT(0)
        handle = wintypes.HANDLE(device_handle)
        user32.GetRawInputDeviceInfoW(handle, RIDI_DEVICENAME, None, ctypes.byref(size))
        if not size.value:
            self._device_name_cache[device_handle] = ""
            return ""
        buffer = ctypes.create_unicode_buffer(size.value + 1)
        copied = user32.GetRawInputDeviceInfoW(
            handle, RIDI_DEVICENAME, buffer, ctypes.byref(size)
        )
        if copied == 0xFFFFFFFF:
            device_name = ""
        else:
            device_name = buffer.value
        self._device_name_cache[device_handle] = device_name
        return device_name

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop()
