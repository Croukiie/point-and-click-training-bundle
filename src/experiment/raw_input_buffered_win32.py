"""Buffered Windows Raw Input capture for polling-rate diagnostics.

Unlike :mod:`raw_input_win32`, this module intentionally leaves ``WM_INPUT``
messages in the thread queue and drains them with ``GetRawInputBuffer``.  It is
kept separate from the production experiment backend so the two acquisition
methods can be compared without changing data collection behavior.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import threading
import time

from .raw_input_win32 import (
    HID_USAGE_GENERIC_MOUSE,
    HID_USAGE_PAGE_GENERIC,
    RIDEV_INPUTSINK,
    RIM_TYPEMOUSE,
    RAWINPUT,
    RAWINPUTDEVICE,
    RAWINPUTHEADER,
    RawInputMouse,
    RawMouseEvent,
    WNDCLASSEXW,
    WNDPROC,
)


WM_INPUT = 0x00FF
PM_REMOVE = 0x0001
UINT_ERROR = 0xFFFFFFFF


class BufferedRawInputMouse(RawInputMouse):
    """Read queued mouse packets in batches with ``GetRawInputBuffer``.

    Buffered packets do not contain hardware timestamps.  Every packet in a
    batch therefore receives the batch-drain timestamp.  Event counts are
    exact, but inter-event intervals from this backend must not be interpreted
    as USB polling intervals.
    """

    def __init__(self, *, buffer_bytes: int = 65536, drain_interval_s: float = 0.0005):
        super().__init__()
        if buffer_bytes < ctypes.sizeof(RAWINPUT):
            raise ValueError("buffer_bytes is too small for one RAWINPUT structure")
        if drain_interval_s <= 0:
            raise ValueError("drain_interval_s must be positive")
        self.buffer_bytes = int(buffer_bytes)
        self.drain_interval_s = float(drain_interval_s)
        self._stop_requested = threading.Event()

    def start(self):
        self._stop_requested.clear()
        super().start()

    def stop(self):
        if self._thread is None:
            return
        self._stop_requested.set()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("Buffered Raw Input thread did not stop")
        self._thread = None

    def _message_loop(self):
        user32 = None
        hinstance = None
        class_name = None
        hwnd = None
        try:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
            kernel32.GetModuleHandleW.restype = wintypes.HMODULE
            user32.RegisterClassExW.argtypes = [ctypes.POINTER(WNDCLASSEXW)]
            user32.RegisterClassExW.restype = wintypes.ATOM
            user32.CreateWindowExW.argtypes = [
                wintypes.DWORD,
                wintypes.LPCWSTR,
                wintypes.LPCWSTR,
                wintypes.DWORD,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                wintypes.HWND,
                wintypes.HMENU,
                wintypes.HINSTANCE,
                wintypes.LPVOID,
            ]
            user32.CreateWindowExW.restype = wintypes.HWND
            user32.DefWindowProcW.argtypes = [
                wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
            ]
            user32.DefWindowProcW.restype = ctypes.c_ssize_t
            user32.DestroyWindow.argtypes = [wintypes.HWND]
            user32.DestroyWindow.restype = wintypes.BOOL
            user32.RegisterRawInputDevices.argtypes = [
                ctypes.POINTER(RAWINPUTDEVICE), wintypes.UINT, wintypes.UINT
            ]
            user32.RegisterRawInputDevices.restype = wintypes.BOOL
            user32.GetRawInputBuffer.argtypes = [
                wintypes.LPVOID,
                ctypes.POINTER(wintypes.UINT),
                wintypes.UINT,
            ]
            user32.GetRawInputBuffer.restype = wintypes.UINT
            user32.GetRawInputDeviceInfoW.argtypes = [
                wintypes.HANDLE,
                wintypes.UINT,
                wintypes.LPVOID,
                ctypes.POINTER(wintypes.UINT),
            ]
            user32.GetRawInputDeviceInfoW.restype = wintypes.UINT
            user32.PeekMessageW.argtypes = [
                ctypes.POINTER(wintypes.MSG),
                wintypes.HWND,
                wintypes.UINT,
                wintypes.UINT,
                wintypes.UINT,
            ]
            user32.PeekMessageW.restype = wintypes.BOOL
            user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
            user32.TranslateMessage.restype = wintypes.BOOL
            user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
            user32.DispatchMessageW.restype = ctypes.c_ssize_t
            user32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
            user32.UnregisterClassW.restype = wintypes.BOOL

            hinstance = kernel32.GetModuleHandleW(None)
            class_name = f"PnCBufferedRawInput_{id(self)}"

            @WNDPROC
            def wndproc(window, message, wparam, lparam):
                return user32.DefWindowProcW(window, message, wparam, lparam)

            self._wndproc_callback = wndproc
            window_class = WNDCLASSEXW()
            window_class.cbSize = ctypes.sizeof(WNDCLASSEXW)
            window_class.lpfnWndProc = wndproc
            window_class.hInstance = hinstance
            window_class.lpszClassName = class_name
            if not user32.RegisterClassExW(ctypes.byref(window_class)):
                raise ctypes.WinError(ctypes.get_last_error())

            # HWND_MESSAGE (-3) creates an invisible message-only window.
            hwnd = user32.CreateWindowExW(
                0,
                class_name,
                class_name,
                0,
                0,
                0,
                0,
                0,
                ctypes.c_void_p(-3),
                None,
                hinstance,
                None,
            )
            if not hwnd:
                raise ctypes.WinError(ctypes.get_last_error())
            self._hwnd = hwnd
            device = RAWINPUTDEVICE(
                HID_USAGE_PAGE_GENERIC,
                HID_USAGE_GENERIC_MOUSE,
                RIDEV_INPUTSINK,
                hwnd,
            )
            if not user32.RegisterRawInputDevices(
                ctypes.byref(device), 1, ctypes.sizeof(RAWINPUTDEVICE)
            ):
                raise ctypes.WinError(ctypes.get_last_error())

            raw_buffer = ctypes.create_string_buffer(self.buffer_bytes)
            self._ready.set()
            while not self._stop_requested.is_set():
                self._drain_os_raw_queue(user32, raw_buffer)
                self._dispatch_non_raw_messages(user32)
                time.sleep(self.drain_interval_s)
            self._drain_os_raw_queue(user32, raw_buffer)
        except BaseException as exc:
            self._error = exc
            self._ready.set()
        finally:
            if user32 is not None and hwnd:
                user32.DestroyWindow(hwnd)
            if user32 is not None and class_name and hinstance:
                user32.UnregisterClassW(class_name, hinstance)
            self._hwnd = None

    def _drain_os_raw_queue(self, user32, raw_buffer) -> int:
        total = 0
        header_size = ctypes.sizeof(RAWINPUTHEADER)
        base_address = ctypes.addressof(raw_buffer)
        while True:
            size = wintypes.UINT(self.buffer_bytes)
            count = int(
                user32.GetRawInputBuffer(
                    ctypes.byref(raw_buffer), ctypes.byref(size), header_size
                )
            )
            if count == 0:
                return total
            if count == UINT_ERROR:
                raise ctypes.WinError(ctypes.get_last_error())

            batch_timestamp_ns = time.perf_counter_ns()
            offset = 0
            for _ in range(count):
                if offset + header_size > self.buffer_bytes:
                    raise RuntimeError("Buffered RAWINPUT header exceeds buffer bounds")
                raw = ctypes.cast(
                    base_address + offset, ctypes.POINTER(RAWINPUT)
                ).contents
                block_size = int(raw.header.dwSize)
                if block_size < header_size or offset + block_size > self.buffer_bytes:
                    raise RuntimeError("Invalid RAWINPUT block size in buffered capture")
                if raw.header.dwType == RIM_TYPEMOUSE:
                    mouse = raw.mouse
                    device_handle = int(raw.header.hDevice or 0)
                    self._events.put(
                        RawMouseEvent(
                            timestamp_ns=batch_timestamp_ns,
                            device_handle=device_handle,
                            device_name=self._get_device_name(user32, device_handle),
                            raw_flags=int(mouse.usFlags),
                            button_flags=int(mouse.usButtonFlags),
                            button_data=int(mouse.usButtonData),
                            dx=int(mouse.lLastX),
                            dy=int(mouse.lLastY),
                            os_cursor_x_px=None,
                            os_cursor_y_px=None,
                        )
                    )
                    total += 1
                offset += block_size

    @staticmethod
    def _dispatch_non_raw_messages(user32) -> None:
        message = wintypes.MSG()
        ranges = ((0, WM_INPUT - 1), (WM_INPUT + 1, 0xFFFF))
        for minimum, maximum in ranges:
            while user32.PeekMessageW(
                ctypes.byref(message), None, minimum, maximum, PM_REMOVE
            ):
                user32.TranslateMessage(ctypes.byref(message))
                user32.DispatchMessageW(ctypes.byref(message))
