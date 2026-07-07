"""Real-time detection + tracking test — shows what the detector actually sees.

Captures the game window, runs discovery every ~2s, uses local_tracker
between discoveries to follow mobs in real time, and draws overlay dots.

Usage:
    py scripts/test_detection.py --mob horn
    py scripts/test_detection.py --hwnd 123456 --mob horn
    py scripts/test_detection.py --fixture path/to/screenshot.png

Hotkeys:
    Q          — reset all tracks (simulates teleport)
    ESC / Ctrl-C — exit
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

# Python 3.14 dropped LRESULT from wintypes
if not hasattr(wintypes, "LRESULT"):
    wintypes.LRESULT = ctypes.c_longlong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_long

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Path setup ────────────────────────────────────────────────────
_MOB_REC = PROJECT_ROOT / "mob-recognition"
_MOB_SIMPLE = _MOB_REC / "simple"
for _p in (_MOB_REC, _MOB_SIMPLE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_MOB_VENV = _MOB_REC / ".venv" / "Lib" / "site-packages"
if _MOB_VENV.is_dir() and str(_MOB_VENV) not in sys.path:
    sys.path.insert(0, str(_MOB_VENV))

from capture import capture_region
from detector import SimpleMobDetector, load_simple_config
from pybot.runtime.capture.window_roi import (
    hunt_roi_from_client_rect,
    player_ignore_box,
    point_inside_ignore,
)
from pybot.app.win32_util import enum_game_windows
from scoring.heatmap_detector import palette_heatmap

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32

# ── Win32 constants ───────────────────────────────────────────────
WS_EX_LAYERED = 0x80000
WS_EX_TRANSPARENT = 0x20
WS_EX_TOOLWINDOW = 0x80
WS_EX_TOPMOST = 0x8
WS_POPUP = 0x80000000
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
WDA_EXCLUDEFROMCAPTURE = 0x00000011

COLOR_KEY = 0x00FF00FF        # magenta → transparent
COLOR_DOT_GREEN = 0x0000FF00
COLOR_DOT_YELLOW = 0x0000FFFF
COLOR_TEXT = 0x00FFFFFF
COLOR_ROI = 0x00FFE066        # amber

# Explicit argtypes so 64-bit WPARAM/LPARAM don't overflow
user32.DefWindowProcW.argtypes = [
    wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
]
user32.DefWindowProcW.restype = wintypes.LRESULT

# ── ctypes structs ────────────────────────────────────────────────


class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", ctypes.c_void_p),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HCURSOR),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


class PAINTSTRUCT(ctypes.Structure):
    _fields_ = [
        ("hdc", wintypes.HDC),
        ("fErase", wintypes.BOOL),
        ("rcPaint", wintypes.RECT),
        ("fRestore", wintypes.BOOL),
        ("fIncUpdate", wintypes.BOOL),
        ("rgbReserved", wintypes.BYTE * 32),
    ]


# ── Production constants (mirrored from hunt_track_rules.py) ────
HUNT_TRACK_MISS_LIMIT = 2
MIN_NEW_TRACK_SCORE = 0.25  # skip weak texture noise
SKIP_NEAR_TRACKED_PX = 60  # discovery skips new detections near existing tracks
HUNT_TRACK_MATCH_RADIUS = 80  # match radius for S-key reconcile
CELL_SIZE_PX = 64
DEFAULT_SEARCH_RANGE_CELLS = 16

# ── Tracked mob ──────────────────────────────────────────────────

@dataclass
class TrackedMob:
    id: int = 0
    x: int = 0
    y: int = 0
    confidence: float = 0.0
    last_seen: float = 0.0
    miss_count: int = 0
    color: int = COLOR_DOT_GREEN
    discovery_scale: float = 0.0
    attack_count: int = 0


# ── Overlay ──────────────────────────────────────────────────────

_WndProcPtr = ctypes.WINFUNCTYPE(
    wintypes.LRESULT, wintypes.HWND, wintypes.UINT,
    wintypes.WPARAM, wintypes.LPARAM,
)


class Overlay:
    """Transparent overlay showing detection dots, ROI border, and stats."""

    def __init__(self, game_hwnd: int) -> None:
        self._game_hwnd = game_hwnd
        self._hwnd: int = 0
        self._font: int = 0
        self._brush_dot_green: int = 0
        self._brush_dot_yellow: int = 0
        self._brush_roi: int = 0
        self._lock = threading.Lock()
        self._dots: list[tuple[int, int, int]] = []
        self._roi_rect: tuple[int, int, int, int] | None = None
        self._stats_text: str = ""
        self._visible = False
        # CRITICAL: Keep the ctypes callback alive as an instance attribute!
        # If this gets garbage collected, the window procedure pointer
        # becomes dangling and UpdateWindow will crash the process.
        self._wnd_proc_cb = None

    def create(self) -> bool:
        if not self._game_hwnd or not user32.IsWindow(self._game_hwnd):
            return False
        client = self._get_client()
        if client is None:
            return False
        _left, _top, cw, ch = client

        hinstance = kernel32.GetModuleHandleW(None)
        wc_name = f"DetectOv_{self._game_hwnd}_{int(time.time())}"

        # Store the callback on self so it stays alive for the window's lifetime
        self._wnd_proc_cb = _WndProcPtr(self._on_wm)

        cls = WNDCLASSW()
        cls.lpfnWndProc = ctypes.cast(self._wnd_proc_cb, ctypes.c_void_p)
        cls.hInstance = hinstance
        cls.hbrBackground = 0
        cls.lpszClassName = wc_name
        user32.RegisterClassW(ctypes.byref(cls))

        self._hwnd = user32.CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_TOPMOST,
            wc_name, "DetectionTest", WS_POPUP,
            0, 0, cw, ch, 0, 0, hinstance, 0,
        )
        if not self._hwnd:
            return False

        self._font = gdi32.CreateFontW(14, 0, 0, 0, 400, 0, 0, 0, 0, 0, 0, 0, 0, "Consolas")
        self._brush_dot_green = gdi32.CreateSolidBrush(COLOR_DOT_GREEN)
        self._brush_dot_yellow = gdi32.CreateSolidBrush(COLOR_DOT_YELLOW)
        self._brush_roi = gdi32.CreateSolidBrush(COLOR_ROI)

        user32.SetLayeredWindowAttributes(self._hwnd, COLOR_KEY, 200, 0x03)
        try:
            user32.SetWindowDisplayAffinity(self._hwnd, WDA_EXCLUDEFROMCAPTURE)
        except AttributeError:
            pass

        self._visible = True
        self._reposition()
        user32.ShowWindow(self._hwnd, 8)  # SW_SHOWNA
        self.invalidate()
        return True

    def _on_wm(self, hwnd: int, msg: int, wparam: int, lparam: int) -> int:
        try:
            if msg == 0x000F:  # WM_PAINT
                self._on_paint()
                return 0
            elif msg == 0x0014:  # WM_ERASEBKGND
                return 1
            elif msg == 0x0002:  # WM_DESTROY
                self._visible = False
                return 0
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)
        except Exception:
            return 0  # Don't let callback exceptions crash the process

    def close(self) -> None:
        self._visible = False
        if self._hwnd and user32.IsWindow(self._hwnd):
            user32.DestroyWindow(self._hwnd)
        self._hwnd = 0

    def update(self,
               dots: list[tuple[int, int, int]],
               roi_rect: tuple[int, int, int, int] | None = None,
               stats_text: str = "") -> None:
        with self._lock:
            self._dots = list(dots)
            self._roi_rect = roi_rect
            self._stats_text = stats_text
        self.invalidate()

    def _get_client(self) -> tuple[int, int, int, int] | None:
        if not self._game_hwnd or not user32.IsWindow(self._game_hwnd):
            return None
        rect = wintypes.RECT()
        if not user32.GetClientRect(self._game_hwnd, ctypes.byref(rect)):
            return None
        origin = wintypes.POINT(0, 0)
        if not user32.ClientToScreen(self._game_hwnd, ctypes.byref(origin)):
            return None
        cw = rect.right - rect.left
        ch = rect.bottom - rect.top
        if cw <= 0 or ch <= 0:
            return None
        return origin.x, origin.y, cw, ch

    def _reposition(self) -> None:
        client = self._get_client()
        if client is None:
            return
        left, top, cw, ch = client
        user32.SetWindowPos(self._hwnd, 0, left, top, cw, ch,
                            SWP_NOZORDER | SWP_NOACTIVATE)

    def _on_paint(self) -> None:
        ps = PAINTSTRUCT()
        hdc = user32.BeginPaint(self._hwnd, ctypes.byref(ps))
        if not hdc:
            return

        rect = wintypes.RECT()
        user32.GetClientRect(self._hwnd, ctypes.byref(rect))
        cw = rect.right - rect.left
        ch = rect.bottom - rect.top

        # Transparent fill (magenta pixels → transparent)
        key_brush = gdi32.CreateSolidBrush(COLOR_KEY)
        full = wintypes.RECT(0, 0, cw, ch)
        user32.FillRect(hdc, ctypes.byref(full), key_brush)
        gdi32.DeleteObject(key_brush)

        client = self._get_client()
        if client is None:
            user32.EndPaint(self._hwnd, ctypes.byref(ps))
            return
        left, top, _, _ = client

        with self._lock:
            dots = list(self._dots)
            roi_rect = self._roi_rect
            stats_text = self._stats_text

        # ROI border
        if roi_rect and self._brush_roi:
            rx, ry, rw, rh = roi_rect
            r = wintypes.RECT(rx - left, ry - top, rx - left + rw, ry - top + rh)
            user32.FrameRect(hdc, ctypes.byref(r), self._brush_roi)

        # Dots — green for fresh, yellow for stale
        if dots:
            old_b = None
            for dx, dy, color in dots:
                brush = (self._brush_dot_yellow if color == COLOR_DOT_YELLOW
                         else self._brush_dot_green)
                if not brush:
                    continue
                rx = dx - left
                ry = dy - top
                if 0 <= rx < cw and 0 <= ry < ch:
                    ob = gdi32.SelectObject(hdc, brush)
                    if old_b is None:
                        old_b = ob
                    gdi32.Ellipse(hdc, rx - 4, ry - 4, rx + 4, ry + 4)
            if old_b is not None:
                gdi32.SelectObject(hdc, old_b)

        # Stats text
        if self._font and stats_text:
            old_f = gdi32.SelectObject(hdc, self._font)
            gdi32.SetTextColor(hdc, COLOR_TEXT)
            gdi32.SetBkMode(hdc, 1)  # TRANSPARENT
            gdi32.TextOutW(hdc, 6, 6, stats_text, len(stats_text))
            gdi32.SelectObject(hdc, old_f)

        user32.EndPaint(self._hwnd, ctypes.byref(ps))

    def invalidate(self) -> None:
        if self._hwnd and user32.IsWindow(self._hwnd):
            user32.InvalidateRect(self._hwnd, None, True)
            user32.UpdateWindow(self._hwnd)


# ── Helpers ──────────────────────────────────────────────────────


def build_roi(hwnd: int,
              search_range_cells: int = 16,
              cell_size_px: int = 64):
    """Build a hunt ROI centered on the game client."""
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    origin = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        return None
    cw = rect.right - rect.left
    ch = rect.bottom - rect.top
    return hunt_roi_from_client_rect(
        origin.x, origin.y, cw, ch,
        search_range_cells=search_range_cells,
        cell_size_px=cell_size_px,
    )


def find_game_hwnd() -> int:
    windows = enum_game_windows()
    if not windows:
        print("ERROR: No game windows found. Is the game running?")
        sys.exit(1)
    if len(windows) == 1:
        return windows[0].hwnd
    print("Found game windows:")
    for idx, entry in enumerate(windows, 1):
        print(f"  [{idx}] {entry.title} (hwnd={entry.hwnd})")
    choice = input("Select window number (or 0 for first): ").strip()
    if choice and choice != "0":
        try:
            return windows[int(choice) - 1].hwnd
        except (IndexError, ValueError):
            pass
    return windows[0].hwnd


def is_key_down(vk: int) -> bool:
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


# ── Main ──────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Detection test with overlay")
    parser.add_argument("--hwnd", type=int, default=0)
    parser.add_argument("--mob", type=str, default="horn")
    parser.add_argument("--fixture", type=str, default="",
                        help="Screenshot path instead of live capture")
    args = parser.parse_args()

    config = load_simple_config()
    detector = SimpleMobDetector(PROJECT_ROOT, config)
    mob_name = args.mob.lower()

    try:
        detector.ensure_descriptor(mob_name)
    except FileNotFoundError:
        print(f"ERROR: Descriptor for '{mob_name}' not found. Build it first.")
        return 1

    # ── Fixture mode ────────────────────────────────────────────────
    if args.fixture:
        path = Path(args.fixture)
        if not path.exists():
            print(f"ERROR: Fixture not found: {path}")
            return 1
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"ERROR: Could not read: {path}")
            return 1
        h, w = frame.shape[:2]
        print(f"Testing: {path.name} ({w}x{h})")
        result = detector.detect(frame, mob_name)
        print(f"  Candidates: {len(result.candidates)}")
        print(f"  Accepted:   {len(result.accepted)}")
        for i, c in enumerate(result.accepted):
            print(f"  [{i}] center=({c.center_x},{c.center_y}) "
                  f"body={c.body_palette_score:.4f} "
                  f"accent={c.accent_score:.4f} "
                  f"size={c.size_score:.4f} "
                  f"heat={c.heatmap_score:.4f}")
        for i, c in enumerate(result.candidates):
            if not c.accepted:
                print(f"  [REJ {i}] center=({c.center_x},{c.center_y}) "
                      f"reason={c.rejection_reason}")
        return 0

    # ── Live mode ───────────────────────────────────────────────────
    hwnd = args.hwnd or find_game_hwnd()
    print(f"Game window: hwnd={hwnd}")
    print(f"Target mob:  {mob_name}")
    print("Controls: Q = reset tracks, S = save frame + detections, ESC = quit")
    print()

    overlay = Overlay(hwnd)
    if not overlay.create():
        print("WARNING: Overlay creation failed. Detection still works.")
    else:
        print("Overlay created.")

    roi = build_roi(hwnd)
    if roi is None:
        print("ERROR: Could not determine game window ROI.")
        return 1
    print(f"ROI: {roi.x},{roi.y} {roi.w}x{roi.h}\n")

    # ── Tracking state ─────────────────────────────────────────────
    # Mirrors HuntTracks: list of TrackedMob with production miss/match logic
    tracked: list[TrackedMob] = []
    track_id_gen = 1
    last_q = False
    last_s = False
    scan_count = 0
    teleport_until = 0.0  # skip discovery until this timestamp (time-based cooldown)
    start_time = time.time()

    # Cache descriptor for fast tracking lookups
    mob_descriptor = detector.ensure_descriptor(mob_name)

    try:
        while not is_key_down(0x1B):  # ESC
            now = time.time()

            # ── Q press → teleport: wipe tracks, skip one discovery ─
            q_down = is_key_down(ord('Q'))
            if q_down and not last_q:
                tracked.clear()
                teleport_until = time.time() + 0.5  # skip discovery for 500ms
                overlay.update([], (roi.x, roi.y, roi.w, roi.h),
                               "TELEPORT")
                print(f"[TELEPORT] scan {scan_count} — track wipe + cooldown")
            last_q = q_down

            # ── S press → detect, reconcile tracks immediately, save ─
            s_down = is_key_down(ord('S'))
            if s_down and not last_s:
                s_start = time.perf_counter()
                try:
                    save_frame = capture_region(roi.x, roi.y, roi.w, roi.h)
                    if save_frame is not None and save_frame.size > 0:
                        save_result = detector.detect(save_frame, mob_name)
                        detect_ms = (time.perf_counter() - s_start) * 1000

                        # Reconcile into tracked list immediately (same logic as discovery)
                        ignore_x, ignore_y, ignore_w, ignore_h = player_ignore_box(roi, CELL_SIZE_PX)
                        reconciles = 0
                        for c in save_result.accepted:
                            if c.is_dead:
                                continue
                            sx = c.center_x + roi.x
                            sy = c.center_y + roi.y
                            if point_inside_ignore(sx, sy, ignore_x, ignore_y, ignore_w, ignore_h):
                                continue
                            # Check if this matches an existing track
                            matched = None
                            for t in tracked:
                                dx = sx - t.x
                                dy = sy - t.y
                                if dx * dx + dy * dy <= HUNT_TRACK_MATCH_RADIUS * HUNT_TRACK_MATCH_RADIUS:
                                    matched = t
                                    break
                            if matched is not None:
                                if matched.attack_count == 0:
                                    matched.x = sx
                                    matched.y = sy
                                matched.last_seen = time.time()
                                matched.miss_count = 0
                                matched.confidence = c.final_score
                                matched.color = COLOR_DOT_GREEN
                                matched.discovery_scale = c.candidate_scale
                            elif c.final_score >= MIN_NEW_TRACK_SCORE:
                                track_id_gen += 1
                                tracked.append(TrackedMob(
                                    id=track_id_gen,
                                    x=sx, y=sy,
                                    confidence=c.final_score,
                                    last_seen=time.time(),
                                    color=COLOR_DOT_GREEN,
                                    discovery_scale=c.candidate_scale,
                                ))
                            reconciles += 1

                        # Update overlay immediately
                        dots = [(t.x, t.y, t.color) for t in tracked]
                        alive = sum(1 for t in tracked if t.color == COLOR_DOT_GREEN)
                        overlay.update(dots, (roi.x, roi.y, roi.w, roi.h),
                                       f"Mobs:{len(tracked)} alive:{alive} scan:{scan_count}")

                        # Save frame + diagnostic JSON
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        save_dir = PROJECT_ROOT / "logs" / "debug_saves"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        png_path = save_dir / f"frame_{ts}.png"
                        cv2.imwrite(str(png_path), save_frame)
                        diag = {
                            "mob": mob_name,
                            "roi": [roi.x, roi.y, roi.w, roi.h],
                            "detectMs": round(detect_ms, 1),
                            "candidates": [{
                                "center": [c.center_x, c.center_y],
                                "accepted": c.accepted,
                                "score": round(c.final_score, 4),
                                "body": round(c.body_palette_score, 4),
                                "accent": round(c.accent_score, 4),
                                "purity": round(c.color_purity_score, 4),
                                "size": round(c.size_score, 4),
                                "pattern": round(c.local_pattern_score, 4),
                                "rare": round(c.rare_color_score, 4),
                                "heat": round(c.heatmap_score, 4),
                                "scale": round(c.candidate_scale, 4),
                                "dead": c.is_dead,
                                "rejection": c.rejection_reason,
                            } for c in save_result.candidates]
                        }
                        (save_dir / f"frame_{ts}.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
                        print(f"[S] {reconciles} tracked in {detect_ms:.0f}ms | {save_result.candidates} raw")
                except Exception as e:
                    print(f"[S ERROR] {e}")
            last_s = s_down

            # ── Rebuild ROI (window may have moved) ─────────────────
            new_roi = build_roi(hwnd, DEFAULT_SEARCH_RANGE_CELLS, CELL_SIZE_PX)
            if new_roi is None:
                if not user32.IsWindow(hwnd):
                    print("ERROR: Game window closed.")
                    break
                time.sleep(0.05)
                continue
            roi = new_roi

            # ── Capture ONCE per iteration (shared by discovery + tracking) ─
            try:
                frame = capture_region(roi.x, roi.y, roi.w, roi.h)
            except Exception as e:
                print(f"[ERROR] capture: {e}")
                time.sleep(1.0)
                continue
            if frame is None or frame.size == 0:
                continue

            now = time.time()

            # ── TRACKING FIRST: every iteration for ALL tracks ─────────
            # Tracking runs BEFORE discovery so dots update immediately.
            # Uses a proper gate chain (body + accent + local pattern)
            # so texture noise doesn't keep false tracks alive.
            for t in tracked:
                cx = t.x - roi.x
                cy = t.y - roi.y
                try:
                    half = 90  # 180x180 window — mob moves ~10px in 50ms
                    x0 = max(0, cx - half)
                    y0 = max(0, cy - half)
                    x1 = min(frame.shape[1], cx + half)
                    y1 = min(frame.shape[0], cy + half)
                    if x1 - x0 < 20 or y1 - y0 < 20:
                        t.miss_count += 1
                        t.color = COLOR_DOT_YELLOW
                        continue
                    crop = frame[y0:y1, x0:x1]
                    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

                    # Gate 1: body palette heatmap (dominant + supporting colors)
                    body = palette_heatmap(crop_hsv, mob_descriptor.body_palette)
                    blur = cv2.blur(body, (21, 21))
                    _, max_val, _, max_loc = cv2.minMaxLoc(blur)

                    # Gate 2: accent heatmap (rejects texture that lacks mob highlights)
                    accent = palette_heatmap(crop_hsv, mob_descriptor.accent_colors)
                    blur_accent = cv2.blur(accent, (15, 15))
                    _, accent_val, _, _ = cv2.minMaxLoc(blur_accent)

                    # Gate 3: local pattern (gradient edges within body heatmap)
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                    edge = cv2.magnitude(grad_x, grad_y)
                    if float(edge.max()) > 0:
                        edge = edge / float(edge.max())
                    pattern = np.maximum(accent * 0.75, body * edge)
                    blur_pattern = cv2.blur(pattern, (15, 15))
                    _, pattern_val, _, _ = cv2.minMaxLoc(blur_pattern)

                    # Combined gate: requires all three scores to be meaningful
                    if (max_val >= 0.12
                            and accent_val >= 0.08
                            and pattern_val >= 0.06):
                        new_cx = max_loc[0] + x0 + roi.x
                        new_cy = max_loc[1] + y0 + roi.y
                        t.x = new_cx
                        t.y = new_cy
                        t.last_seen = now
                        t.miss_count = 0
                        t.confidence = float(max_val)
                        t.color = COLOR_DOT_GREEN
                        t.discovery_scale = 1.0
                    else:
                        t.miss_count += 1
                        t.color = COLOR_DOT_YELLOW
                except Exception:
                    t.miss_count += 1
                    t.color = COLOR_DOT_YELLOW

            # ── OVERLAY: show tracking positions immediately ─
            dots = [(t.x, t.y, t.color) for t in tracked]
            alive = sum(1 for t in tracked if t.color == COLOR_DOT_GREEN)
            overlay.update(dots, (roi.x, roi.y, roi.w, roi.h),
                           f"Mobs:{len(tracked)} alive:{alive} scan:{scan_count}")

            # ── DISCOVERY: every iteration — uses track info to skip ───
            # After teleport, skip discovery for 2 iterations so old mobs
            # on the same frame don't immediately recreate tracks.
            if time.time() < teleport_until:
                pass  # still in teleport cooldown
            else:
                try:
                    result = detector.detect(frame, mob_name)
                except Exception as e:
                    print(f"[ERROR] detect: {e}")
                    time.sleep(0.05)
                    continue

                scan_count += 1

                ignore_x, ignore_y, ignore_w, ignore_h = player_ignore_box(roi, CELL_SIZE_PX)
                # Each detection is checked against CURRENT tracked positions
                # (updated by tracking above). If a detection is within the
                # skip radius of any tracked mob, it's assumed to be the
                # same mob and is NOT added as a new track.
                for c in result.accepted:
                    if c.is_dead:
                        continue
                    sx = c.center_x + roi.x
                    sy = c.center_y + roi.y
                    if point_inside_ignore(sx, sy, ignore_x, ignore_y, ignore_w, ignore_h):
                        continue
                    already_tracked = any(
                        (sx - t.x) ** 2 + (sy - t.y) ** 2
                        <= SKIP_NEAR_TRACKED_PX * SKIP_NEAR_TRACKED_PX
                        for t in tracked
                    )
                    if already_tracked:
                        continue
                    if c.final_score >= MIN_NEW_TRACK_SCORE:
                        track_id_gen += 1
                        tracked.append(TrackedMob(
                            id=track_id_gen,
                            x=sx, y=sy,
                            confidence=c.final_score,
                            last_seen=now,
                            color=COLOR_DOT_GREEN,
                            discovery_scale=c.candidate_scale,
                        ))

                # Diagnostics: every 5th scan
                if scan_count <= 5 or scan_count % 10 == 0:
                    reasons: dict[str, int] = {}
                    for cand in result.candidates:
                        if cand.accepted:
                            continue
                        r = cand.rejection_reason or "unknown"
                        reasons[r] = reasons.get(r, 0) + 1
                    reason_str = " ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
                    print(f"[#{scan_count}] raw={len(result.candidates)} "
                          f"accepted={len(result.accepted)} "
                          f"tracks={len(tracked)} "
                          f"| {reason_str}")

            # ── Remove stale tracks ─
            removed: list[int] = []
            for t in tracked:
                if t.miss_count >= HUNT_TRACK_MISS_LIMIT:
                    removed.append(t.id)
            if removed:
                tracked = [t for t in tracked if t.id not in set(removed)]

            # ── OVERLAY: update every iteration ─
            dots = [(t.x, t.y, t.color) for t in tracked]
            alive = sum(1 for t in tracked if t.color == COLOR_DOT_GREEN)
            stats = (f"Mobs:{len(tracked)} alive:{alive} scan:{scan_count}")
            overlay.update(dots, (roi.x, roi.y, roi.w, roi.h), stats)

            time.sleep(0.05)  # 20 Hz poll for ESC/Q without busy-wait

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[Exit] shutting down...")
        overlay.close()
    elapsed = time.time() - start_time
    print(f"Scans: {scan_count} in {elapsed:.1f}s "
          f"({scan_count / max(elapsed, 0.01):.1f} FPS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
