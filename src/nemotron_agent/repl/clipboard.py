"""Bounded, image-only access to the desktop clipboard.

The clipboard is read only on an explicit paste action. Bytes never touch disk,
and helper commands use fixed argument arrays without a shell.
"""

from __future__ import annotations

import binascii
import io
import os
import re
import selectors
import shutil
import subprocess
import sys
import time
import warnings
from typing import Sequence

from PIL import BmpImagePlugin, Image, UnidentifiedImageError

from nemotron_agent.logging_utils import AgentError
from nemotron_agent.providers.base import MAX_IMAGE_BYTES, ImageAttachment

MAX_IMAGE_PIXELS = 16_000_000
_FORMATS = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    "GIF": "image/gif",
}
_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif")


def _run_bounded(
    argv: Sequence[str], *, limit: int, timeout: float = 3.0
) -> tuple[int, bytes]:
    """Read a fixed clipboard helper's output with a hard byte and time limit."""
    try:
        proc = subprocess.Popen(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            shell=False,
        )
    except OSError as exc:
        raise AgentError(f"Could not access the clipboard helper: {exc}") from exc
    selector = selectors.DefaultSelector()
    data = bytearray()
    deadline = time.monotonic() + timeout
    try:
        assert proc.stdout is not None
        os.set_blocking(proc.stdout.fileno(), False)
        selector.register(proc.stdout, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AgentError("Clipboard access timed out. Try /paste again.")
            if not selector.select(remaining):
                raise AgentError("Clipboard access timed out. Try /paste again.")
            chunk = os.read(proc.stdout.fileno(), min(65536, limit + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > limit:
                raise AgentError("Clipboard data exceeds the 5 MB image limit.")
        try:
            return proc.wait(timeout=max(0.01, deadline - time.monotonic())), bytes(
                data
            )
        except subprocess.TimeoutExpired as exc:
            raise AgentError("Clipboard access timed out. Try /paste again.") from exc
    finally:
        selector.close()
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        if proc.stdout is not None:
            proc.stdout.close()


def _check_dimensions(image: Image.Image) -> None:
    width, height = image.size
    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
        raise AgentError(
            "The image dimensions are too large; use an image under 16 megapixels."
        )


def validate_image(data: bytes) -> ImageAttachment:
    """Validate actual image bytes, not a clipboard's claimed MIME type."""
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise AgentError("Clipboard image must be nonempty and at most 5 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                _check_dimensions(image)
                mime = _FORMATS.get(image.format or "")
                if mime is None:
                    raise AgentError(
                        "Use a PNG, JPEG, WebP, or non-animated GIF image."
                    )
                if getattr(image, "is_animated", False):
                    raise AgentError(
                        "Animated images are not supported; copy a still image instead."
                    )
                width, height = image.size
                image.verify()
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise AgentError("Clipboard content is not a valid, safe image.") from exc
    return ImageAttachment(mime_type=mime, data=data, width=width, height=height)


class _BoundedBuffer(io.BytesIO):
    def write(self, value: bytes) -> int:
        if self.tell() + len(value) > MAX_IMAGE_BYTES:
            raise AgentError(
                "Converted clipboard image exceeds 5 MB; use a smaller image."
            )
        return super().write(value)


def _dib_to_png(data: bytes) -> bytes:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with BmpImagePlugin.DibImageFile(io.BytesIO(data)) as image:
                _check_dimensions(image)
                image.load()
                output = _BoundedBuffer()
                image.save(output, format="PNG")
                return output.getvalue()
    except (
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise AgentError(
            "Clipboard bitmap could not be safely converted to PNG."
        ) from exc


def _windows_image() -> bytes:
    # Read raw clipboard data under the OS lock and inspect GlobalSize before
    # copying it into Python. This avoids unbounded ImageGrab allocations.
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.CloseClipboard.restype = wintypes.BOOL
    user32.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
    user32.RegisterClipboardFormatW.restype = wintypes.UINT
    user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
    user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    kernel32.GlobalSize.argtypes = [wintypes.HANDLE]
    kernel32.GlobalSize.restype = ctypes.c_size_t
    kernel32.GlobalLock.argtypes = [wintypes.HANDLE]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [wintypes.HANDLE]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    if not user32.OpenClipboard(None):
        raise AgentError("The clipboard is busy. Copy an image and try /paste again.")
    try:
        png = user32.RegisterClipboardFormatW("PNG")
        choices = [(png, "png"), (17, "dib"), (8, "dib")]
        selected = next(
            (
                (fmt, kind)
                for fmt, kind in choices
                if fmt and user32.IsClipboardFormatAvailable(fmt)
            ),
            None,
        )
        if selected is None:
            raise AgentError(
                "No image is on the clipboard. Copy an image or screenshot, not a filename."
            )
        fmt, kind = selected
        handle = user32.GetClipboardData(fmt)
        size = kernel32.GlobalSize(handle) if handle else 0
        if not size:
            raise AgentError(
                "Could not read the clipboard image. Copy it again and retry."
            )
        if size > MAX_IMAGE_BYTES:
            raise AgentError(
                "Clipboard image exceeds the 5 MB limit; copy a smaller image."
            )
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            raise AgentError("Could not access the clipboard image. Try again.")
        try:
            data = ctypes.string_at(pointer, size)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()
    return _dib_to_png(data) if kind == "dib" else data


def _mac_image() -> bytes:
    # osascript returns a PNG as a hex-encoded AppleScript data literal.
    code, raw = _run_bounded(
        ["osascript", "-e", "get the clipboard as «class PNGf»"],
        limit=MAX_IMAGE_BYTES * 2 + 128,
    )
    if code != 0:
        raise AgentError(
            "No clipboard image found. Copy an image or screenshot, then try again."
        )
    match = re.fullmatch(rb"\xc2\xabdata PNGf([0-9A-Fa-f]+)\xc2\xbb\s*", raw)
    if not match:
        raise AgentError(
            "Could not decode the clipboard image. Copy it again and retry."
        )
    try:
        return binascii.unhexlify(match.group(1))
    except binascii.Error as exc:
        raise AgentError("Could not decode the clipboard image.") from exc


def _linux_image() -> bytes:
    wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
    display = bool(os.environ.get("DISPLAY"))
    wl = shutil.which("wl-paste") if wayland else None
    xclip = shutil.which("xclip") if display else None
    if wl:
        list_cmd = [wl, "--list-types"]
    elif xclip:
        list_cmd = [xclip, "-selection", "clipboard", "-o", "-t", "TARGETS"]
    else:
        raise AgentError(
            "Clipboard images need a local desktop session and wl-paste (Wayland) or xclip (X11). "
            "If Ctrl+V is intercepted by your terminal, use /paste."
        )
    code, raw_types = _run_bounded(list_cmd, limit=16 * 1024)
    if code != 0:
        raise AgentError(
            "Could not inspect the clipboard. Copy an image and try again."
        )
    available = {
        line.strip().lower()
        for line in raw_types.decode("utf-8", "replace").splitlines()
    }
    mime = next((item for item in _TYPES if item in available), None)
    if not mime:
        raise AgentError(
            "No supported image is on the clipboard. Copy a PNG, JPEG, WebP, or still GIF image, not a filename."
        )
    if wl:
        data_cmd = [wl, "--no-newline", "--type", mime]
    else:
        data_cmd = [xclip, "-selection", "clipboard", "-o", "-t", mime]
    code, data = _run_bounded(data_cmd, limit=MAX_IMAGE_BYTES)
    if code != 0:
        raise AgentError("Could not read the clipboard image. Copy it again and retry.")
    return data


def read_clipboard_image() -> ImageAttachment:
    if sys.platform == "win32":
        data = _windows_image()
    elif sys.platform == "darwin":
        data = _mac_image()
    else:
        data = _linux_image()
    return validate_image(data)
