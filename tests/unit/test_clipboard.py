"""Clipboard images are bounded, validated, and never treated as text pastes."""

from __future__ import annotations

import ctypes
import io
import sys

import pytest
from PIL import Image
from pydantic import ValidationError

from nemotron_agent.logging_utils import AgentError
from nemotron_agent.providers.base import MAX_IMAGE_BYTES, ImageAttachment
from nemotron_agent.repl import clipboard


def png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (3, 2), "purple").save(output, format="PNG")
    return output.getvalue()


def test_image_validation_size_type_and_no_serialized_bytes():
    raw = png_bytes()
    image = clipboard.validate_image(raw)
    assert (image.mime_type, image.width, image.height) == ("image/png", 3, 2)
    assert image.data == raw
    assert "data" not in image.model_dump()
    assert raw.hex() not in repr(image)
    with pytest.raises(AgentError, match="5 MB"):
        clipboard.validate_image(b"x" * (MAX_IMAGE_BYTES + 1))
    with pytest.raises(AgentError, match="valid"):
        clipboard.validate_image(b"not an image")
    with pytest.raises(ValidationError):
        ImageAttachment(
            mime_type="image/png", data=b"x" * (MAX_IMAGE_BYTES + 1), width=1, height=1
        )


def test_animated_images_and_large_dimensions_are_rejected(monkeypatch):
    output = io.BytesIO()
    first = Image.new("RGB", (3, 2), "red")
    second = Image.new("RGB", (3, 2), "blue")
    first.save(
        output,
        format="GIF",
        save_all=True,
        append_images=[second],
        duration=100,
        loop=0,
    )
    with pytest.raises(AgentError, match="Animated"):
        clipboard.validate_image(output.getvalue())
    monkeypatch.setattr(clipboard, "MAX_IMAGE_PIXELS", 5)
    with pytest.raises(AgentError, match="dimensions"):
        clipboard.validate_image(png_bytes())


def test_helper_output_has_hard_size_and_time_limits():
    with pytest.raises(AgentError, match="limit"):
        clipboard._run_bounded(
            [sys.executable, "-c", "import os; os.write(1, b'x' * 10000)"], limit=100
        )
    with pytest.raises(AgentError, match="timed out"):
        clipboard._run_bounded(
            [sys.executable, "-c", "import time; time.sleep(3)"], limit=100, timeout=0.1
        )
    code, raw = clipboard._run_bounded([sys.executable, "-c", "print('ok')"], limit=100)
    assert code == 0 and raw == b"ok\n"


def test_wayland_clipboard_uses_fixed_image_types(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(
        clipboard.shutil,
        "which",
        lambda name: "/usr/bin/wl-paste" if name == "wl-paste" else None,
    )
    calls = []

    def run(args, *, limit, timeout=3):
        calls.append((args, limit))
        return (0, b"text/plain\nimage/png\n") if len(calls) == 1 else (0, png_bytes())

    monkeypatch.setattr(clipboard, "_run_bounded", run)
    assert clipboard.validate_image(clipboard._linux_image()).mime_type == "image/png"
    assert calls[0][0] == ["/usr/bin/wl-paste", "--list-types"]
    assert calls[1] == (
        ["/usr/bin/wl-paste", "--no-newline", "--type", "image/png"],
        MAX_IMAGE_BYTES,
    )


def test_x11_clipboard_and_missing_image_are_clear(monkeypatch):
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(
        clipboard.shutil,
        "which",
        lambda name: "/usr/bin/xclip" if name == "xclip" else None,
    )
    calls = []

    def run(args, *, limit, timeout=3):
        calls.append(args)
        return (0, b"TARGETS\nimage/png\n") if len(calls) == 1 else (0, png_bytes())

    monkeypatch.setattr(clipboard, "_run_bounded", run)
    assert clipboard.validate_image(clipboard._linux_image()).mime_type == "image/png"
    assert calls[1][-2:] == ["-t", "image/png"]
    monkeypatch.setattr(
        clipboard, "_run_bounded", lambda *args, **kwargs: (0, b"text/plain\n")
    )
    with pytest.raises(AgentError, match="No supported image"):
        clipboard._linux_image()


def test_mac_clipboard_hex_decoding(monkeypatch):
    raw = png_bytes()
    apple_output = "«data PNGf".encode() + raw.hex().upper().encode() + "»\n".encode()
    monkeypatch.setattr(
        clipboard, "_run_bounded", lambda *args, **kwargs: (0, apple_output)
    )
    assert clipboard._mac_image() == raw
    monkeypatch.setattr(
        clipboard, "_run_bounded", lambda *args, **kwargs: (0, b"invalid")
    )
    with pytest.raises(AgentError, match="decode"):
        clipboard._mac_image()


def test_windows_checks_size_before_copying_clipboard(monkeypatch):
    raw = png_bytes()
    buffer = ctypes.create_string_buffer(raw)
    state = {"size": len(raw), "locks": 0, "closed": 0}

    class Fn:
        def __init__(self, callback):
            self.callback = callback

        def __call__(self, *args):
            return self.callback(*args)

    class User32:
        OpenClipboard = Fn(lambda *_: 1)
        CloseClipboard = Fn(
            lambda: state.__setitem__("closed", state["closed"] + 1) or 1
        )
        RegisterClipboardFormatW = Fn(lambda *_: 123)
        IsClipboardFormatAvailable = Fn(lambda fmt: fmt == 123)
        GetClipboardData = Fn(lambda *_: 456)

    class Kernel32:
        GlobalSize = Fn(lambda *_: state["size"])
        GlobalLock = Fn(
            lambda *_: (
                state.__setitem__("locks", state["locks"] + 1)
                or ctypes.addressof(buffer)
            )
        )
        GlobalUnlock = Fn(lambda *_: 1)

    libraries = {"user32": User32(), "kernel32": Kernel32()}
    monkeypatch.setattr(
        ctypes, "WinDLL", lambda name, **_: libraries[name], raising=False
    )
    assert clipboard._windows_image() == raw
    assert state["closed"] == 1 and state["locks"] == 1
    state["size"] = MAX_IMAGE_BYTES + 1
    with pytest.raises(AgentError, match="5 MB"):
        clipboard._windows_image()
    assert state["locks"] == 1 and state["closed"] == 2
