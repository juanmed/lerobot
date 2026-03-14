#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import queue
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class _TimerPhaseStyle:
    foreground: str
    background: str = "black"


class CountdownTimerWindow:
    """Fullscreen Tkinter timer that can be updated from a non-UI thread."""

    _EPISODE_STYLE = _TimerPhaseStyle(foreground="#b0b0b0")
    _RESET_STYLE = _TimerPhaseStyle(foreground="#ff8c42")
    _READY_STYLE = _TimerPhaseStyle(foreground="#b0b0b0")

    def __init__(self):
        self._commands: queue.Queue[tuple[str, str | float | _TimerPhaseStyle | None]] = queue.Queue()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="lerobot-timer-window", daemon=True)
        self._enabled = True
        self._thread.start()
        self._ready.wait(timeout=2.0)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def format_remaining_seconds(remaining_s: float) -> str:
        return str(max(0, math.ceil(remaining_s)))

    def start_episode(self, duration_s: float) -> None:
        self._send("phase", self._EPISODE_STYLE)
        self._send("remaining", duration_s)

    def start_reset(self, duration_s: float) -> None:
        self._send("phase", self._RESET_STYLE)
        self._send("remaining", duration_s)

    def show_processing(self) -> None:
        self._send("phase", self._READY_STYLE)
        self._send("text", "Processing...")

    def show_ready(self) -> None:
        self._send("phase", self._READY_STYLE)
        self._send("text", "Ready")

    def update_remaining(self, remaining_s: float) -> None:
        self._send("remaining", remaining_s)

    def close(self) -> None:
        if not self._enabled:
            return
        self._send("close", None)
        self._thread.join(timeout=2.0)

    def _send(self, command: str, payload: str | float | _TimerPhaseStyle | None) -> None:
        if self._enabled:
            self._commands.put((command, payload))

    def _run(self) -> None:
        try:
            import tkinter as tk

            root = tk.Tk()
        except Exception as exc:  # nosec B110
            logging.warning(f"Could not start Tk timer window. Disabling `dataset.show_timer`. Reason: {exc}")
            self._enabled = False
            self._ready.set()
            return

        root.title("LeRobot Timer")
        root.configure(bg=self._EPISODE_STYLE.background)
        root.minsize(320, 200)
        try:
            root.state("zoomed")
        except Exception:  # nosec B110
            root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")

        label = tk.Label(
            root,
            text="0",
            bg=self._EPISODE_STYLE.background,
            fg=self._EPISODE_STYLE.foreground,
            font=("Helvetica", 180, "bold"),
        )
        label.pack(expand=True, fill="both")

        def apply_style(style: _TimerPhaseStyle) -> None:
            root.configure(bg=style.background)
            label.configure(bg=style.background, fg=style.foreground)

        def pump_commands() -> None:
            try:
                while True:
                    command, payload = self._commands.get_nowait()
                    if command == "phase":
                        apply_style(payload)
                    elif command == "remaining":
                        label.configure(text=self.format_remaining_seconds(payload))
                    elif command == "text":
                        label.configure(text=payload)
                    elif command == "close":
                        root.destroy()
                        return
            except queue.Empty:
                pass
            root.after(50, pump_commands)

        self._ready.set()
        root.after(0, pump_commands)
        root.mainloop()
