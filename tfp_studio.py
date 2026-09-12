from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from tfp.checkpoints import checkpoint_path, resolve_checkpoint_path
from tfp.evaluation import evaluate_policy, load_checkpoint_model
from tfp.models import discover_model_plugins
from tfp.models.model_api import normalize_model_name
from tfp.policies import discover_policy_plugins
from tfp.rewards import discover_reward_plugins
from tfp.reporting import collect_evaluation_runs, rebuild_all_reports
from tfp.runtime import resolve_device, runtime_status
from tfp.tasks import create_task_env, discover_tasks, get_task
from tfp.training import PPOConfig, train_ppo
from tfp.utils import discover_maps

ROOT = Path(__file__).resolve().parent

PALETTE = {
    # Soft release palette: blush pink, milk-mint, powder blue and clean white.
    "bg": "#FFF9FC",
    "card": "#FFFFFF",
    "ink": "#5D505B",
    "muted": "#9B8793",
    "pink": "#E6A9C0",
    "pink_hover": "#D993AF",
    "pink_soft": "#FBEAF1",
    "pink_pale": "#FFF3F7",
    "mint": "#CDEEE2",
    "mint_soft": "#EFFAF6",
    "mint_ink": "#527D72",
    "blue": "#DDEFFA",
    "blue_soft": "#F0F8FD",
    "blue_ink": "#66889E",
    "field": "#FFFCFE",
    "border": "#F0E0E8",
    "grid": "#F6EAF0",
    "scroll": "#EFD6E1",
    "scroll_trough": "#FAF2F6",
}



class MetricChart(tk.Canvas):
    """Small dependency-free training chart for quick experiment inspection."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, background=PALETTE["card"], highlightthickness=1, highlightbackground=PALETTE["border"], **kwargs)
        self.points: list[tuple[float, float]] = []
        self.bind("<Configure>", lambda _: self.redraw())

    def add_point(self, progress: float, value: float) -> None:
        self.points.append((float(progress), float(value)))
        self.redraw()

    def reset(self) -> None:
        self.points.clear()
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        w = max(1, self.winfo_width())
        h = max(1, self.winfo_height())
        pad = 28
        self.create_text(12, 10, anchor="nw", text="Quick-test success", fill=PALETTE["muted"], font=("Segoe UI", 9))
        for frac in (0.0, 0.5, 1.0):
            y = h - pad - frac * max(1, h - 2 * pad)
            self.create_line(pad, y, w - 12, y, fill=PALETTE["grid"])
            self.create_text(6, y, anchor="w", text=f"{frac:.1f}", fill="#A8949F", font=("Segoe UI", 8))
        if len(self.points) < 2:
            return
        coords: list[float] = []
        for progress, value in self.points:
            x = pad + max(0.0, min(1.0, progress)) * max(1, w - pad - 16)
            y = h - pad - max(0.0, min(1.0, value)) * max(1, h - 2 * pad)
            coords.extend([x, y])
        self.create_line(*coords, fill=PALETTE["pink"], width=3, smooth=True)
        x, y = coords[-2], coords[-1]
        self.create_oval(x - 4, y - 4, x + 4, y + 4, fill=PALETTE["pink"], outline="")



def _draw_round_rect(canvas: tk.Canvas, x1: float, y1: float, x2: float, y2: float, radius: float, **kwargs):
    """Draw a true rounded rectangle from overlapping ovals/rectangles."""
    radius = max(2.0, min(radius, (x2 - x1) / 2, (y2 - y1) / 2))
    fill = kwargs.get("fill", "")
    outline = kwargs.get("outline", "")
    canvas.create_rectangle(x1 + radius, y1, x2 - radius, y2, fill=fill, outline=outline)
    canvas.create_rectangle(x1, y1 + radius, x2, y2 - radius, fill=fill, outline=outline)
    canvas.create_oval(x1, y1, x1 + 2 * radius, y1 + 2 * radius, fill=fill, outline=outline)
    canvas.create_oval(x2 - 2 * radius, y1, x2, y1 + 2 * radius, fill=fill, outline=outline)
    canvas.create_oval(x1, y2 - 2 * radius, x1 + 2 * radius, y2, fill=fill, outline=outline)
    return canvas.create_oval(x2 - 2 * radius, y2 - 2 * radius, x2, y2, fill=fill, outline=outline)


class SoftRoundedButton(tk.Canvas):
    """Small rounded Tk button used for the primary actions.

    Tk/ttk does not expose a portable border-radius property. Drawing only the key
    action controls on a Canvas keeps the dependency footprint unchanged while giving
    the Studio a softer, more app-like visual language.
    """

    def __init__(
        self,
        parent,
        text: str,
        command,
        *,
        width: int = 132,
        height: int = 38,
        fill: str | None = None,
        hover: str | None = None,
        foreground: str = "#FFFFFF",
        surface: str | None = None,
        font=("Segoe UI Semibold", 10),
    ) -> None:
        self._surface = surface or PALETTE["bg"]
        super().__init__(parent, width=width, height=height, background=self._surface, highlightthickness=0, bd=0, cursor="hand2")
        self._text = text
        self._command = command
        self._fill = fill or PALETTE["pink"]
        self._hover = hover or PALETTE["pink_hover"]
        self._foreground = foreground
        self._font = font
        self._state = "normal"
        self._active = False
        self.bind("<Configure>", lambda _e: self._redraw())
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self._redraw()

    def _redraw(self) -> None:
        self.delete("all")
        w = max(10, self.winfo_width())
        h = max(10, self.winfo_height())
        disabled = self._state == "disabled"
        fill = "#F1E5EA" if disabled else (self._hover if self._active else self._fill)
        fg = "#B8A8B0" if disabled else self._foreground
        _draw_round_rect(self, 2, 2, w - 2, h - 2, min(15, h / 2), fill=fill, outline="")
        self.create_text(w / 2, h / 2, text=self._text, fill=fg, font=self._font)

    def _on_enter(self, _event=None) -> None:
        if self._state != "disabled":
            self._active = True
            self._redraw()

    def _on_leave(self, _event=None) -> None:
        self._active = False
        self._redraw()

    def _on_click(self, _event=None) -> None:
        if self._state != "disabled" and callable(self._command):
            self._command()

    def configure(self, cnf=None, **kwargs):  # type: ignore[override]
        if "state" in kwargs:
            self._state = str(kwargs.pop("state"))
            self.configure(cursor="" if self._state == "disabled" else "hand2") if False else None
            self._redraw()
        if kwargs or cnf is not None:
            return super().configure(cnf, **kwargs)
        return None

    config = configure


class RoundedMetricCard(tk.Canvas):
    """Rounded pastel metric tile with a live StringVar value."""

    def __init__(self, parent, name: str, variable: tk.StringVar, fill: str, **kwargs) -> None:
        super().__init__(parent, height=78, background=PALETTE["card"], highlightthickness=0, bd=0, **kwargs)
        self.name = name
        self.variable = variable
        self.fill = fill
        self.bind("<Configure>", lambda _e: self.redraw())
        self.variable.trace_add("write", lambda *_: self.redraw())
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        w = max(80, self.winfo_width())
        h = max(64, self.winfo_height())
        _draw_round_rect(self, 2, 2, w - 2, h - 2, 16, fill=self.fill, outline="")
        self.create_text(14, 14, anchor="nw", text=self.name, fill="#987F8D", font=("Segoe UI", 9))
        self.create_text(14, 38, anchor="nw", text=self.variable.get(), fill=PALETTE["mint_ink"], font=("Segoe UI Semibold", 17))


class VerticalScrolledFrame(ttk.Frame):
    """A vertical mouse-wheel scroll container for dense Studio forms."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, background=PALETTE["bg"], highlightthickness=0, borderwidth=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.content = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

    def _on_content_configure(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        self.canvas.itemconfigure(self._window, width=event.width)

    def _bind_mousewheel(self, _event=None) -> None:
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        self.bind_all("<Button-4>", self._on_mousewheel)
        self.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event=None) -> None:
        self.unbind_all("<MouseWheel>")
        self.unbind_all("<Button-4>")
        self.unbind_all("<Button-5>")

    def _on_mousewheel(self, event) -> None:
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = -int(event.delta / 120) if event.delta else 0
        if delta:
            self.canvas.yview_scroll(delta, "units")


class TFPStudio(tk.Tk):
    """Desktop research interface for TFP experiments."""

    def __init__(self) -> None:
        super().__init__()
        self.title("TFP Studio · Traveling Fox Problems")
        self.geometry("1320x900")
        self.minsize(1120, 760)
        self.log_queue: queue.Queue[object] = queue.Queue()
        self.worker: threading.Thread | None = None
        self._configure_style()
        self._build_vars()
        self._build_ui()
        self._refresh_plugins()
        self._update_runtime_card()
        self.after(100, self._drain_log_queue)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        bg = PALETTE["bg"]
        card = PALETTE["card"]
        ink = PALETTE["ink"]
        muted = PALETTE["muted"]
        accent = PALETTE["pink"]
        accent_hover = PALETTE["pink_hover"]
        field = PALETTE["field"]
        self.configure(background=bg)
        style.configure("TFrame", background=bg)
        style.configure("Card.TFrame", background=card, relief="flat")
        style.configure("PinkCard.TFrame", background=PALETTE["pink_soft"], relief="flat")
        style.configure("MintCard.TFrame", background=PALETTE["mint_soft"], relief="flat")
        style.configure("BlueCard.TFrame", background=PALETTE["blue_soft"], relief="flat")
        style.configure("TLabel", background=bg, foreground=ink, font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=card, foreground=ink, font=("Segoe UI", 10))
        style.configure("PinkCard.TLabel", background=PALETTE["pink_soft"], foreground=ink, font=("Segoe UI", 10))
        style.configure("MintCard.TLabel", background=PALETTE["mint_soft"], foreground=ink, font=("Segoe UI", 10))
        style.configure("BlueCard.TLabel", background=PALETTE["blue_soft"], foreground=ink, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=bg, foreground="#775A68", font=("Segoe UI Semibold", 23))
        style.configure("Subtitle.TLabel", background=bg, foreground=muted, font=("Segoe UI", 10))
        style.configure("Metric.TLabel", background=card, foreground=PALETTE["mint_ink"], font=("Segoe UI Semibold", 17))
        style.configure("MetricName.TLabel", background=card, foreground="#9A8791", font=("Segoe UI", 9))
        style.configure("Accent.TButton", background=accent, foreground="#FFFFFF", font=("Segoe UI Semibold", 10), padding=(14, 8), borderwidth=0)
        style.map("Accent.TButton", background=[("active", accent_hover), ("disabled", "#E4CFD8")])
        style.configure("TButton", padding=(10, 6), background=PALETTE["pink_soft"], foreground=ink, borderwidth=0)
        style.map("TButton", background=[("active", PALETTE["mint_soft"]), ("pressed", PALETTE["blue_soft"])])
        style.configure("TNotebook", background=bg, borderwidth=0)
        style.configure("TNotebook.Tab", background=PALETTE["blue_soft"], foreground="#756A72", padding=(16, 9), font=("Segoe UI Semibold", 10), borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", PALETTE["pink_soft"]), ("active", PALETTE["mint_soft"])], foreground=[("selected", "#7C5668")])
        style.configure("Horizontal.TProgressbar", troughcolor=PALETTE["blue"], background=accent)
        style.configure("Treeview", background=card, fieldbackground=card, foreground=ink, rowheight=28, borderwidth=0)
        style.configure("Treeview.Heading", background=PALETTE["blue_soft"], foreground="#625861", font=("Segoe UI Semibold", 9), relief="flat")
        style.map("Treeview", background=[("selected", PALETTE["mint"])] , foreground=[("selected", "#435F58")])
        style.configure("TEntry", fieldbackground=field, bordercolor=PALETTE["border"], lightcolor=PALETTE["border"], darkcolor=PALETTE["border"])
        style.configure("TCombobox", fieldbackground=PALETTE["pink_pale"], background=PALETTE["blue_soft"], foreground=ink, arrowcolor="#A2778B", bordercolor=PALETTE["border"], lightcolor=PALETTE["border"], darkcolor=PALETTE["border"])
        style.map("TCombobox",
                  fieldbackground=[("readonly", PALETTE["pink_pale"]), ("disabled", PALETTE["blue_soft"])],
                  selectbackground=[("readonly", PALETTE["pink_pale"])],
                  selectforeground=[("readonly", ink)],
                  background=[("readonly", PALETTE["blue_soft"]), ("active", PALETTE["mint_soft"])])
        style.configure("TSeparator", background=PALETTE["border"])
        style.configure("TLabelframe", background=card, foreground=ink, bordercolor=PALETTE["border"], lightcolor=PALETTE["border"], darkcolor=PALETTE["border"], relief="flat")
        style.configure("TLabelframe.Label", background=card, foreground=PALETTE["blue_ink"], font=("Segoe UI Semibold", 9))
        style.configure("TCheckbutton", background=card, foreground=ink, focuscolor=PALETTE["pink_soft"])
        style.map("TCheckbutton", background=[("active", card)], indicatorcolor=[("selected", PALETTE["pink"]), ("!selected", PALETTE["blue_soft"])])
        style.configure("Vertical.TScrollbar", background=PALETTE["scroll"], troughcolor=PALETTE["scroll_trough"], bordercolor=PALETTE["scroll_trough"], arrowcolor=PALETTE["muted"], relief="flat")
        style.configure("Horizontal.TScrollbar", background=PALETTE["scroll"], troughcolor=PALETTE["scroll_trough"], bordercolor=PALETTE["scroll_trough"], arrowcolor=PALETTE["muted"], relief="flat")

    def _build_vars(self) -> None:
        self.task_var = tk.StringVar(value="TFP-LocalTransport")
        self.model_var = tk.StringVar(value="001_simple_cnn")
        self.policy_var = tk.StringVar(value="001_ppo_categorical")
        self.reward_var = tk.StringVar(value="001_dense_transport")
        self.device_var = tk.StringVar(value="cuda")
        self.timesteps_var = tk.StringVar(value="120000")
        self.num_envs_var = tk.StringVar(value="16")
        self.train_seed_var = tk.StringVar(value="7")
        self.obs_var = tk.StringVar(value="local")
        self.view_var = tk.StringVar(value="5")
        self.mask_var = tk.StringVar(value="task")
        self.curriculum_var = tk.BooleanVar(value=True)
        self.rollout_var = tk.StringVar(value="64")
        self.epochs_var = tk.StringVar(value="4")
        self.minibatch_var = tk.StringVar(value="256")
        self.gamma_var = tk.StringVar(value="0.98")
        self.gae_var = tk.StringVar(value="0.95")
        self.clip_var = tk.StringVar(value="0.2")
        self.lr_var = tk.StringVar(value="0.0004")
        self.entropy_var = tk.StringVar(value="0.012")
        self.eval_every_var = tk.StringVar(value="20000")
        # Checkpoints are derived from task + model plugin name. Researchers do not
        # browse for weights manually, which prevents accidental cross-model loads.
        self.checkpoint_var = tk.StringVar(value="Auto")
        self.eval_checkpoint_var = tk.StringVar(value="Auto")
        self.eval_checkpoint_role_var = tk.StringVar(value="last")
        self.eval_seed_var = tk.StringVar(value="101,211,307,401,503,601,701,809,907,1009")
        self.eval_mask_var = tk.StringVar(value="checkpoint")
        self.eval_policy_var = tk.StringVar(value="checkpoint")
        self.eval_reward_var = tk.StringVar(value="checkpoint")
        self.presentation_var = tk.StringVar(value="Data only")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.metric_steps_var = tk.StringVar(value="0")
        self.metric_train_var = tk.StringVar(value="0.000")
        self.metric_test_var = tk.StringVar(value="—")
        self.metric_eff_var = tk.StringVar(value="—")
        self.metric_loss_var = tk.StringVar(value="—")
        self.runtime_var = tk.StringVar(value="Checking runtime…")
        self.experiment_tag_var = tk.StringVar(value="")
        self.compare_task_var = tk.StringVar(value="LOCAL-TRANSPORT")
        self.compare_scope_var = tk.StringVar(value="All records")

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=(24, 18, 24, 10))
        header.pack(fill="x")
        title_row = ttk.Frame(header)
        title_row.pack(fill="x")
        ttk.Label(title_row, text="TFP Studio", style="Title.TLabel").pack(side="left")
        self.save_config_button = SoftRoundedButton(title_row, "Save Config", self._save_config, width=112, height=34, surface=PALETTE["bg"])
        self.save_config_button.pack(side="right", padx=(8, 0))
        self.load_config_button = SoftRoundedButton(
            title_row, "Load Config", self._load_config, width=112, height=34,
            fill=PALETTE["blue"], hover="#CFE6F5", foreground=PALETTE["blue_ink"], surface=PALETTE["bg"]
        )
        self.load_config_button.pack(side="right")
        ttk.Label(
            header,
            text="Traveling Fox Problems · compact deep-RL robotics research with reproducible map × seed evaluation.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        runtime_card = ttk.Frame(self, style="MintCard.TFrame", padding=(16, 10))
        runtime_card.pack(fill="x", padx=24, pady=(0, 10))
        ttk.Label(runtime_card, textvariable=self.runtime_var, style="MintCard.TLabel").pack(side="left")
        SoftRoundedButton(
            runtime_card, "Launcher Source", lambda: self._open_path(ROOT / "launchers" / "windows"),
            width=124, height=32, fill=PALETTE["blue"], hover="#CFE6F5", foreground=PALETTE["blue_ink"], surface=PALETTE["mint_soft"]
        ).pack(side="right")
        SoftRoundedButton(
            runtime_card, "Research Plugins", lambda: self._open_path(ROOT / "tfp"),
            width=126, height=32, fill=PALETTE["pink_soft"], hover="#F5DCE7", foreground="#875F72", surface=PALETTE["mint_soft"]
        ).pack(side="right", padx=(0, 8))

        # A vertical paned layout guarantees the Research Log remains reachable
        # even on maximized windows or small laptop displays.
        workspace = tk.PanedWindow(self, orient="vertical", bg=PALETTE["bg"], bd=0, sashwidth=7, sashrelief="flat")
        workspace.pack(fill="both", expand=True, padx=24, pady=(0, 18))

        notebook_host = ttk.Frame(workspace)
        notebook = ttk.Notebook(notebook_host)
        notebook.pack(fill="both", expand=True)
        train_tab = VerticalScrolledFrame(notebook)
        eval_tab = VerticalScrolledFrame(notebook)
        compare_tab = ttk.Frame(notebook, padding=14)
        research_tab = VerticalScrolledFrame(notebook)
        notebook.add(train_tab, text="Train")
        notebook.add(eval_tab, text="Evaluate")
        notebook.add(compare_tab, text="Compare")
        notebook.add(research_tab, text="Research Plugins")
        self._build_train_tab(train_tab.content)
        self._build_eval_tab(eval_tab.content)
        self._build_compare_tab(compare_tab)
        self._build_research_tab(research_tab.content)

        log_frame = ttk.Frame(workspace, style="Card.TFrame", padding=(12, 9))
        log_header = ttk.Frame(log_frame, style="Card.TFrame")
        log_header.pack(fill="x")
        ttk.Label(log_header, text="Research Log", style="Card.TLabel", font=("Segoe UI Semibold", 10)).pack(side="left")
        ttk.Label(log_header, text="Drag the divider to resize", style="Card.TLabel", foreground="#B298A6").pack(side="right")
        log_body = ttk.Frame(log_frame, style="Card.TFrame")
        log_body.pack(fill="both", expand=True, pady=(6, 0))
        self.log = tk.Text(log_body, height=8, wrap="word", font=("Consolas", 9), relief="flat", background=PALETTE["field"], foreground=PALETTE["ink"], padx=8, pady=6)
        log_scroll = ttk.Scrollbar(log_body, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=log_scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")
        self.log.bind("<MouseWheel>", lambda e: self.log.yview_scroll(-int(e.delta / 120), "units") if e.delta else None)

        workspace.add(notebook_host, stretch="always", minsize=420)
        workspace.add(log_frame, stretch="never", minsize=135)
        self.after(80, lambda: self._set_initial_log_sash(workspace))

    def _set_initial_log_sash(self, workspace: tk.PanedWindow) -> None:
        try:
            workspace.sash_place(0, 0, max(430, self.winfo_height() - 225))
        except tk.TclError:
            pass

    def _build_train_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=0)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        config = ttk.Frame(parent, style="Card.TFrame", padding=16)
        config.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        ttk.Label(config, text="Experiment Composition", style="Card.TLabel", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        r = 1
        self.task_combo = self._row_combo(config, r, "Task", self.task_var, []); r += 1
        self.model_combo = self._row_combo(config, r, "Model", self.model_var, []); r += 1
        self.task_combo.bind("<<ComboboxSelected>>", self._on_task_changed)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_checkpoint_selection_changed)
        self.policy_combo = self._row_combo(config, r, "Action policy", self.policy_var, []); r += 1
        self.reward_combo = self._row_combo(config, r, "Reward function", self.reward_var, []); r += 1
        self._row_combo(config, r, "Device", self.device_var, ["cuda", "cpu", "auto"]); r += 1
        self._row_entry(config, r, "Experiment tag", self.experiment_tag_var); r += 1
        tk.Frame(config, height=1, background=PALETTE["border"]).grid(row=r, column=0, columnspan=2, sticky="ew", pady=10); r += 1
        self._row_entry(config, r, "Timesteps", self.timesteps_var); r += 1
        self._row_entry(config, r, "Parallel envs", self.num_envs_var); r += 1
        self._row_entry(config, r, "Train seed", self.train_seed_var); r += 1
        self.view_combo = self._row_combo(config, r, "Local view", self.view_var, ["5", "7"]); r += 1
        self.train_mask_combo = self._row_combo(config, r, "Action mask", self.mask_var, ["task", "valid"]); r += 1
        ttk.Label(config, text="task: force required interaction · valid: pickup/delivery timing remains a policy decision; select the task-specific valid-interaction reward when studying this setting", style="Card.TLabel", foreground=PALETTE["blue_ink"], wraplength=290).grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 6)); r += 1
        ttk.Label(config, text="Curriculum", style="Card.TLabel").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Checkbutton(config, variable=self.curriculum_var).grid(row=r, column=1, sticky="w"); r += 1

        advanced = ttk.LabelFrame(config, text="PPO research variables", padding=10)
        advanced.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(10, 8)); r += 1
        ar = 0
        for label, var in [
            ("Learning rate", self.lr_var), ("Gamma", self.gamma_var), ("GAE lambda", self.gae_var),
            ("Clip", self.clip_var), ("Entropy", self.entropy_var), ("Rollout", self.rollout_var),
            ("Epochs", self.epochs_var), ("Minibatch", self.minibatch_var), ("Eval every", self.eval_every_var),
        ]:
            self._row_entry(advanced, ar, label, var, width=14); ar += 1

        self._row_readonly(config, r, "Auto checkpoint", self.checkpoint_var, width=34); r += 1
        ttk.Label(
            config,
            text="Derived from Task + model plugin filename.",
            style="Card.TLabel",
            foreground=PALETTE["blue_ink"],
        ).grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 8)); r += 1
        self.train_button = SoftRoundedButton(
            config, "Start Training", self._start_training, width=282, height=42, surface=PALETTE["card"]
        )
        self.train_button.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        dash = ttk.Frame(parent, style="Card.TFrame", padding=18)
        dash.grid(row=0, column=1, sticky="nsew")
        dash.columnconfigure((0, 1, 2, 3), weight=1)
        ttk.Label(dash, text="Training Dashboard", style="Card.TLabel", font=("Segoe UI", 13, "bold")).grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Progressbar(dash, variable=self.progress_var, maximum=1.0).grid(row=1, column=0, columnspan=4, sticky="ew", pady=(12, 16))
        self._metric_card(dash, 2, 0, "Environment steps", self.metric_steps_var)
        self._metric_card(dash, 2, 1, "Train success / 100", self.metric_train_var)
        self._metric_card(dash, 2, 2, "Quick test success", self.metric_test_var)
        self._metric_card(dash, 2, 3, "Path efficiency", self.metric_eff_var)
        self._metric_card(dash, 3, 0, "Latest loss", self.metric_loss_var)
        self.train_chart = MetricChart(dash, height=320)
        self.train_chart.grid(row=4, column=0, columnspan=4, sticky="nsew", pady=(20, 0))
        dash.rowconfigure(4, weight=1)

    def _build_eval_tab(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        card.pack(fill="both", expand=True)
        card.columnconfigure(1, weight=1)
        ttk.Label(card, text="Fixed Map × Seed Evaluation", style="Card.TLabel", font=("Segoe UI", 13, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))
        r = 1
        self.eval_task_combo = self._row_combo(card, r, "Task", self.task_var, []); r += 1
        self.eval_model_combo = self._row_combo(card, r, "Model", self.model_var, []); r += 1
        self.eval_task_combo.bind("<<ComboboxSelected>>", self._on_task_changed)
        self.eval_model_combo.bind("<<ComboboxSelected>>", self._on_checkpoint_selection_changed)
        role_combo = self._row_combo(card, r, "Checkpoint role", self.eval_checkpoint_role_var, ["last", "best"]); r += 1
        role_combo.bind("<<ComboboxSelected>>", self._on_checkpoint_selection_changed)
        self._row_readonly(card, r, "Resolved checkpoint", self.eval_checkpoint_var, width=72); r += 1
        self.eval_policy_combo = self._row_combo(card, r, "Action policy", self.eval_policy_var, []); r += 1
        self.eval_reward_combo = self._row_combo(card, r, "Reward function", self.eval_reward_var, []); r += 1
        self._row_combo(card, r, "Action mask", self.eval_mask_var, ["checkpoint", "task", "valid"]); r += 1
        ttk.Label(card, text="Use checkpoint for protocol-matched evaluation. task forces required interaction on cargo/goal cells; valid allows leaving those cells.", style="Card.TLabel", foreground=PALETTE["blue_ink"], wraplength=900).grid(row=r, column=0, columnspan=3, sticky="w", pady=(0, 6)); r += 1
        self._row_entry(card, r, "Experiment tag", self.experiment_tag_var, width=40); r += 1
        self._row_combo(card, r, "Presentation", self.presentation_var, ["Data only", "Data + display", "Data + display + video"]); r += 1
        self._row_entry(card, r, "Evaluation seeds", self.eval_seed_var, width=72); r += 1
        ttk.Label(
            card,
            text="All seeds are evaluated on every test map. Display/video mode visualizes one representative seed per map while metrics still cover the complete matrix.",
            style="Card.TLabel",
            wraplength=900,
        ).grid(row=r, column=0, columnspan=3, sticky="w", pady=(12, 14)); r += 1
        self.eval_button = SoftRoundedButton(
            card, "Run Evaluation", self._start_evaluation, width=420, height=42, surface=PALETTE["card"]
        )
        self.eval_button.grid(row=r, column=0, columnspan=3, sticky="ew")

    def _build_compare_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)
        top = ttk.Frame(parent, style="Card.TFrame", padding=16)
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, text="Experiment Comparison", style="Card.TLabel", font=("Segoe UI Semibold", 13)).pack(side="left")

        ttk.Label(top, text="Task", style="Card.TLabel").pack(side="left", padx=(22, 6))
        self.compare_task_combo = ttk.Combobox(top, textvariable=self.compare_task_var, state="readonly", width=14)
        self.compare_task_combo.pack(side="left")
        self.compare_task_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_comparison())

        ttk.Label(top, text="Scope", style="Card.TLabel").pack(side="left", padx=(14, 6))
        self.compare_scope_combo = ttk.Combobox(
            top, textvariable=self.compare_scope_var, state="readonly", width=20,
            values=["All records", "Controlled benchmark"],
        )
        self.compare_scope_combo.pack(side="left")
        self.compare_scope_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_comparison())

        SoftRoundedButton(
            top, "Refresh Results", self._refresh_comparison, width=120, height=32,
            fill=PALETTE["pink_soft"], hover="#F5DCE7", foreground="#875F72", surface=PALETTE["card"]
        ).pack(side="right")
        SoftRoundedButton(
            top, "Open Results", lambda: self._open_path(ROOT / "results"), width=110, height=32,
            fill=PALETTE["mint"], hover="#BFE7D8", foreground=PALETTE["mint_ink"], surface=PALETTE["card"]
        ).pack(side="right", padx=(0, 8))
        SoftRoundedButton(
            top, "Delete Selected", self._delete_selected_result, width=122, height=32,
            fill=PALETTE["blue"], hover="#CFE6F5", foreground=PALETTE["blue_ink"], surface=PALETTE["card"]
        ).pack(side="right", padx=(0, 8))
        ttk.Label(
            parent,
            text=(
                "Results are task-scoped. Use All records for exploratory/smoke/ablation runs, or Controlled benchmark for the "
                "full protocol used by the README tables. Each row is one complete evaluation record; inference latency is hardware-specific."
            ),
            wraplength=1080, justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(10, 8))

        table_card = ttk.Frame(parent, style="Card.TFrame", padding=10)
        table_card.grid(row=2, column=0, sticky="nsew")
        table_card.rowconfigure(0, weight=1); table_card.columnconfigure(0, weight=1)
        columns = ("task", "source", "tag", "model", "policy", "pt_reward", "reward", "pt_mask", "mask", "suite", "episodes", "params", "inference_ms", "success", "completion", "pickup", "steps", "return", "efficiency", "collisions", "invalid", "cycles", "interaction_cycles", "revisit")
        self.compare_tree = ttk.Treeview(table_card, columns=columns, show="headings", selectmode="extended")
        headings = {
            "task":"Task", "source":"Source", "tag":"Tag", "model":"Model", "policy":"Policy", "pt_reward":"PT Reward", "reward":"Eval Reward", "pt_mask":"PT Mask", "mask":"Eval Mask", "suite":"Suite", "episodes":"Episodes",
            "params":"Params", "inference_ms":"Infer ms", "success":"Success", "completion":"Completion", "pickup":"Pickup",
            "steps":"Mean steps", "return":"Mean return", "efficiency":"Efficiency", "collisions":"Collisions", "invalid":"Invalid",
            "cycles":"Cycles", "interaction_cycles":"Interact cycles", "revisit":"State revisit"
        }
        widths = {"task":92,"source":86,"tag":125,"model":175,"policy":155,"pt_reward":165,"reward":165,"pt_mask":82,"mask":82,"suite":90,"episodes":72,"params":86,"inference_ms":78,"success":76,"completion":86,"pickup":76,"steps":90,"return":90,"efficiency":80,"collisions":80,"invalid":72,"cycles":72,"interaction_cycles":95,"revisit":88}
        left = {"task", "source", "tag", "model", "policy", "pt_reward", "reward"}
        for key in columns:
            self.compare_tree.heading(key, text=headings[key], command=lambda k=key: self._sort_comparison(k, False))
            self.compare_tree.column(key, width=widths[key], minwidth=65, anchor="w" if key in left else "center")
        vbar = ttk.Scrollbar(table_card, orient="vertical", command=self.compare_tree.yview)
        hbar = ttk.Scrollbar(table_card, orient="horizontal", command=self.compare_tree.xview)
        self.compare_tree.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self.compare_tree.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        self.compare_tree.bind("<Double-1>", self._open_selected_result)
        self.compare_tree.bind("<MouseWheel>", lambda e: self.compare_tree.yview_scroll(-int(e.delta / 120), "units") if e.delta else None)
        self._comparison_paths: dict[str, Path] = {}
        self.after(150, self._refresh_comparison)

    def _comparison_rows(self):
        rows = []
        for run in collect_evaluation_runs(ROOT):
            payload = run.payload
            summary = payload.get("summary", {})
            records = payload.get("records", [])
            first = records[0] if records else {}
            unique_maps = len({r.get("map_name") for r in records if r.get("map_name") is not None})
            unique_seeds = len({r.get("seed") for r in records if r.get("seed") is not None})
            benchmark = payload.get("benchmark", {})
            metadata = payload.get("metadata", {})
            suite = f'{int(benchmark.get("maps", unique_maps))}m×{int(benchmark.get("seeds", unique_seeds))}s'
            rows.append((run.path, {
                "task": run.task_code,
                "source": "benchmark" if run.controlled_benchmark else run.source,
                "controlled": run.controlled_benchmark,
                "tag": str(metadata.get("experiment_tag", "")) or "—",
                "model": str(first.get("model", run.model)),
                "policy": str(first.get("policy", "—")),
                "pt_reward": str(metadata.get("checkpoint_reward_module", "—")),
                "reward": str(first.get("reward", "—")),
                "pt_mask": str(metadata.get("checkpoint_action_mask_mode", "—")),
                "mask": str(metadata.get("action_mask_mode", "—")),
                "suite": suite,
                "episodes": int(summary.get("episodes", 0)),
                "params": int(metadata.get("model_parameters", 0) or 0),
                "inference_ms": float(summary.get("mean_inference_ms", 0.0) or 0.0),
                "success": float(summary.get("success_rate", 0.0) or 0.0),
                "completion": float(summary.get("mean_completion_rate", summary.get("success_rate", 0.0)) or 0.0),
                "pickup": float(summary.get("pickup_rate", 0.0) or 0.0),
                "steps": float(summary.get("mean_steps", 0.0) or 0.0),
                "return": float(summary.get("mean_return", 0.0) or 0.0),
                "efficiency": float(summary.get("mean_path_efficiency", 0.0) or 0.0),
                "collisions": float(summary.get("mean_collisions", 0.0) or 0.0),
                "invalid": float(summary.get("mean_invalid_actions", 0.0) or 0.0),
                "cycles": float(summary.get("mean_cycle_events", 0.0) or 0.0),
                "interaction_cycles": float(summary.get("mean_interaction_cycle_events", 0.0) or 0.0),
                "revisit": float(summary.get("mean_state_revisit_rate", 0.0) or 0.0),
            }))
        return rows

    def _refresh_comparison(self) -> None:
        if not hasattr(self, "compare_tree"):
            return
        tasks = discover_tasks()
        codes = [spec.code for spec in tasks.values()]
        if hasattr(self, "compare_task_combo"):
            self.compare_task_combo.configure(values=["All tasks", *codes])
        if self.compare_task_var.get() not in ["All tasks", *codes]:
            try:
                self.compare_task_var.set(get_task(self.task_var.get()).code)
            except Exception:
                self.compare_task_var.set("All tasks")
        for item in self.compare_tree.get_children():
            self.compare_tree.delete(item)
        self._comparison_paths.clear()
        selected_task = self.compare_task_var.get()
        benchmark_only = self.compare_scope_var.get() == "Controlled benchmark"
        for path, row in self._comparison_rows():
            if selected_task != "All tasks" and row["task"] != selected_task:
                continue
            if benchmark_only and not row["controlled"]:
                continue
            params_text = f'{row["params"] / 1000.0:.1f}k' if row["params"] else "—"
            inference_text = f'{row["inference_ms"]:.3f}' if row["inference_ms"] else "—"
            values = (
                row["task"], row["source"], row["tag"], row["model"], row["policy"], row["pt_reward"], row["reward"], row["pt_mask"], row["mask"], row["suite"], row["episodes"], params_text, inference_text,
                f'{row["success"]:.3f}', f'{row["completion"]:.3f}', f'{row["pickup"]:.3f}', f'{row["steps"]:.1f}', f'{row["return"]:.2f}', f'{row["efficiency"]:.3f}',
                f'{row["collisions"]:.2f}', f'{row["invalid"]:.2f}', f'{row["cycles"]:.2f}', f'{row["interaction_cycles"]:.2f}', f'{row["revisit"]:.3f}'
            )
            iid = self.compare_tree.insert("", "end", values=values)
            self._comparison_paths[iid] = path

    def _sort_comparison(self, key: str, reverse: bool) -> None:
        idx = list(self.compare_tree["columns"]).index(key)
        items = []
        for iid in self.compare_tree.get_children(""):
            value = self.compare_tree.item(iid, "values")[idx]
            try:
                if key == "params" and str(value).lower().endswith("k"):
                    sort_value = float(str(value)[:-1]) * 1000.0
                else:
                    sort_value = float(value)
            except ValueError:
                sort_value = str(value).lower()
            items.append((sort_value, iid))
        items.sort(reverse=reverse, key=lambda x: x[0])
        for pos, (_, iid) in enumerate(items): self.compare_tree.move(iid, "", pos)
        self.compare_tree.heading(key, command=lambda: self._sort_comparison(key, not reverse))

    def _delete_selected_result(self) -> None:
        selection = tuple(iid for iid in self.compare_tree.selection() if iid in self._comparison_paths)
        if not selection:
            messagebox.showinfo("Delete result", "Select one or more comparison rows first.")
            return
        paths = [self._comparison_paths[iid] for iid in selection]
        reference_count = sum((ROOT / "reference_results") in path.parents for path in paths)
        warning = f"\n\n{reference_count} selected row(s) are packaged REFERENCE results." if reference_count else ""
        if not messagebox.askyesno(
            "Delete comparison records",
            f"Delete {len(paths)} selected comparison record(s)?" + warning +
            "\n\nEach paired CSV with the same stem will also be removed when present.",
        ):
            return
        deleted: list[str] = []
        for path in paths:
            for candidate in (path, path.with_suffix(".csv")):
                try:
                    if candidate.exists():
                        candidate.unlink()
                        deleted.append(candidate.name)
                except OSError as exc:
                    messagebox.showerror("Delete result", f"Could not delete {candidate}:\n{exc}")
                    self._refresh_comparison()
                    return
        self._write_log(f"Deleted {len(paths)} comparison row(s): " + ", ".join(deleted))
        rebuild_all_reports(ROOT)
        self._refresh_comparison()

    def _open_selected_result(self, _event=None) -> None:
        selection = self.compare_tree.selection()
        if selection and selection[0] in self._comparison_paths:
            self._open_path(self._comparison_paths[selection[0]])

    def _build_research_tab(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        card.pack(fill="both", expand=True)
        ttk.Label(card, text="Research Extension Points", style="Card.TLabel", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        ttk.Label(
            card,
            text=(
                "TFP uses task-scoped research plugins. Add models and rewards inside the selected task folder; policies remain shared. "
                "The task registry owns environment, maps, view sizes, models, rewards, and result namespace, so new tasks do not require trainer edits."
            ),
            style="Card.TLabel", wraplength=1000, justify="left",
        ).pack(anchor="w", pady=(8, 18))
        buttons = ttk.Frame(card, style="Card.TFrame")
        buttons.pack(anchor="w")
        for label, path in [
            ("Task Models", None),
            ("Policies", ROOT / "tfp" / "policies"),
            ("Task Rewards", None),
            ("Named Tasks", ROOT / "tfp" / "tasks"),
            ("Launchers", ROOT / "launchers"),
            ("Runtime Config", ROOT / "config"),
            ("Experiment Checklist", ROOT / "docs" / "EXPERIMENT_CHECKLIST.md"),
            ("Model Plugin Guide", ROOT / "docs" / "MODEL_PLUGIN_GUIDE.md"),
        ]:
            if label == "Task Models":
                command = lambda: self._open_path(get_task(self.task_var.get()).train_map_dir.parent.parent / "models")
            elif label == "Task Rewards":
                command = lambda: self._open_path(get_task(self.task_var.get()).train_map_dir.parent.parent / "rewards")
            else:
                command = lambda p=path: self._open_path(p)
            ttk.Button(buttons, text=label, command=command).pack(side="left", padx=(0, 8))
        SoftRoundedButton(
            card, "Refresh Plugins", self._refresh_plugins, width=130, height=34,
            fill=PALETTE["blue"], hover="#CFE6F5", foreground=PALETTE["blue_ink"], surface=PALETTE["card"]
        ).pack(anchor="w", pady=(16, 10))
        text_body = ttk.Frame(card, style="Card.TFrame")
        text_body.pack(fill="both", expand=True)
        self.plugin_text = tk.Text(text_body, height=18, font=("Consolas", 9), relief="flat", background=PALETTE["field"], foreground=PALETTE["ink"])
        plugin_scroll = ttk.Scrollbar(text_body, orient="vertical", command=self.plugin_text.yview)
        self.plugin_text.configure(yscrollcommand=plugin_scroll.set)
        self.plugin_text.pack(side="left", fill="both", expand=True)
        plugin_scroll.pack(side="right", fill="y")

    def _metric_card(self, parent, row: int, col: int, name: str, variable: tk.StringVar) -> None:
        fills = (PALETTE["pink_soft"], PALETTE["mint_soft"], PALETTE["blue_soft"])
        card = RoundedMetricCard(parent, name, variable, fills[(row + col) % len(fills)])
        card.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)

    def _row_entry(self, parent, row: int, label: str, variable: tk.StringVar, width: int = 24) -> None:
        label_style = "Card.TLabel" if isinstance(parent, ttk.Frame) else "TLabel"
        ttk.Label(parent, text=label, style=label_style).grid(row=row, column=0, sticky="w", pady=4, padx=(0, 14))
        ttk.Entry(parent, textvariable=variable, width=width).grid(row=row, column=1, sticky="ew", pady=4)

    def _row_combo(self, parent, row: int, label: str, variable: tk.StringVar, values, width: int = 28):
        ttk.Label(parent, text=label, style="Card.TLabel").grid(row=row, column=0, sticky="w", pady=4, padx=(0, 14))
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=width)
        combo.grid(row=row, column=1, sticky="ew", pady=4)
        return combo

    def _row_readonly(self, parent, row: int, label: str, variable: tk.StringVar, width: int = 24) -> None:
        ttk.Label(parent, text=label, style="Card.TLabel").grid(row=row, column=0, sticky="w", pady=4, padx=(0, 14))
        ttk.Entry(parent, textvariable=variable, width=width, state="readonly").grid(row=row, column=1, sticky="ew", pady=4)

    def _on_checkpoint_selection_changed(self, _event=None) -> None:
        self._sync_checkpoint_paths()

    def _on_task_changed(self, _event=None) -> None:
        task = get_task(self.task_var.get())
        self.view_var.set(str(task.default_view_size))
        self.eval_seed_var.set(",".join(str(v) for v in task.default_eval_seeds))
        self._refresh_plugins()
        self.compare_task_var.set(task.code)
        self._refresh_comparison()

    def _sync_checkpoint_paths(self) -> None:
        try:
            final_path = checkpoint_path(self.task_var.get(), self.model_var.get())
            eval_path = resolve_checkpoint_path(
                self.task_var.get(), self.model_var.get(), role=self.eval_checkpoint_role_var.get()
            )
            self.checkpoint_var.set(str(final_path))
            self.eval_checkpoint_var.set(str(eval_path))
        except Exception:
            self.checkpoint_var.set("Auto")
            self.eval_checkpoint_var.set("Auto")

    def _refresh_plugins(self) -> None:
        tasks = discover_tasks()
        if self.task_var.get() not in tasks and tasks:
            self.task_var.set(next(iter(tasks)))
        task_id = self.task_var.get()
        task = tasks[task_id]
        models = discover_model_plugins(task_id)
        policies = discover_policy_plugins()
        rewards = discover_reward_plugins(task_id)
        if hasattr(self, "task_combo"):
            self.task_combo.configure(values=list(tasks.keys()))
            self.model_combo.configure(values=list(models.keys()))
            self.policy_combo.configure(values=list(policies.keys()))
            self.reward_combo.configure(values=list(rewards.keys()))
            self.eval_task_combo.configure(values=list(tasks.keys()))
            self.eval_model_combo.configure(values=list(models.keys()))
            self.eval_policy_combo.configure(values=["checkpoint", *list(policies.keys())])
            self.eval_reward_combo.configure(values=["checkpoint", *list(rewards.keys())])
        if models and self.model_var.get() not in models:
            self.model_var.set(task.default_model if task.default_model in models else next(iter(models)))
        if policies and self.policy_var.get() not in policies:
            self.policy_var.set(next(iter(policies)))
        if rewards and self.reward_var.get() not in rewards:
            self.reward_var.set(task.default_reward if task.default_reward in rewards else next(iter(rewards)))
        self._sync_checkpoint_paths()
        if hasattr(self, "plugin_text"):
            self.plugin_text.delete("1.0", "end")
            self.plugin_text.insert("end", "NAMED TASKS\n")
            for task_id, spec in tasks.items():
                self.plugin_text.insert("end", f"  {task_id}  [{spec.code}]  {spec.display_name}\n")
            self.plugin_text.insert("end", "\nMODELS\n")
            for key, spec in models.items():
                self.plugin_text.insert("end", f"  {key}: {spec.display_name} — {spec.description}\n")
            self.plugin_text.insert("end", "\nPOLICIES\n")
            for key, spec in policies.items():
                self.plugin_text.insert("end", f"  {key}: {spec.display_name} — {spec.description}\n")
            self.plugin_text.insert("end", "\nREWARDS\n")
            for key, spec in rewards.items():
                self.plugin_text.insert("end", f"  {key}: {spec.display_name} — {spec.description}\n")
        self._write_log(
            f"Plugins refreshed: tasks={len(tasks)} models={len(models)} policies={len(policies)} rewards={len(rewards)}"
        )

    def _update_runtime_card(self) -> None:
        status = runtime_status()
        env_ok = status.current_conda_env == status.expected_conda_env
        cuda_text = f"CUDA {status.cuda_version} · {status.gpu_name}" if status.cuda_available else "CUDA unavailable"
        marker = "READY" if env_ok else "CHECK"
        self.runtime_var.set(
            f"{marker}  Conda expected: {status.expected_conda_env} · current: {status.current_conda_env} · "
            f"PyTorch {status.torch_version} · {cuda_text}"
        )

    def _config_payload(self) -> dict[str, object]:
        return {
            "task": self.task_var.get(), "model": self.model_var.get(), "policy": self.policy_var.get(), "reward": self.reward_var.get(),
            "device": self.device_var.get(), "timesteps": self.timesteps_var.get(), "num_envs": self.num_envs_var.get(), "seed": self.train_seed_var.get(),
            "observation": "local", "view_size": self.view_var.get(), "action_mask": self.mask_var.get(), "curriculum": self.curriculum_var.get(),
            "rollout": self.rollout_var.get(), "epochs": self.epochs_var.get(), "minibatch": self.minibatch_var.get(), "gamma": self.gamma_var.get(),
            "gae_lambda": self.gae_var.get(), "clip": self.clip_var.get(), "lr": self.lr_var.get(), "entropy": self.entropy_var.get(),
            "eval_every": self.eval_every_var.get(), "checkpoint": self.checkpoint_var.get(), "eval_seeds": self.eval_seed_var.get(),
            "experiment_tag": self.experiment_tag_var.get(), "eval_checkpoint_role": self.eval_checkpoint_role_var.get(),
        }

    def _save_config(self) -> None:
        task_dir = ROOT / "experiments" / get_task(self.task_var.get()).code
        task_dir.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("TFP experiment config", "*.json")], initialdir=str(task_dir))
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(self._config_payload(), indent=2), encoding="utf-8")
            self._write_log(f"Saved experiment config: {path}")

    def _load_config(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("TFP experiment config", "*.json"), ("JSON", "*.json")])
        if not path:
            return
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        mapping = {
            "task": self.task_var, "model": self.model_var, "policy": self.policy_var, "reward": self.reward_var, "device": self.device_var,
            "timesteps": self.timesteps_var, "num_envs": self.num_envs_var, "seed": self.train_seed_var,
            "view_size": self.view_var, "action_mask": self.mask_var, "rollout": self.rollout_var, "epochs": self.epochs_var,
            "minibatch": self.minibatch_var, "gamma": self.gamma_var, "gae_lambda": self.gae_var, "clip": self.clip_var, "lr": self.lr_var,
            "entropy": self.entropy_var, "eval_every": self.eval_every_var, "eval_seeds": self.eval_seed_var, "experiment_tag": self.experiment_tag_var,
            "eval_checkpoint_role": self.eval_checkpoint_role_var,
        }
        for key, variable in mapping.items():
            if key in payload:
                variable.set(str(payload[key]))
        self.obs_var.set("local")
        if self.mask_var.get() == "none":
            self.mask_var.set("task")
            self._write_log("Migrated legacy action_mask=none to task (unmasked mode is not supported).")
        if "curriculum" in payload:
            self.curriculum_var.set(bool(payload["curriculum"]))
        self._refresh_plugins()
        self._sync_checkpoint_paths()
        self._write_log(f"Loaded experiment config: {path}")

    def _build_ppo_config(self) -> PPOConfig:
        return PPOConfig(
            timesteps=int(self.timesteps_var.get()), num_envs=int(self.num_envs_var.get()), rollout=int(self.rollout_var.get()),
            epochs=int(self.epochs_var.get()), minibatch=int(self.minibatch_var.get()), gamma=float(self.gamma_var.get()),
            gae_lambda=float(self.gae_var.get()), clip=float(self.clip_var.get()), lr=float(self.lr_var.get()), entropy=float(self.entropy_var.get()),
            eval_every=int(self.eval_every_var.get()), seed=int(self.train_seed_var.get()), task_id=self.task_var.get(),
            observation_mode="local", view_size=int(self.view_var.get()), action_mask_mode=self.mask_var.get(),
            curriculum=bool(self.curriculum_var.get()), reward_module=self.reward_var.get(), policy_module=self.policy_var.get(), device=self.device_var.get(),
            experiment_tag=self.experiment_tag_var.get().strip(),
        )

    def _start_training(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("TFP Studio", "An experiment is already running.")
            return
        try:
            config = self._build_ppo_config()
            task = get_task(config.task_id)
            # Fail early when CUDA was explicitly requested.
            resolve_device(config.device)
        except Exception as exc:
            messagebox.showerror("Invalid configuration", str(exc))
            return

        selected_model = self.model_var.get()
        checkpoint = checkpoint_path(config.task_id, selected_model)
        self._sync_checkpoint_paths()
        train_maps = discover_maps(task.train_map_dir)
        test_maps = discover_maps(task.test_map_dir)
        self._write_log(
            f"Starting {task.env_id}: model={selected_model} policy={config.policy_module} reward={config.reward_module} "
            f"mask={config.action_mask_mode} maps={len(train_maps)} device={config.device}"
        )
        self.progress_var.set(0.0)
        self.metric_steps_var.set("0")
        self.metric_train_var.set("0.000")
        self.metric_test_var.set("—")
        self.metric_eff_var.set("—")
        self.metric_loss_var.set("—")
        self.train_chart.reset()
        self._set_buttons(False)

        def work() -> None:
            try:
                train_ppo(
                    selected_model, train_maps, test_maps, checkpoint, config,
                    log_callback=self.log_queue.put,
                    progress_callback=self.log_queue.put,
                )
            except Exception as exc:
                self.log_queue.put(f"ERROR: {type(exc).__name__}: {exc}")
            finally:
                self.log_queue.put("__WORKER_DONE__")

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _start_evaluation(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("TFP Studio", "An experiment is already running.")
            return
        checkpoint = resolve_checkpoint_path(
            self.task_var.get(), self.model_var.get(), role=self.eval_checkpoint_role_var.get()
        )
        self._sync_checkpoint_paths()
        if not checkpoint.exists():
            messagebox.showerror(
                "Missing checkpoint",
                f"No checkpoint exists yet for model '{self.model_var.get()}'.\n\n"
                f"Expected:\n{checkpoint}\n\n"
                "Train this model first, or place its compatible .pt file at the expected path.",
            )
            return
        try:
            seeds = tuple(int(v.strip()) for v in self.eval_seed_var.get().split(",") if v.strip())
            if not seeds:
                raise ValueError("At least one seed is required.")
            device = resolve_device(self.device_var.get())
        except Exception as exc:
            messagebox.showerror("Invalid evaluation configuration", str(exc))
            return

        presentation_map = {"Data only": "data", "Data + display": "display", "Data + display + video": "video"}
        presentation = presentation_map[self.presentation_var.get()]
        self._set_buttons(False)

        def work() -> None:
            try:
                model, ckpt = load_checkpoint_model(checkpoint, device)
                loaded_model = normalize_model_name(str(ckpt.get("model_module", "unknown")))
                if loaded_model != normalize_model_name(self.model_var.get()):
                    raise ValueError(
                        f"Checkpoint/model mismatch: selected '{self.model_var.get()}' but checkpoint contains '{loaded_model}'."
                    )
                task_id = str(ckpt.get("task_id", self.task_var.get()))
                if task_id != self.task_var.get():
                    raise ValueError(
                        f"Checkpoint/task mismatch: selected '{self.task_var.get()}' but checkpoint contains '{task_id}'."
                    )
                task = get_task(task_id)
                test_maps = discover_maps(task.test_map_dir)
                reward_module = (
                    str(ckpt.get("reward_module", task.default_reward))
                    if self.eval_reward_var.get() == "checkpoint" else self.eval_reward_var.get()
                )
                policy_module = (
                    str(ckpt.get("policy_module", "001_ppo_categorical"))
                    if self.eval_policy_var.get() == "checkpoint" else self.eval_policy_var.get()
                )
                action_mask_mode = (
                    str(ckpt.get("action_mask_mode", "task"))
                    if self.eval_mask_var.get() == "checkpoint" else self.eval_mask_var.get()
                )
                env = create_task_env(
                    task_id, test_maps,
                    observation_mode=str(ckpt.get("observation_mode", "local")),
                    view_size=int(ckpt.get("view_size", task.default_view_size)),
                    reward_module=reward_module,
                )
                self.log_queue.put(
                    f"Starting evaluation: {task.env_id} · {len(test_maps)} maps x {len(seeds)} seeds = {len(test_maps) * len(seeds)} episodes · "
                    f"checkpoint={checkpoint.name} role={ckpt.get('checkpoint_role', 'legacy')} · "
                    f"policy={policy_module} reward={reward_module} mask={action_mask_mode}"
                )
                summary, _, paths = evaluate_policy(
                    model, env, test_maps, device, seeds=seeds, action_mask_mode=action_mask_mode,
                    policy_module=policy_module, model_name=loaded_model, presentation=presentation,
                    output_dir=ROOT / "results", video_dir=ROOT / "videos", progress_callback=self.log_queue.put,
                    run_metadata={
                        "experiment_tag": self.experiment_tag_var.get().strip(),
                        "checkpoint": str(checkpoint),
                        "checkpoint_role": str(ckpt.get("checkpoint_role", "legacy")),
                        "requested_checkpoint_role": self.eval_checkpoint_role_var.get(),
                        "checkpoint_size_mb": round(checkpoint.stat().st_size / (1024 * 1024), 4),
                        "training_seed": ckpt.get("seed"),
                        "checkpoint_reward_module": str(ckpt.get("reward_module", "unknown")),
                        "checkpoint_policy_module": str(ckpt.get("policy_module", "unknown")),
                        "checkpoint_action_mask_mode": str(ckpt.get("action_mask_mode", "unknown")),
                        "model_parameters": sum(p.numel() for p in model.parameters()),
                        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    },
                )
                self.log_queue.put(
                    f"SUMMARY success={summary.success_rate:.3f} mean_steps={summary.mean_steps:.1f} "
                    f"efficiency={summary.mean_path_efficiency:.3f} collisions={summary.mean_collisions:.2f} "
                    f"cycles={summary.mean_cycle_events:.2f} interaction_cycles={summary.mean_interaction_cycle_events:.2f} "
                    f"revisit={summary.mean_state_revisit_rate:.3f} inference={summary.mean_inference_ms:.3f}ms"
                )
                self.log_queue.put(f"Saved metrics: {paths[0]} | {paths[1]}")
                self.log_queue.put({"kind": "refresh_comparison"})
            except Exception as exc:
                self.log_queue.put(f"ERROR: {type(exc).__name__}: {exc}")
            finally:
                self.log_queue.put("__WORKER_DONE__")

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _set_buttons(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.train_button.configure(state=state)
        self.eval_button.configure(state=state)
        self.load_config_button.configure(state=state)
        # Protocol-critical selectors are locked while a worker is active.  The
        # training thread receives an immutable PPOConfig snapshot, and locking the
        # widgets prevents the UI from visually suggesting that a running task/valid
        # protocol changed midway through training.
        combo_state = "readonly" if enabled else "disabled"
        for name in (
            "task_combo", "model_combo", "policy_combo", "reward_combo",
            "train_mask_combo", "eval_task_combo", "eval_model_combo",
            "eval_policy_combo", "eval_reward_combo",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.configure(state=combo_state)

    def _drain_log_queue(self) -> None:
        try:
            while True:
                message = self.log_queue.get_nowait()
                if message == "__WORKER_DONE__":
                    self._set_buttons(True)
                elif isinstance(message, dict):
                    self._handle_progress(message)
                else:
                    self._write_log(str(message))
        except queue.Empty:
            pass
        self.after(100, self._drain_log_queue)

    def _handle_progress(self, event: dict[str, object]) -> None:
        if event.get("kind") == "refresh_comparison":
            self._refresh_comparison()
            return
        progress = float(event.get("progress", 0.0))
        self.progress_var.set(progress)
        self.metric_steps_var.set(str(int(event.get("steps", 0))))
        self.metric_train_var.set(f"{float(event.get('train_success100', 0.0)):.3f}")
        self.metric_loss_var.set(f"{float(event.get('loss', 0.0)):.4f}")
        if event.get("kind") == "evaluation":
            test = float(event.get("quick_test_success", 0.0))
            eff = float(event.get("efficiency", 0.0))
            self.metric_test_var.set(f"{test:.3f}")
            self.metric_eff_var.set(f"{eff:.3f}")
            self.train_chart.add_point(progress, test)

    def _write_log(self, message: str) -> None:
        if not hasattr(self, "log"):
            return
        self.log.insert("end", message + "\n")
        self.log.see("end")

    def _open_path(self, path: Path) -> None:
        path = path.resolve()
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("Open path", f"Could not open:\n{path}\n\n{exc}")


if __name__ == "__main__":
    app = TFPStudio()
    app.mainloop()
