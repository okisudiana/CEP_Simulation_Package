# =============================================================================
# Project     : CEP Rocket Simulation
# File        : CEP_GUI_V2.py
# Author      : Oka Sudiana
# Created     : 2026-10-06
# Updated     : 2026-10-06
# Version     : 1.0
# =============================================================================
#
# Description:
#     6-DOF rocket flight dynamics and Circular Error Probable main GUI.
#
# =============================================================================

# -*- coding: utf-8 -*-
"""
cep_gui.py
Desktop GUI (Tkinter) for the refactored CEP / 6-DOF simulation engine
(core_sim.py).

Run with:  python cep_gui.py
Requires:  numpy, scipy, matplotlib  (pip install numpy scipy matplotlib)

IMPORTANT ASSUMPTIONS (see chat message for full list):
- data_aero .mat must contain: ALPHA, MACH, CA, CN, CYB, CLLB, CLNB, CLLP,
  CMQ, CLNR, XCP  (same schema as the original RX200 dataset).
- data_thrust .mat must contain: RTX200TCsimTh [t, thrust_kgf],
  massflow [t, mass_burned_kg].
- Burn_time is not a user input: it is read directly from data_thrust's
  own time axis (last timestamp of the thrust curve).
- The aerodynamic Moment Reference Center is assumed equal to Xcg_wet.
- Monte Carlo runs execute in a background thread (not multiprocessing) so
  the dispersion plot can update live; this is slower than a fully
  parallel batch run for large N.
"""

import threading
import queue
import traceback

import matplotlib
try:
    matplotlib.use('TkAgg', force=True)
except ImportError as e:
    raise ImportError(
        "Could not load the TkAgg backend. This usually means the Python "
        "process already has a different GUI framework running (e.g. Qt) "
        "-- typically because the script is being run inside Spyder or "
        "Jupyter with a Qt graphics backend enabled. Fix: run this script "
        "from a plain system terminal ('python cep_gui.py'), or in Spyder "
        "set Tools > Preferences > IPython console > Graphics > Backend "
        "to 'Inline' and restart the kernel, then run again.\n"
        f"Original error: {e}"
    )

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core_sim import Geometry, AeroDB, ThrustDB, Perturb, run_single_simulation, \
    generate_samples, run_sensitivity_analysis

import os

# ==============================================================================
# Visual identity
# ==============================================================================
# Institutional colors (BRIN corporate palette: maroon/red primary, dark navy
# secondary). Adjust freely if your organization has an official style guide.
THEME = {
    "primary": "#8B1E2B",       # BRIN maroon/red
    "primary_dark": "#5E1119",
    "accent": "#C9A227",        # gold accent line
    "bg": "#F4F1EC",            # warm off-white app background
    "panel": "#FFFFFF",
    "text_on_primary": "#FFFFFF",
    "text": "#2B2B2B",
    "muted": "#6b6b6b",
    "ok": "#1E7B3A",
    "err": "#B3261E",
    "warn": "#B3261E",
}

# Logo / banner background image lookup: place files with these names next
# to cep_gui.py. Not bundled here -- an official institutional logo cannot
# be fabricated by this script; drop your own file in and it is picked up
# automatically. If absent, a text placeholder is drawn instead so the
# layout never breaks.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_CANDIDATES = ["brin_logo.png", "logo.png", "logo_brin.png"]
HEADER_BG_CANDIDATES = ["header_bg.png", "header_bg.jpg", "background.png", "background.jpg"]


def _find_asset(candidates):
    for name in candidates:
        p = os.path.join(_SCRIPT_DIR, name)
        if os.path.isfile(p):
            return p
    return None


class ToolTip:
    """Small hover tooltip for widgets that benefit from extra explanation."""
    def __init__(self, widget, text, delay=450):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tip = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def _schedule(self, _event=None):
        self._after_id = self.widget.after(self.delay, self._show)

    def _show(self):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(self.tip, text=self.text, justify="left", background="#FFFDE7",
                        relief="solid", borderwidth=1, font=("", 9), wraplength=280,
                        padx=6, pady=4)
        lbl.pack()

    def _hide(self, _event=None):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        if self.tip:
            self.tip.destroy()
            self.tip = None


# Short explanations for fields whose meaning isn't obvious from the label
# alone -- shown as hover tooltips in the Inputs tab.
FIELD_HELP = {
    "Xthrust": "Titik aplikasi gaya dorong (nozzle) diukur dari hidung "
               "roket, terpisah dari panjang total L.",
    "Xcg_wet": "Titik berat saat propelan penuh (kondisi awal peluncuran).",
    "Xcg_dry": "Titik berat setelah propelan habis terbakar (burnout).",
    "theta0_deg": "Sudut elevasi rel peluncur diukur dari bidang horizontal.",
    "psi0_deg": "Sudut azimuth rel peluncur diukur dari utara/referensi.",
    "L_rail": "Panjang rel peluncur; wahana dianggap 'lepas rel' setelah "
              "jarak tempuh sepanjang rel ini terlampaui.",
}


GEOM_FIELDS = [
    ("D", "Diameter D (m)", 0.203),
    ("L", "Length L (m)", 4.150),
    ("mass_wet", "Mass wet (kg)", 197.03),
    ("mass_dry", "Mass dry (kg)", 123.35),
    ("Ixx_wet", "Ixx wet (kg.m2)", 1.252),
    ("Iyy_wet", "Iyy wet (kg.m2)", 201.264),
    ("Izz_wet", "Izz wet (kg.m2)", 201.268),
    ("Ixx_dry", "Ixx dry (kg.m2)", 0.971),
    ("Iyy_dry", "Iyy dry (kg.m2)", 153.906),
    ("Izz_dry", "Izz dry (kg.m2)", 153.909),
    ("Xcg_wet", "Xcg wet (m, from nose)", 2.401),
    ("Xcg_dry", "Xcg dry (m, from nose)", 2.16123),
    ("Xthrust", "Xthrust (m, from nose)", 4.150),
]

RAIL_FIELDS = [
    ("theta0_deg", "Rail elevation theta0 (deg)", 70.0),
    ("psi0_deg", "Rail azimuth psi0 (deg)", 0.0),
    ("L_rail", "Rail length L_rail (m)", 6.0),
]


class ScrollableFrame(ttk.Frame):
    """A frame with a vertical scrollbar for long input forms."""
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


class CEPApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CEP / 6-DOF Trajectory Simulation - BRIN")
        self.geometry("1280x840")
        self.configure(bg=THEME["bg"])

        self.aero_path = tk.StringVar(value="")
        self.thrust_path = tk.StringVar(value="")
        self.aero_db = None
        self.thrust_db = None

        self.geom_vars = {}
        self.rail_vars = {}

        self.mc_queue = queue.Queue()
        self.mc_thread = None
        self.mc_stop_flag = threading.Event()
        self.mc_results = {}  # method -> list of (x_rel, y_rel)

        self._logo_img = None       # keep references alive (tk garbage collects otherwise)
        self._header_bg_img = None

        self._setup_style()
        self._build_ui()

    # ------------------------------------------------------------------
    def _setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=THEME["bg"], foreground=THEME["text"],
                         font=("Segoe UI", 10))
        style.configure("TFrame", background=THEME["bg"])
        style.configure("TLabel", background=THEME["bg"], foreground=THEME["text"])
        style.configure("TLabelframe", background=THEME["bg"], foreground=THEME["primary"],
                         bordercolor=THEME["primary"])
        style.configure("TLabelframe.Label", background=THEME["bg"],
                         foreground=THEME["primary"], font=("Segoe UI", 10, "bold"))
        style.configure("TNotebook", background=THEME["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8), font=("Segoe UI", 10, "bold"),
                         background="#E4DED2", foreground=THEME["text"])
        style.map("TNotebook.Tab",
                  background=[("selected", THEME["primary"])],
                  foreground=[("selected", THEME["text_on_primary"])])
        style.configure("Accent.TButton", background=THEME["primary"],
                         foreground=THEME["text_on_primary"], font=("Segoe UI", 10, "bold"),
                         padding=(10, 6))
        style.map("Accent.TButton",
                  background=[("active", THEME["primary_dark"]), ("disabled", "#B7A9A9")])
        style.configure("TButton", padding=(8, 5))
        style.configure("Horizontal.TProgressbar", background=THEME["primary"],
                         troughcolor="#E4DED2")

    # ------------------------------------------------------------------
    def _build_ui(self):
        self._build_header()

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=(6, 0))

        self.tab_inputs = ttk.Frame(nb)
        self.tab_single = ttk.Frame(nb)
        self.tab_mc = ttk.Frame(nb)
        self.tab_sens = ttk.Frame(nb)

        nb.add(self.tab_inputs, text="1. Inputs")
        nb.add(self.tab_single, text="2. Single Trajectory")
        nb.add(self.tab_mc, text="3. Monte Carlo / CEP")
        nb.add(self.tab_sens, text="4. Sensitivity")

        self._build_inputs_tab()
        self._build_single_tab()
        self._build_mc_tab()
        self._build_sens_tab()

        self._build_footer()

    def _build_header(self):
        """Banner with logo + title. Uses brin_logo.png / header_bg.* next to
        this script if present (see LOGO_CANDIDATES / HEADER_BG_CANDIDATES);
        otherwise falls back to a plain color banner with a text placeholder
        so the layout still looks intentional."""
        banner_h = 84
        canvas = tk.Canvas(self, height=banner_h, highlightthickness=0, bd=0)
        canvas.pack(fill="x", side="top")

        def draw(_event=None):
            canvas.delete("all")
            w = max(canvas.winfo_width(), 1)

            bg_path = _find_asset(HEADER_BG_CANDIDATES)
            drew_bg_image = False
            if bg_path:
                try:
                    img = tk.PhotoImage(file=bg_path)
                    # stretch-ish: tile/zoom crudely to cover width (PhotoImage
                    # has no true scale-to-fit without PIL; zoom by integer
                    # factor gets us close enough for a banner strip)
                    if img.width() > 0:
                        factor = max(1, round(w / img.width()))
                        img = img.zoom(factor, 1)
                    self._header_bg_img = img
                    canvas.create_image(0, 0, image=img, anchor="nw")
                    drew_bg_image = True
                except tk.TclError:
                    drew_bg_image = False

            if not drew_bg_image:
                canvas.create_rectangle(0, 0, w, banner_h, fill=THEME["primary"], width=0)
                canvas.create_rectangle(0, banner_h - 4, w, banner_h, fill=THEME["accent"],
                                         width=0)

            logo_path = _find_asset(LOGO_CANDIDATES)
            text_x = 20
            if logo_path:
                try:
                    logo = tk.PhotoImage(file=logo_path)
                    target_h = banner_h - 20
                    if logo.height() > target_h:
                        factor = max(1, round(logo.height() / target_h))
                        logo = logo.subsample(factor, factor)
                    self._logo_img = logo
                    canvas.create_image(20, banner_h // 2, image=logo, anchor="w")
                    text_x = 20 + logo.width() + 16
                except tk.TclError:
                    self._logo_img = None
            if not logo_path:
                # Placeholder so absence of a real logo file doesn't look broken
                canvas.create_oval(18, 14, 18 + (banner_h - 28), 14 + (banner_h - 28),
                                    fill=THEME["text_on_primary"], outline="")
                canvas.create_text(18 + (banner_h - 28) / 2, banner_h / 2, text="BRIN",
                                    fill=THEME["primary"], font=("Segoe UI", 10, "bold"))
                text_x = 18 + (banner_h - 28) + 16

            canvas.create_text(text_x, banner_h / 2 - 12, anchor="w",
                                text="CEP / 6-DOF Trajectory Simulation",
                                fill=THEME["text_on_primary"], font=("Segoe UI", 15, "bold"))
            canvas.create_text(text_x, banner_h / 2 + 12, anchor="w",
                                text="Badan Riset dan Inovasi Nasional (BRIN)",
                                fill="#EFE3C8", font=("Segoe UI", 10))

        canvas.bind("<Configure>", draw)
        self._header_canvas = canvas
        if not _find_asset(LOGO_CANDIDATES):
            self._note_missing_logo = True

    def _build_footer(self):
        footer = tk.Frame(self, bg=THEME["primary_dark"], height=24)
        footer.pack(fill="x", side="bottom")
        tk.Label(footer, text="CEP Simulation Tool  |  Flight Dynamics & GNC",
                 bg=THEME["primary_dark"], fg="#EFE3C8", font=("Segoe UI", 8)).pack(
            side="left", padx=10, pady=3)
        if not _find_asset(LOGO_CANDIDATES):
            tk.Label(footer, text="(logo BRIN belum ditemukan - taruh 'brin_logo.png' di folder ini)",
                     bg=THEME["primary_dark"], fg="#D8B9B9", font=("Segoe UI", 8, "italic")).pack(
                side="right", padx=10, pady=3)

    def _badge(self, parent, text, kind="muted"):
        """Small colored status pill (green=ok, red=error, gray=neutral)."""
        colors = {"ok": (THEME["ok"], "#E4F3E8"), "err": (THEME["err"], "#FBE7E5"),
                  "muted": (THEME["muted"], "#EEEAE2")}
        fg, bg = colors.get(kind, colors["muted"])
        return tk.Label(parent, text=text, fg=fg, bg=bg, font=("Segoe UI", 9, "bold"),
                         padx=8, pady=3)

    def _section(self, parent, title):
        """Consistent LabelFrame wrapper for grouping related inputs."""
        lf = ttk.LabelFrame(parent, text=title)
        return lf

    # ------------------------------------------------------------------
    def _build_inputs_tab(self):
        outer = ScrollableFrame(self.tab_inputs)
        outer.pack(fill="both", expand=True)
        f = outer.scrollable_frame

        geom_lf = self._section(f, "Geometri & Massa (Geometry & Mass Properties)")
        geom_lf.pack(fill="x", padx=10, pady=(10, 6))
        for i, (key, label, default) in enumerate(GEOM_FIELDS):
            r, c = divmod(i, 2)
            cell = ttk.Frame(geom_lf)
            cell.grid(row=r, column=c, sticky="w", padx=10, pady=4)
            lbl = ttk.Label(cell, text=label, width=24, anchor="w")
            lbl.pack(side="left")
            var = tk.StringVar(value=str(default))
            entry = ttk.Entry(cell, textvariable=var, width=14)
            entry.pack(side="left", padx=(4, 0))
            self.geom_vars[key] = var
            if key in FIELD_HELP:
                ToolTip(lbl, FIELD_HELP[key])
                ToolTip(entry, FIELD_HELP[key])

        rail_lf = self._section(f, "Parameter Rel Peluncur (Launch Rail)")
        rail_lf.pack(fill="x", padx=10, pady=6)
        for i, (key, label, default) in enumerate(RAIL_FIELDS):
            cell = ttk.Frame(rail_lf)
            cell.grid(row=0, column=i, sticky="w", padx=10, pady=6)
            lbl = ttk.Label(cell, text=label, width=24, anchor="w")
            lbl.pack(side="left")
            var = tk.StringVar(value=str(default))
            entry = ttk.Entry(cell, textvariable=var, width=14)
            entry.pack(side="left", padx=(4, 0))
            self.rail_vars[key] = var
            if key in FIELD_HELP:
                ToolTip(lbl, FIELD_HELP[key])
                ToolTip(entry, FIELD_HELP[key])

        data_lf = self._section(f, "Data Aerodinamika & Thrust (.mat)")
        data_lf.pack(fill="x", padx=10, pady=6)

        row0 = ttk.Frame(data_lf)
        row0.grid(row=0, column=0, sticky="w", padx=10, pady=6)
        ttk.Label(row0, text="data_aero (.mat)", width=18, anchor="w").pack(side="left")
        ttk.Entry(row0, textvariable=self.aero_path, width=42, state="readonly").pack(
            side="left", padx=6)
        ttk.Button(row0, text="Browse...", command=self._browse_aero).pack(side="left")

        row1 = ttk.Frame(data_lf)
        row1.grid(row=1, column=0, sticky="w", padx=10, pady=6)
        ttk.Label(row1, text="data_thrust (.mat)", width=18, anchor="w").pack(side="left")
        ttk.Entry(row1, textvariable=self.thrust_path, width=42, state="readonly").pack(
            side="left", padx=6)
        ttk.Button(row1, text="Browse...", command=self._browse_thrust).pack(side="left")

        row2 = ttk.Frame(data_lf)
        row2.grid(row=2, column=0, sticky="w", padx=10, pady=(2, 8))
        ttk.Label(row2, text="Status:", width=18, anchor="w").pack(side="left")
        self.load_status = self._badge(row2, "Belum ada data dimuat", "muted")
        self.load_status.pack(side="left")

        note = ("Catatan: data_aero harus berisi variabel ALPHA, MACH, CA, CN, CYB, CLLB,\n"
                "CLNB, CLLP, CMQ, CLNR, XCP. data_thrust harus berisi 'RTX200TCsimTh'\n"
                "[t, thrust_kgf] dan 'massflow' [t, mass_burned_kg] (skema sama seperti\n"
                "dataset RX200 asli). Burn_time otomatis diambil dari sumbu waktu\n"
                "data_thrust (bukan input manual). MRC aero database diasumsikan = Xcg_wet.")
        ttk.Label(f, text=note, foreground=THEME["muted"], justify="left").pack(
            anchor="w", padx=16, pady=(4, 16))

    def _browse_aero(self):
        path = filedialog.askopenfilename(title="Select data_aero .mat file",
                                           filetypes=[("MATLAB files", "*.mat")])
        if path:
            self.aero_path.set(path)
            self._try_load_data()

    def _browse_thrust(self):
        path = filedialog.askopenfilename(title="Select data_thrust .mat file",
                                           filetypes=[("MATLAB files", "*.mat")])
        if path:
            self.thrust_path.set(path)
            self._try_load_data()

    def _try_load_data(self):
        if not self.aero_path.get() or not self.thrust_path.get():
            return
        try:
            self.aero_db = AeroDB(self.aero_path.get())
            self.thrust_db = ThrustDB(self.thrust_path.get())
            self._set_load_status(
                f"Loaded OK - burn time from data_thrust = {self.thrust_db.burn_time:.3f} s",
                "ok")
        except Exception as e:
            self.aero_db = None
            self.thrust_db = None
            self._set_load_status(f"Load error: {e}", "err")

    def _set_load_status(self, text, kind):
        colors = {"ok": (THEME["ok"], "#E4F3E8"), "err": (THEME["err"], "#FBE7E5"),
                  "muted": (THEME["muted"], "#EEEAE2")}
        fg, bg = colors.get(kind, colors["muted"])
        self.load_status.config(text=text, fg=fg, bg=bg)

    # ------------------------------------------------------------------
    def _read_geometry(self):
        try:
            vals = {k: float(v.get()) for k, v in self.geom_vars.items()}
            rvals = {k: float(v.get()) for k, v in self.rail_vars.items()}
        except ValueError as e:
            raise ValueError(f"Invalid numeric input: {e}")
        return Geometry(
            D=vals["D"], L=vals["L"], mass_wet=vals["mass_wet"], mass_dry=vals["mass_dry"],
            Ixx_wet=vals["Ixx_wet"], Iyy_wet=vals["Iyy_wet"], Izz_wet=vals["Izz_wet"],
            Ixx_dry=vals["Ixx_dry"], Iyy_dry=vals["Iyy_dry"], Izz_dry=vals["Izz_dry"],
            Xcg_wet=vals["Xcg_wet"], Xcg_dry=vals["Xcg_dry"],
            Xthrust=vals["Xthrust"], L_rail=rvals["L_rail"],
            theta0_deg=rvals["theta0_deg"], psi0_deg=rvals["psi0_deg"],
        )

    def _ensure_data_ready(self):
        if self.aero_db is None or self.thrust_db is None:
            messagebox.showerror("Data not loaded",
                                  "Please select valid data_aero and data_thrust .mat files "
                                  "in the Inputs tab first.")
            return False
        return True

    # ------------------------------------------------------------------
    def _build_single_tab(self):
        f = self.tab_single
        top = ttk.Frame(f)
        top.pack(fill="x", padx=10, pady=10)
        ttk.Button(top, text="▶ Run Single Trajectory (Nominal)", style="Accent.TButton",
                   command=self._run_single).pack(side="left")
        self.single_status = self._badge(top, "Belum dijalankan", "muted")
        self.single_status.pack(side="left", padx=12)

        body = ttk.Frame(f)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        text_lf = self._section(body, "Ringkasan Metrik")
        text_lf.pack(side="left", fill="y", padx=(0, 8))
        self.single_text = tk.Text(text_lf, width=48, height=22, font=("Courier New", 11),
                                    bg=THEME["panel"], relief="flat", padx=10, pady=10)
        self.single_text.pack(fill="both", expand=True)

        plot_lf = self._section(body, "Profil Trajektori")
        plot_lf.pack(side="left", fill="both", expand=True)
        fig = Figure(figsize=(6, 5), dpi=100, facecolor=THEME["panel"])
        self.single_ax = fig.add_subplot(111)
        self.single_ax.set_facecolor(THEME["panel"])
        self.single_ax.set_xlabel("Downrange (m)")
        self.single_ax.set_ylabel("Altitude (m)")
        self.single_ax.set_title("Trajectory (ideal, nominal)")
        self.single_canvas = FigureCanvasTkAgg(fig, master=plot_lf)
        self.single_canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
        self.single_fig = fig

    def _run_single(self):
        if not self._ensure_data_ready():
            return
        try:
            geo = self._read_geometry()
        except ValueError as e:
            messagebox.showerror("Input error", str(e))
            return

        self.single_status.config(text="Menjalankan...", fg=THEME["muted"], bg="#EEEAE2")
        self.update_idletasks()

        def worker():
            try:
                p_nom = Perturb(theta0=np.deg2rad(geo.theta0_deg), psi0=np.deg2rad(geo.psi0_deg))
                res = run_single_simulation(geo, self.aero_db, self.thrust_db, p_nom,
                                             full_output=True)
                self.after(0, lambda: self._show_single_result(res))
            except Exception:
                err = traceback.format_exc()
                self.after(0, lambda: self._show_single_error(err))

        threading.Thread(target=worker, daemon=True).start()

    def _show_single_error(self, err):
        self.single_status.config(text="Error", fg=THEME["err"], bg="#FBE7E5")
        messagebox.showerror("Simulation error", err)

    def _show_single_result(self, res):
        if res is None:
            self.single_status.config(text="Error", fg=THEME["err"], bg="#FBE7E5")
            messagebox.showerror("Simulation error", "Integration failed (see console).")
            return
        self.single_status.config(text="Selesai", fg=THEME["ok"], bg="#E4F3E8")

        report = (
            "=== FLIGHT SIMULATION METRICS ===\n"
            f"Total Flight Time : {res['total_flight_time']:8.2f} s\n"
            f"Apogee Altitude   : {res['apogee_altitude']:8.1f} m (at t = {res['apogee_time']:6.2f} s)\n"
            f"Impact Range      : {res['impact_range']:8.1f} m\n"
            f"Maximum Velocity  : {res['max_velocity']:8.1f} m/s\n"
            f"Maximum Mach      : {res['max_mach']:8.2f}\n"
            "=================================\n"
        )
        self.single_text.delete("1.0", tk.END)
        self.single_text.insert(tk.END, report)

        self.single_ax.clear()
        downrange = np.sqrt(res['x'] ** 2 + res['y'] ** 2) * np.sign(res['x'] + 1e-9)
        self.single_ax.plot(downrange, res['alt'], color="#2b7bba")
        self.single_ax.set_xlabel("Downrange (m)")
        self.single_ax.set_ylabel("Altitude (m)")
        self.single_ax.set_title("Trajectory (ideal, nominal)")
        self.single_ax.grid(alpha=0.3)
        self.single_canvas.draw()

    # ------------------------------------------------------------------
    def _build_mc_tab(self):
        f = self.tab_mc
        top = ttk.Frame(f)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Label(top, text="N iterations (1 - 10000):").pack(side="left")
        self.n_var = tk.StringVar(value="200")
        ttk.Spinbox(top, from_=1, to=10000, textvariable=self.n_var, width=8).pack(
            side="left", padx=6)

        self.method_vars = {
            'uniform': tk.BooleanVar(value=True),
            'gaussian': tk.BooleanVar(value=True),
            'mcmc': tk.BooleanVar(value=False),
        }
        for m in ['uniform', 'gaussian', 'mcmc']:
            ttk.Checkbutton(top, text=m, variable=self.method_vars[m]).pack(side="left", padx=6)

        self.mc_run_btn = ttk.Button(top, text="▶ Run Monte Carlo", style="Accent.TButton",
                                      command=self._start_mc)
        self.mc_run_btn.pack(side="left", padx=12)
        self.mc_stop_btn = ttk.Button(top, text="■ Stop", command=self._stop_mc, state="disabled")
        self.mc_stop_btn.pack(side="left")

        self.mc_progress = ttk.Progressbar(top, length=200, mode="determinate")
        self.mc_progress.pack(side="left", padx=12)
        self.mc_status = ttk.Label(top, text="")
        self.mc_status.pack(side="left", padx=6)

        plot_lf = self._section(f, "Peta Dispersi Dampak (CEP)")
        plot_lf.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        fig = Figure(figsize=(7, 6), dpi=100, facecolor=THEME["panel"])
        self.mc_ax = fig.add_subplot(111)
        self.mc_ax.set_facecolor(THEME["panel"])
        self.mc_ax.set_title("CEP Impact Dispersion Map (live)")
        self.mc_ax.set_xlabel("Crossrange / East (m)")
        self.mc_ax.set_ylabel("Downrange / North (m)")
        self.mc_ax.set_aspect('equal', 'box')
        self.mc_ax.grid(alpha=0.3)
        self.mc_canvas = FigureCanvasTkAgg(fig, master=plot_lf)
        self.mc_canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
        self.mc_fig = fig
        self.mc_scatters = {}
        self.mc_colors = {'uniform': '#e47d2c', 'gaussian': '#2b7bba', 'mcmc': '#8e54a2'}

    def _start_mc(self):
        if not self._ensure_data_ready():
            return
        try:
            geo = self._read_geometry()
            N = int(self.n_var.get())
            if not (1 <= N <= 10000):
                raise ValueError("N must be between 1 and 10000")
        except ValueError as e:
            messagebox.showerror("Input error", str(e))
            return

        methods = [m for m, v in self.method_vars.items() if v.get()]
        if not methods:
            messagebox.showerror("Input error", "Select at least one Monte Carlo method.")
            return

        self.mc_ax.clear()
        self.mc_ax.set_title("CEP Impact Dispersion Map (live)")
        self.mc_ax.set_xlabel("Crossrange / East (m)")
        self.mc_ax.set_ylabel("Downrange / North (m)")
        self.mc_ax.grid(alpha=0.3)
        self.mc_ax.scatter(0, 0, color='green', marker='*', s=200, zorder=5,
                            edgecolors='black', label='Nominal Target')
        self.mc_scatters = {}
        for m in methods:
            sc = self.mc_ax.scatter([], [], s=10, alpha=0.4, color=self.mc_colors[m], label=m)
            self.mc_scatters[m] = sc
        self.mc_ax.legend(loc='upper right')
        self.mc_canvas.draw()

        self.mc_results = {m: {'x': [], 'y': []} for m in methods}
        self.mc_progress['maximum'] = N * len(methods)
        self.mc_progress['value'] = 0
        self.mc_run_btn.config(state="disabled")
        self.mc_stop_btn.config(state="normal")
        self.mc_stop_flag.clear()

        self.mc_thread = threading.Thread(target=self._mc_worker, args=(geo, N, methods),
                                           daemon=True)
        self.mc_thread.start()
        self.after(100, self._poll_mc_queue)

    def _stop_mc(self):
        self.mc_stop_flag.set()

    def _mc_worker(self, geo, N, methods):
        try:
            p_nom = Perturb(theta0=np.deg2rad(geo.theta0_deg), psi0=np.deg2rad(geo.psi0_deg))
            x_ref, y_ref = run_single_simulation(geo, self.aero_db, self.thrust_db, p_nom)
            self.mc_queue.put(('ref', x_ref, y_ref))

            for method in methods:
                samples = generate_samples(method, N, geo, seed=42)
                for i, s in enumerate(samples):
                    if self.mc_stop_flag.is_set():
                        self.mc_queue.put(('done', None, None))
                        return
                    x, y = run_single_simulation(geo, self.aero_db, self.thrust_db, s)
                    self.mc_queue.put(('point', method, (x - x_ref, y - y_ref)))
            self.mc_queue.put(('done', None, None))
        except Exception:
            self.mc_queue.put(('error', traceback.format_exc(), None))

    def _poll_mc_queue(self):
        updated = False
        try:
            while True:
                item = self.mc_queue.get_nowait()
                tag = item[0]
                if tag == 'ref':
                    pass
                elif tag == 'point':
                    _, method, (dx, dy) = item
                    if not (np.isnan(dx) or np.isnan(dy)):
                        self.mc_results[method]['x'].append(dy)  # east = y
                        self.mc_results[method]['y'].append(dx)  # north = x
                    self.mc_progress['value'] += 1
                    updated = True
                elif tag == 'error':
                    messagebox.showerror("Monte Carlo error", item[1])
                    self._mc_finished()
                    return
                elif tag == 'done':
                    self._mc_finished()
                    if updated:
                        self._redraw_mc()
                    return
        except queue.Empty:
            pass

        if updated:
            self._redraw_mc()

        if self.mc_thread and self.mc_thread.is_alive():
            self.after(150, self._poll_mc_queue)

    def _redraw_mc(self):
        for method, sc in self.mc_scatters.items():
            xs = self.mc_results[method]['x']
            ys = self.mc_results[method]['y']
            if xs:
                sc.set_offsets(np.column_stack([xs, ys]))
        all_vals = []
        for method in self.mc_scatters:
            all_vals += self.mc_results[method]['x'] + self.mc_results[method]['y']
        if all_vals:
            lim = max(1.0, max(abs(v) for v in all_vals) * 1.15)
            self.mc_ax.set_xlim(-lim, lim)
            self.mc_ax.set_ylim(-lim, lim)
        self.mc_canvas.draw_idle()

        n_done = int(self.mc_progress['value'])
        n_total = int(self.mc_progress['maximum'])
        self.mc_status.config(text=f"{n_done}/{n_total}")

    def _mc_finished(self):
        self.mc_run_btn.config(state="normal")
        self.mc_stop_btn.config(state="disabled")
        summary_lines = []
        for method, d in self.mc_results.items():
            xs, ys = np.array(d['x']), np.array(d['y'])
            if len(xs) == 0:
                continue
            r = np.sqrt(xs ** 2 + ys ** 2)
            cep50 = np.percentile(r, 50)
            circ = Circle((0, 0), cep50, fill=False, edgecolor=self.mc_colors[method],
                           linestyle='--', linewidth=1.8,
                           label=f"{method} CEP50={cep50:.1f}m")
            self.mc_ax.add_patch(circ)
            summary_lines.append(f"{method}: N={len(xs)}, CEP50={cep50:.1f} m, mean={r.mean():.1f} m")
        self.mc_ax.legend(loc='upper right', fontsize=8)
        self.mc_canvas.draw_idle()
        self.mc_status.config(text=" | ".join(summary_lines) if summary_lines else "Stopped.")

    # ------------------------------------------------------------------
    def _build_sens_tab(self):
        f = self.tab_sens
        top = ttk.Frame(f)
        top.pack(fill="x", padx=10, pady=10)
        ttk.Button(top, text="▶ Run Sensitivity Analysis (OAT)", style="Accent.TButton",
                   command=self._run_sensitivity).pack(side="left")
        self.sens_status = self._badge(top, "Belum dijalankan", "muted")
        self.sens_status.pack(side="left", padx=12)

        plot_lf = self._section(f, "Analisis Sensitivitas Parameter (One-At-a-Time)")
        plot_lf.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        fig = Figure(figsize=(8, 6), dpi=100, facecolor=THEME["panel"])
        self.sens_ax = fig.add_subplot(111)
        self.sens_ax.set_facecolor(THEME["panel"])
        self.sens_canvas = FigureCanvasTkAgg(fig, master=plot_lf)
        self.sens_canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
        self.sens_fig = fig

    def _run_sensitivity(self):
        if not self._ensure_data_ready():
            return
        try:
            geo = self._read_geometry()
        except ValueError as e:
            messagebox.showerror("Input error", str(e))
            return

        self.sens_status.config(text="Menjalankan (one-at-a-time)...", fg=THEME["muted"],
                                 bg="#EEEAE2")
        self.update_idletasks()

        def worker():
            try:
                sens, ref = run_sensitivity_analysis(geo, self.aero_db, self.thrust_db)
                self.after(0, lambda: self._show_sensitivity(sens, ref))
            except Exception:
                err = traceback.format_exc()
                self.after(0, lambda: (self.sens_status.config(
                    text="Error", fg=THEME["err"], bg="#FBE7E5"),
                    messagebox.showerror("Sensitivity error", err)))

        threading.Thread(target=worker, daemon=True).start()

    def _show_sensitivity(self, sens, ref):
        self.sens_status.config(text=f"Selesai - nominal impact = ({ref[0]:.1f}, {ref[1]:.1f}) m",
                                 fg=THEME["ok"], bg="#E4F3E8")
        sorted_items = sorted(sens.items(), key=lambda kv: kv[1])
        names = [k for k, _ in sorted_items]
        shifts = [v for _, v in sorted_items]

        self.sens_ax.clear()
        y_pos = np.arange(len(names))
        colors = plt.cm.plasma(np.linspace(0.3, 0.8, len(names)))
        self.sens_ax.barh(y_pos, shifts, color=colors, height=0.6)
        self.sens_ax.set_yticks(y_pos)
        self.sens_ax.set_yticklabels(names)
        self.sens_ax.set_xlabel("Maximum Impact Displacement (m)")
        self.sens_ax.set_title("OAT Parameter Sensitivity Analysis")
        self.sens_ax.grid(axis='x', alpha=0.3)
        for i, v in enumerate(shifts):
            self.sens_ax.text(v + max(shifts) * 0.01, i, f"{v:.1f} m", va='center', fontsize=8)
        self.sens_fig.tight_layout()
        self.sens_canvas.draw()


if __name__ == "__main__":
    app = CEPApp()
    app.mainloop()
