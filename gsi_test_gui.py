# gsi_test_gui.py
# CS2 GSI test with ONE smart countdown:
# - Start 1:55 when round.phase transitions freezetime -> live (competitive only)
# - If bomb becomes planted while running, reset remaining to 0:39 and continue
# - Stop on round over or when the timer hits 0

from flask import Flask, request
import threading, json, time
import tkinter as tk
from tkinter import ttk

app = Flask(__name__)

# ------------------
# Shared latest view
# ------------------
_latest_view = {}
_latest_meta = "Waiting for GSI data…"
_lock = threading.Lock()

# ------------------
# Timer settings
# ------------------
ROUND_FULL_S = 115  # 1:55
BOMB_PLANT_S = 39   # 0:39

# Internal state for transitions + unified timer
_last_round_phase = None
_last_bomb_planted = None

_timer_start_t = None     # monotonic() when current mode started
_timer_total_s = None     # 115 normally, switches to 39 on plant

def now():
    return time.monotonic()

def fmt_mmss(seconds_left):
    if seconds_left is None:
        return "—"
    s = max(0, int(seconds_left))
    m = s // 60
    s = s % 60
    return f"{m:02d}:{s:02d}"

def remaining_from(start_t, total_s):
    if start_t is None or total_s is None:
        return None
    return max(0.0, total_s - (now() - start_t))

def safe_get(d, *path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default

def parse_view(state: dict):
    map_mode    = safe_get(state, "map", "mode")
    map_phase   = safe_get(state, "map", "phase")
    round_phase = safe_get(state, "round", "phase")
    bomb_state  = safe_get(state, "round", "bomb")  # "planted", "exploded", "defused", or None
    bomb_planted = (bomb_state == "planted")
    return map_mode, map_phase, round_phase, bomb_planted

@app.route("/clutch", methods=["POST"])
def gsi_listener():
    global _latest_view, _latest_meta
    global _last_round_phase, _last_bomb_planted
    global _timer_start_t, _timer_total_s

    state = request.json or {}
    map_mode, map_phase, round_phase, bomb_planted = parse_view(state)
    competitive = (map_mode == "competitive")

    # ---------- Unified timer logic ----------
    if competitive:
        # Start 1:55 exactly on freezetime -> live
        if _last_round_phase == "freezetime" and round_phase == "live":
            _timer_start_t = now()
            _timer_total_s = ROUND_FULL_S

        # If bomb flips False -> True while timer is running, reset to 0:39
        if (_last_bomb_planted is False or _last_bomb_planted is None) and bomb_planted is True:
            if _timer_start_t is not None:
                _timer_start_t = now()
                _timer_total_s = BOMB_PLANT_S

        # Stop on round over
        if round_phase == "over":
            _timer_start_t = None
            _timer_total_s = None
    else:
        # Leaving competitive clears timer
        _timer_start_t = None
        _timer_total_s = None

    # Update last-seen markers
    _last_round_phase = round_phase
    _last_bomb_planted = bomb_planted

    # Compute remaining; if hits zero, stop
    t_left = remaining_from(_timer_start_t, _timer_total_s)
    if t_left is not None and t_left <= 0:
        _timer_start_t = None
        _timer_total_s = None
        t_left = None

    view = {
        "map_mode": map_mode,
        "map_phase": map_phase,
        "round_phase": round_phase,
        "bomb_planted": bomb_planted,
        "round_timer": fmt_mmss(t_left),
    }

    _latest_meta = (
        f"mode={map_mode} | map.phase={map_phase} | round.phase={round_phase} | "
        f"bomb.planted={bomb_planted} | round_timer={view['round_timer']}"
    )

    with _lock:
        _latest_view = view

    return "ok"

# ------------------
# Tiny Tkinter GUI
# ------------------
def run_gui():
    root = tk.Tk()
    root.title("GSI Test — Latest Call")
    root.geometry("640x340")

    status = tk.StringVar(value="Waiting for GSI data…")
    lbl = ttk.Label(root, textvariable=status, font=("Consolas", 11))
    lbl.pack(anchor="w", padx=12, pady=(10, 6))

    txt = tk.Text(root, wrap="none", font=("Consolas", 12), height=10)
    txt.pack(fill="both", expand=True, padx=12, pady=(0, 12))
    txt.configure(state="disabled")

    yscroll = ttk.Scrollbar(root, orient="vertical", command=txt.yview)
    yscroll.place(in_=txt, relx=1.0, rely=0, relheight=1.0, x=-2, width=12)
    txt.configure(yscrollcommand=yscroll.set)

    def refresh():
        # live tick in GUI so the seconds move smoothly between POSTs
        with _lock:
            meta = _latest_meta
            view = _latest_view.copy()
            start = globals().get("_timer_start_t", None)
            total = globals().get("_timer_total_s", None)

        if start is not None and total is not None:
            t_live = remaining_from(start, total)
            if t_live is not None and t_live > 0:
                view["round_timer"] = fmt_mmss(t_live)
            else:
                view["round_timer"] = "—"

        status.set(meta)
        pretty = json.dumps(view, indent=2)
        txt.configure(state="normal")
        txt.delete("1.0", "end")
        txt.insert("1.0", pretty)
        txt.configure(state="disabled")

        root.after(100, refresh)

    refresh()
    root.mainloop()

def run_flask():
    app.run(port=3200, debug=False, use_reloader=False, threaded=True)

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    time.sleep(0.2)
    run_gui()
