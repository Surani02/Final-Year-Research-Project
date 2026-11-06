"""
Run from Research_Implementation/:
    streamlit run web_demo/app.py
"""

import os
import sys
import time

# ── path setup ────────────────────────────────────────────────────────────── #
_ROOT = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Streamlit  (page_config MUST come first) ──────────────────────────────── #
import streamlit as st

st.set_page_config(
    page_title = "FYP Demo",
    page_icon  = "🧠",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)

# ── project imports ────────────────────────────────────────────────────────── #
from web_demo.episode import EpisodeRunner, MODE_DISPLAY, ACTION_LABELS
from envs.gym_env     import SUPPORTED_ENVS
from train_full_model import is_success

# ══════════════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════════════

ENVS   = [env for env in SUPPORTED_ENVS if env != "MountainCar-v0"]
MODES  = list(MODE_DISPLAY.keys())
SEEDS  = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

ENV_DESCRIPTIONS = {
    "CartPole-v1":    "Balance a pole on a cart — goal: keep pole upright for 500 steps.",
    "Acrobot-v1":     "Swing a two-link pendulum to raise the tip above a threshold.",
    "LunarLander-v3": "Safely land a spacecraft between two flags.",
    "MountainCar-v0": "Drive a car up a steep hill using momentum.",
}

MODE_DESCRIPTIONS = {
    "baseline":       "Pure PPO-Env. No appraisal, no cognitive agent, no constraint. Performance floor.",
    "affect_reward":  "PPO-Env + Appraisal. Cognitive reward R_cog shapes training. No blending.",
    "appraisal_only": "PPO-Env + PPO-Cog + Appraisal. Policy blending active. No constraint module.",
    "full":           "Complete Framework. All components active: appraisal, blending, constraint enforcement.",
}

MODE_BADGE_COLOR = {
    "baseline":       "#64748b",
    "affect_reward":  "#f59e0b",
    "appraisal_only": "#3b82f6",
    "full":           "#10b981",
}

# ══════════════════════════════════════════════════════════════════════════════
#  Custom CSS
# ══════════════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
<style>
/* ── global ── */
/* Use local fallbacks so the packaged demo never requests remote resources. */
html, body, [class*="css"] { font-family: Arial, sans-serif; }

/* hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }

/* ── hero header ── */
.hero {
    background: linear-gradient(135deg, #0f1928 0%, #0a0f1e 50%, #111827 100%);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,212,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-size: 1.9rem; font-weight: 700; margin: 0;
    background: linear-gradient(90deg, #00d4ff, #7b61ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p {
    font-size: 0.95rem; color: #94a3b8; margin: 6px 0 0;
}

/* ── mode badge ── */
.mode-badge {
    display: inline-block;
    padding: 4px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600;
    letter-spacing: 0.04em; text-transform: uppercase;
}

/* ── metric card ── */
.metric-card {
    background: rgba(15, 25, 40, 0.7);
    border: 1px solid rgba(0, 212, 255, 0.12);
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 10px;
    backdrop-filter: blur(8px);
}
.metric-card-title {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #64748b; margin-bottom: 10px;
}

/* ── individual metric row ── */
.m-row {
    display: flex; align-items: center;
    gap: 8px; margin-bottom: 8px;
}
.m-label {
    font-size: 0.8rem; color: #94a3b8;
    min-width: 140px; flex-shrink: 0;
}
.m-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem; font-weight: 500;
    min-width: 70px; text-align: right; flex-shrink: 0;
}
.m-bar-track {
    flex: 1; height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 3px; overflow: hidden;
    min-width: 60px;
}
.m-bar-fill {
    height: 100%; border-radius: 3px;
    transition: width 0.25s ease;
}

/* ── n/a tag ── */
.na-row {
    display: flex; align-items: center; gap: 8px; margin-bottom: 8px;
}
.na-tag {
    font-size: 0.75rem; color: #475569;
    font-style: italic; padding: 2px 8px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 4px;
}

/* ── episode summary ── */
.summary-card {
    background: linear-gradient(135deg, rgba(16,185,129,0.08), rgba(0,212,255,0.05));
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 14px; padding: 20px 24px;
    margin-top: 12px;
}
.summary-fail {
    background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(220,38,38,0.04));
    border-color: rgba(239,68,68,0.25);
}
.summary-title {
    font-size: 1.1rem; font-weight: 700; margin-bottom: 14px;
}
.summary-row {
    display: flex; justify-content: space-between;
    font-size: 0.85rem; padding: 5px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

/* ── game frame overlay ── */
.frame-container {
    position: relative; border-radius: 12px; overflow: hidden;
    border: 1px solid rgba(0, 212, 255, 0.18);
    background: #000;
}
.done-overlay {
    background: rgba(0,0,0,0.75);
    border-radius: 12px;
    text-align: center;
    padding: 20px;
    font-size: 1.1rem; font-weight: 600; color: #e2e8f0;
}

/* ── comparison header ── */
.compare-header {
    background: rgba(15,25,40,0.5);
    border: 1px solid rgba(0,212,255,0.1);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    text-align: center;
}
.compare-label {
    font-size: 0.9rem; font-weight: 600; color: #e2e8f0;
}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px; background: transparent;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    background: rgba(255,255,255,0.04);
    color: #94a3b8;
    font-weight: 500;
    border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [aria-selected="true"] {
    background: rgba(0,212,255,0.12) !important;
    color: #00d4ff !important;
    border-color: rgba(0,212,255,0.3) !important;
}

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #7b61ff);
    color: #000; font-weight: 700; border: none; border-radius: 8px;
    padding: 10px 28px; font-size: 0.95rem;
    transition: opacity 0.2s, transform 0.1s;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

/* ── selectbox / number input ── */
.stSelectbox > div, .stNumberInput > div {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HTML helpers
# ══════════════════════════════════════════════════════════════════════════════

def _bar_fill_color(value, lo, hi, kind="neutral"):
    """Return a CSS color string based on value position and kind."""
    if kind == "goal":
        if value > 0.3:  return "#10b981"
        if value < -0.3: return "#ef4444"
        return "#f59e0b"
    if kind == "divergence":
        if value > 0.5:  return "#ef4444"
        if value > 0.2:  return "#f59e0b"
        return "#10b981"
    if kind == "constraint":
        if value > 0.20: return "#ef4444"
        if value > 0.10: return "#f59e0b"
        return "#10b981"
    if kind == "reward":
        return "#10b981" if value >= 0 else "#ef4444"
    if kind == "novelty":  return "#f59e0b"
    if kind == "coping":   return "#4fc3f7"
    if kind == "beta":     return "#a78bfa"
    return "#00d4ff"


def _pct(value, lo, hi):
    return max(0.0, min(100.0, (value - lo) / (hi - lo + 1e-9) * 100))


def metric_row(label, value, lo, hi, fmt="+.4f", kind="neutral"):
    color = _bar_fill_color(value, lo, hi, kind)
    pct   = _pct(value, lo, hi)
    v_str = format(value, fmt)
    return f"""
    <div class="m-row">
        <span class="m-label">{label}</span>
        <span class="m-value" style="color:{color}">{v_str}</span>
        <div class="m-bar-track">
            <div class="m-bar-fill" style="width:{pct:.1f}%;background:{color}"></div>
        </div>
    </div>"""


def metric_na(label):
    return f"""
    <div class="na-row">
        <span class="m-label">{label}</span>
        <span class="na-tag">n/a — not active in this mode</span>
    </div>"""


def mode_badge(mode):
    color = MODE_BADGE_COLOR.get(mode, "#64748b")
    name  = MODE_DISPLAY.get(mode, mode)
    return (f'<span class="mode-badge" '
            f'style="background:rgba({_hex_to_rgba(color)},0.15);'
            f'color:{color};border:1px solid rgba({_hex_to_rgba(color)},0.4)">'
            f'{name}</span>')


def _hex_to_rgba(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"{r},{g},{b}"


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics panel (HTML block rendered via st.markdown)
# ══════════════════════════════════════════════════════════════════════════════

def build_metrics_html(m: dict) -> str:
    flags = m["flags"]
    g, n, p   = m["g"], m["n"], m["p"]
    cas       = m["cas"]
    D_t, bt, C_t = m["D_t"], m["beta_t"], m["C_t"]

    sections = []

    # ── Episode info ──
    act_str = m["action_label"]
    r_color = "#10b981" if m["reward"] >= 0 else "#ef4444"
    ep_color= "#10b981" if m["ep_reward"] >= 0 else "#ef4444"
    sections.append(f"""
    <div class="metric-card">
        <div class="metric-card-title">📊 Episode</div>
        <div class="m-row">
            <span class="m-label">Step</span>
            <span class="m-value" style="color:#e2e8f0">{m["step"]}</span>
        </div>
        <div class="m-row">
            <span class="m-label">Action</span>
            <span class="m-value" style="color:#fbbf24; font-size:0.78rem">{m["action"]} · {act_str}</span>
        </div>
        {metric_row("Step Reward",    m["reward"],    -1,   1,   fmt="+.4f", kind="reward")}
        {metric_row("Episode Reward", m["ep_reward"], -500, 500, fmt="+.2f", kind="reward")}
    </div>""")

    # ── Appraisal ──
    if flags["use_appraisal"]:
        sections.append(f"""
    <div class="metric-card">
        <div class="metric-card-title">🧠 Appraisal Variables</div>
        {metric_row("Goal Congruence (ĝ)",  g,  -1, 1, fmt="+.4f", kind="goal")}
        {metric_row("Novelty (n̂)",          n,   0, 2, fmt=".4f",  kind="novelty")}
        {metric_row("Coping Potential (p̂)", p,   0, 1, fmt=".4f",  kind="coping")}
    </div>""")
        sections.append(f"""
    <div class="metric-card">
        <div class="metric-card-title">💭 Cognitive-Affective State  c_t</div>
        {metric_row("c[0] Goal Memory", cas[0], -1, 1, fmt="+.4f", kind="goal")}
        {metric_row("c[1] Novelty",     cas[1],  0, 2, fmt=".4f",  kind="novelty")}
        {metric_row("c[2] Coping",      cas[2],  0, 1, fmt=".4f",  kind="coping")}
    </div>""")
    else:
        sections.append(f"""
    <div class="metric-card">
        <div class="metric-card-title">🧠 Appraisal / Cognitive State</div>
        {metric_na("Appraisal (ĝ, n̂, p̂)")}
        {metric_na("Cognitive State c_t")}
    </div>""")

    # ── Divergence ──
    if flags["use_cog_ppo"]:
        sections.append(f"""
    <div class="metric-card">
        <div class="metric-card-title">📐 Policy Divergence & Influence</div>
        {metric_row("JS Divergence (D_t)", D_t, 0, 1,   fmt=".4f", kind="divergence")}
        {metric_row("Influence β",         bt,  0, 0.3, fmt=".4f", kind="beta")}
    </div>""")
    else:
        sections.append(f"""
    <div class="metric-card">
        <div class="metric-card-title">📐 Policy Divergence & Influence</div>
        {metric_na("JS Divergence (D_t)")}
        {metric_na("Influence β")}
    </div>""")

    # ── Constraint ──
    if flags["use_constraint"]:
        violation = C_t > 0.20
        vstatus   = ('🚨 <span style="color:#ef4444;font-weight:600">VIOLATION</span>'
                     if violation else
                     '✅ <span style="color:#10b981;font-weight:600">Safe</span>')
        sections.append(f"""
    <div class="metric-card">
        <div class="metric-card-title">🔒 Constraint Module</div>
        {metric_row("Constraint Cost (C_t)", C_t, 0, 0.4, fmt=".4f", kind="constraint")}
        <div class="m-row">
            <span class="m-label">Threshold d</span>
            <span class="m-value" style="color:#94a3b8">0.20</span>
        </div>
        <div class="m-row">
            <span class="m-label">Status</span>
            <span style="font-size:0.82rem">{vstatus}</span>
        </div>
    </div>""")
    else:
        sections.append(f"""
    <div class="metric-card">
        <div class="metric-card-title">🔒 Constraint Module</div>
        {metric_na("Constraint Cost (C_t)")}
    </div>""")

    return "\n".join(sections)


# ══════════════════════════════════════════════════════════════════════════════
#  Episode summary card
# ══════════════════════════════════════════════════════════════════════════════

def render_summary(runner: EpisodeRunner):
    m      = runner._last_metrics
    ep_r   = runner.ep_reward
    steps  = runner.step_count
    succ   = bool(is_success(runner.env_name, ep_r, steps,
                             runner._terminated, runner._truncated))
    flags  = runner.flags
    css    = "" if succ else "summary-fail"
    icon   = "✅ SUCCESS" if succ else "❌ Failed"
    color  = "#10b981" if succ else "#ef4444"

    def srow(label, val):
        return f'<div class="summary-row"><span>{label}</span><span style="font-weight:600">{val}</span></div>'

    html = f"""
    <div class="summary-card {css}">
        <div class="summary-title" style="color:{color}">{icon}</div>
        {srow("Mode",           MODE_DISPLAY[runner.mode])}
        {srow("Environment",    runner.env_name)}
        {srow("Total Steps",    str(steps))}
        {srow("Episode Reward", f"{ep_r:+.4f}")}
    """
    if flags["use_appraisal"]:
        html += srow("Avg Goal Congruence",   f"{m['g']:+.4f}")
        html += srow("Avg Novelty",            f"{m['n']:.4f}")
        html += srow("Avg Coping Potential",   f"{m['p']:.4f}")
    if flags["use_cog_ppo"]:
        html += srow("JS Divergence (last)",   f"{m['D_t']:.4f}")
        html += srow("Influence β (last)",     f"{m['beta_t']:.4f}")
    if flags["use_constraint"]:
        html += srow("Constraint Cost (last)", f"{m['C_t']:.4f}")
    html += "</div>"
    return html


# ══════════════════════════════════════════════════════════════════════════════
#  Control panel (shared by both tabs)
# ══════════════════════════════════════════════════════════════════════════════

def control_panel(prefix="single", default_env="CartPole-v1",
                  default_mode="full", default_seed=42):
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    env  = c1.selectbox("Environment",  ENVS,  index=ENVS.index(default_env),  key=f"{prefix}_env")
    mode = c2.selectbox("Mode",         MODES, index=MODES.index(default_mode), key=f"{prefix}_mode",
                        format_func=lambda m: MODE_DISPLAY[m])
    seed = c3.selectbox("Seed",   SEEDS, index=SEEDS.index(default_seed), key=f"{prefix}_seed")
    delay= c4.slider("Speed (s/step)", 0.0, 0.3, 0.04, 0.01, key=f"{prefix}_delay")
    # Description strip
    st.markdown(
        f'<div style="font-size:0.8rem;color:#64748b;padding:4px 0 8px">'
        f'🌍 {ENV_DESCRIPTIONS[env]} &nbsp;&nbsp;|&nbsp;&nbsp; '
        f'🤖 {MODE_DESCRIPTIONS[mode]}</div>',
        unsafe_allow_html=True
    )
    return env, mode, seed, delay


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 1: Single Model Demo
# ══════════════════════════════════════════════════════════════════════════════

def tab_single_demo():
    env, mode, seed, delay = control_panel("single")
    run_btn = st.button("▶  Run Episode", key="run_single")

    if not run_btn:
        st.markdown(
            '<div style="text-align:center;padding:60px;color:#475569;">'
            '⬆  Configure the model above, then click <strong>Run Episode</strong> to start.</div>',
            unsafe_allow_html=True,
        )
        return

    # Create runner
    with st.spinner("Loading checkpoints …"):
        try:
            runner = EpisodeRunner(env, mode, seed)
        except FileNotFoundError as e:
            st.error(str(e))
            return

    frame = runner.reset()

    col_frame, col_metrics = st.columns([5, 3], gap="medium")

    # placeholders
    frame_ph   = col_frame.empty()
    done_ph    = col_frame.empty()
    metrics_ph = col_metrics.empty()
    summary_ph = st.empty()

    frame_ph.image(frame, use_container_width=True, caption=f"{env}  ·  {MODE_DISPLAY[mode]}")
    with metrics_ph.container():
        st.markdown(build_metrics_html(runner._last_metrics), unsafe_allow_html=True)

    # ── episode loop ──
    while not runner.done:
        frame, metrics = runner.step()
        frame_ph.image(frame, use_container_width=True,
                       caption=f"Step {metrics['step']}  ·  {env}  ·  {MODE_DISPLAY[mode]}")
        with metrics_ph.container():
            st.markdown(build_metrics_html(metrics), unsafe_allow_html=True)
        if delay > 0:
            time.sleep(delay)

    # ── episode done ──
    runner.close()
    done_ph.markdown(
        '<div class="done-overlay" style="background:rgba(0,0,0,0.6);'
        'border-radius:8px;padding:12px;text-align:center;margin-top:4px;">'
        '🏁 Episode Complete</div>',
        unsafe_allow_html=True,
    )
    summary_ph.markdown(render_summary(runner), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 2: Side-by-Side Comparison
# ══════════════════════════════════════════════════════════════════════════════

def tab_comparison():
    st.markdown(
        '<div style="font-size:0.85rem;color:#64748b;margin-bottom:12px">'
        'Configure two models independently and run them simultaneously '
        'to compare their behaviour and cognitive metrics in real time.</div>',
        unsafe_allow_html=True,
    )

    col_a_ctrl, _sep, col_b_ctrl = st.columns([5, 0.2, 5])

    with col_a_ctrl:
        st.markdown('<div style="font-weight:600;color:#00d4ff;margin-bottom:6px">🅰  Model A</div>',
                    unsafe_allow_html=True)
        env_a, mode_a, seed_a, delay_a = control_panel(
            "a", default_env="CartPole-v1", default_mode="baseline")

    with col_b_ctrl:
        st.markdown('<div style="font-weight:600;color:#7b61ff;margin-bottom:6px">🅱  Model B</div>',
                    unsafe_allow_html=True)
        env_b, mode_b, seed_b, delay_b = control_panel(
            "b", default_env="CartPole-v1", default_mode="full")

    delay = min(delay_a, delay_b)   # sync both to the faster setting

    run_btn = st.button("▶  Run Comparison", key="run_compare")
    if not run_btn:
        st.markdown(
            '<div style="text-align:center;padding:50px;color:#475569;">'
            '⬆  Configure both models above, then click <strong>Run Comparison</strong>.</div>',
            unsafe_allow_html=True,
        )
        return

    # ── load both runners ──
    with st.spinner("Loading checkpoints for both models …"):
        try:
            runner_a = EpisodeRunner(env_a, mode_a, seed_a)
        except FileNotFoundError as e:
            st.error(f"Model A: {e}")
            return
        try:
            runner_b = EpisodeRunner(env_b, mode_b, seed_b)
        except FileNotFoundError as e:
            st.error(f"Model B: {e}")
            runner_a.close()
            return

    runner_a.reset()
    runner_b.reset()

    # ── layout ──
    col_a, col_b = st.columns(2, gap="medium")

    with col_a:
        st.markdown(
            f'<div class="compare-header">'
            f'<span class="compare-label">🅰 Model A &nbsp;·&nbsp; {env_a}</span><br>'
            f'{mode_badge(mode_a)}</div>',
            unsafe_allow_html=True,
        )
        frame_a_ph   = st.empty()
        metrics_a_ph = st.empty()
        done_a_ph    = st.empty()

    with col_b:
        st.markdown(
            f'<div class="compare-header">'
            f'<span class="compare-label">🅱 Model B &nbsp;·&nbsp; {env_b}</span><br>'
            f'{mode_badge(mode_b)}</div>',
            unsafe_allow_html=True,
        )
        frame_b_ph   = st.empty()
        metrics_b_ph = st.empty()
        done_b_ph    = st.empty()

    summary_ph = st.empty()

    # Initial frames
    frame_a_ph.image(runner_a._last_frame, use_container_width=True)
    frame_b_ph.image(runner_b._last_frame, use_container_width=True)

    frame_a = runner_a._last_frame
    frame_b = runner_b._last_frame

    # ── synchronized episode loop ──
    while not (runner_a.done and runner_b.done):
        if not runner_a.done:
            frame_a, m_a = runner_a.step()
        else:
            m_a = runner_a._last_metrics

        if not runner_b.done:
            frame_b, m_b = runner_b.step()
        else:
            m_b = runner_b._last_metrics

        frame_a_ph.image(frame_a, use_container_width=True,
                         caption=f"Step {m_a['step']}  ·  Reward {m_a['ep_reward']:+.2f}")
        frame_b_ph.image(frame_b, use_container_width=True,
                         caption=f"Step {m_b['step']}  ·  Reward {m_b['ep_reward']:+.2f}")

        with metrics_a_ph.container():
            st.markdown(build_metrics_html(m_a), unsafe_allow_html=True)
        with metrics_b_ph.container():
            st.markdown(build_metrics_html(m_b), unsafe_allow_html=True)

        if delay > 0:
            time.sleep(delay)

    # ── both done ──
    runner_a.close()
    runner_b.close()

    done_a_ph.markdown(
        '<div class="done-overlay" style="background:rgba(0,0,0,0.55);border-radius:8px;'
        'padding:10px;text-align:center;margin-top:4px;">🏁 Episode Complete</div>',
        unsafe_allow_html=True,
    )
    done_b_ph.markdown(
        '<div class="done-overlay" style="background:rgba(0,0,0,0.55);border-radius:8px;'
        'padding:10px;text-align:center;margin-top:4px;">🏁 Episode Complete</div>',
        unsafe_allow_html=True,
    )

    # ── combined summary ──
    ep_a = runner_a.ep_reward
    ep_b = runner_b.ep_reward
    if ep_a > ep_b:
        winner = f"🏆 Model A ({MODE_DISPLAY[mode_a]}) wins with reward {ep_a:+.2f}"
        w_col  = "#00d4ff"
    elif ep_b > ep_a:
        winner = f"🏆 Model B ({MODE_DISPLAY[mode_b]}) wins with reward {ep_b:+.2f}"
        w_col  = "#7b61ff"
    else:
        winner = "🤝 Tied episode!"
        w_col  = "#f59e0b"

    sum_html = f"""
    <div style="background:rgba(15,25,40,0.8);border:1px solid rgba(0,212,255,0.2);
                border-radius:14px;padding:20px 24px;margin-top:16px;">
        <div style="font-size:1.1rem;font-weight:700;color:{w_col};margin-bottom:14px">{winner}</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
            <div>
                <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;
                            letter-spacing:0.06em;margin-bottom:8px">Model A · {MODE_DISPLAY[mode_a]}</div>
                <div class="summary-row"><span>Environment</span><span>{env_a}</span></div>
                <div class="summary-row"><span>Steps</span><span>{runner_a.step_count}</span></div>
                <div class="summary-row"><span>Episode Reward</span>
                    <span style="font-weight:700;color:{'#10b981' if ep_a>=0 else '#ef4444'}">{ep_a:+.4f}</span></div>
            </div>
            <div>
                <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;
                            letter-spacing:0.06em;margin-bottom:8px">Model B · {MODE_DISPLAY[mode_b]}</div>
                <div class="summary-row"><span>Environment</span><span>{env_b}</span></div>
                <div class="summary-row"><span>Steps</span><span>{runner_b.step_count}</span></div>
                <div class="summary-row"><span>Episode Reward</span>
                    <span style="font-weight:700;color:{'#10b981' if ep_b>=0 else '#ef4444'}">{ep_b:+.4f}</span></div>
            </div>
        </div>
    </div>"""
    summary_ph.markdown(sum_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Main layout
# ══════════════════════════════════════════════════════════════════════════════

def main():
    inject_css()

    # Hero header
    st.markdown("""
    <div class="hero">
        <h1>Cognitive-Affective Constraint Framework for PPO</h1>
        <p>Demonstration</p>
    </div>
    """, unsafe_allow_html=True)

    tab_single, tab_compare = st.tabs([
        "🎮  Single Model Demo",
        "⚖️  Side-by-Side Comparison",
    ])

    with tab_single:
        tab_single_demo()

    with tab_compare:
        tab_comparison()


if __name__ == "__main__":
    main()
