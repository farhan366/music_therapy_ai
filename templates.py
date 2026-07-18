"""
templates.py
============
Custom HTML/CSS snippets injected via st.markdown(..., unsafe_allow_html=True).
Keeping these separate from app.py keeps the Streamlit control flow readable.
"""

STRESS_ALERT_CARD = """
<div style="
    background: linear-gradient(135deg, #2b1055 0%, #7597de 100%);
    border-radius: 14px;
    padding: 18px 22px;
    margin: 10px 0 18px 0;
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    color: white;
    border-left: 6px solid #ff6b6b;
">
    <div style="font-size: 15px; font-weight: 700; letter-spacing: 0.5px; opacity: 0.9;">
        ⚠️ ACUTE STRESS STATE DETECTED
    </div>
    <div style="font-size: 14px; margin-top: 6px; line-height: 1.4;">
        Your physiological signals (HRV / EDA) indicate elevated stress
        (model confidence: <b>{confidence:.0%}</b>).
        Consider switching to a low-tempo therapeutic track below to help
        bring your body back to baseline.
    </div>
</div>
"""

RECOVERY_CARD = """
<div style="
    background: linear-gradient(135deg, #134e5e 0%, #71b280 100%);
    border-radius: 14px;
    padding: 18px 22px;
    margin: 10px 0 18px 0;
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    color: white;
    border-left: 6px solid #6bffb0;
">
    <div style="font-size: 15px; font-weight: 700; letter-spacing: 0.5px; opacity: 0.9;">
        ✅ HOMEOSTASIS RESTORED
    </div>
    <div style="font-size: 14px; margin-top: 6px; line-height: 1.4;">
        Your physiological signals have returned to the relaxed baseline
        zone. Nice work -- the down-regulation worked.
    </div>
</div>
"""

STATUS_BADGE_STRESS = """
<span style="
    background:#ff6b6b; color:white; padding:4px 12px; border-radius:20px;
    font-size:13px; font-weight:600;">🔴 Acute Stress</span>
"""

STATUS_BADGE_BASELINE = """
<span style="
    background:#4caf82; color:white; padding:4px 12px; border-radius:20px;
    font-size:13px; font-weight:600;">🟢 Relaxed Baseline</span>
"""


def track_card_html(track_name, bpm, genre, is_recommended=False):
    border = "border: 2px solid #6bffb0;" if is_recommended else "border: 1px solid #3a3a3a;"
    badge = (
        '<div style="font-size:11px; color:#6bffb0; font-weight:700; '
        'margin-bottom:4px;">RECOMMENDED · LOW TEMPO</div>'
        if is_recommended else ""
    )
    return f"""
    <div style="
        background:#1e1e2f; {border} border-radius:10px; padding:12px 14px;
        margin-bottom:8px;">
        {badge}
        <div style="font-weight:600; font-size:14px; color:white;">{track_name}</div>
        <div style="font-size:12px; color:#9a9ab0; margin-top:2px;">{genre} · {bpm} BPM</div>
    </div>
    """


def metric_tile_html(label, value, unit="", color="#7597de"):
    return f"""
    <div style="
        background:#1a1a2a; border-radius:10px; padding:14px; text-align:center;
        border-top: 3px solid {color};">
        <div style="font-size:12px; color:#9a9ab0; text-transform:uppercase; letter-spacing:0.5px;">
            {label}
        </div>
        <div style="font-size:24px; font-weight:700; color:white; margin-top:4px;">
            {value}<span style="font-size:13px; color:#9a9ab0;"> {unit}</span>
        </div>
    </div>
    """
