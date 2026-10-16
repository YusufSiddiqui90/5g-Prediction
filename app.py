import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="5G Experience Predictor", page_icon="📡", layout="wide")

# ---------- Load ----------
@st.cache_resource
def load():
    bundle = joblib.load("model.pkl")
    meta = json.load(open("meta.json"))
    return bundle, meta

bundle, meta = load()
reg, clf, COLS = bundle["reg"], bundle["clf"], bundle["columns"]
LV, RG = meta["levels"], meta["ranges"]

CLASS_COLOR = {"Poor": "#ff4d6d", "Fair": "#ffa94d", "Good": "#51cf66", "Excellent": "#22d3ee"}
CLASS_EMOJI = {"Poor": "😖", "Fair": "😐", "Good": "🙂", "Excellent": "🤩"}
CLASS_MSG = {
    "Poor": "Experience kharab rahega. Buffering, lag aur drops expect kar.",
    "Fair": "Chal jayega, par heavy use mein dikkat aa sakti hai.",
    "Good": "Smooth experience, zyadatar kaam bina rukawat ke.",
    "Excellent": "Ekdum mast! Network bilkul flawless chalega.",
}
nice = lambda s: s.replace("_", " ")

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg,#1b1450 0%,#3a1c71 45%,#0e7490 100%);
  padding: 28px 34px; border-radius: 20px; margin-bottom: 22px;
  box-shadow: 0 10px 40px rgba(124,92,255,.25);}
.hero h1 {margin:0; font-size: 2.1rem; color:#fff;}
.hero p {margin:6px 0 0; color:#cfd8ff; font-size:1.02rem;}
.card {background:#131a2b; border:1px solid #232c45; border-radius:18px; padding:22px;}
.badge {display:inline-block; padding:10px 26px; border-radius:999px; font-weight:800;
  font-size:1.7rem; letter-spacing:.5px;}
.small {color:#93a0c4; font-size:.88rem;}
.kpi {background:#131a2b; border:1px solid #232c45; border-radius:14px; padding:14px 18px; text-align:center;}
.kpi b {font-size:1.4rem; color:#fff;} .kpi span {display:block; color:#93a0c4; font-size:.8rem;}
div.stButton > button {background: linear-gradient(90deg,#7c5cff,#22d3ee); color:white; border:0;
  border-radius:12px; padding:.7rem 1rem; font-weight:700; font-size:1.05rem;}
div.stButton > button:hover {filter: brightness(1.12); color:white;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📡 India 5G Experience Predictor</h1>
  <p>Session ki details daal aur AI batayega ki user ka network experience kaisa rahega.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Inputs (sidebar) ----------
SCENARIOS = {
    "Custom": {},
    "🏏 IPL match, Dense Urban": dict(Application_Type="Live_Sports_Streaming", Event_Context="IPL_Match",
        Area_Type="Dense_Urban", Cell_Load_pct=92, Active_Users_In_Cell=280, Jitter_ms=70),
    "🌾 Rural, normal din": dict(Application_Type="Web_Browsing", Event_Context="Normal",
        Area_Type="Rural", Cell_Load_pct=30, Active_Users_In_Cell=20, Jitter_ms=8),
    "🎮 Cloud Gaming, rush hour": dict(Application_Type="Cloud_Gaming", Event_Context="Office_Rush",
        Area_Type="Urban", Cell_Load_pct=80, Active_Users_In_Cell=190, Jitter_ms=50),
}
sb = st.sidebar
sb.header("⚙️ Session details")
pick = sb.selectbox("Quick scenario", list(SCENARIOS))
pre = SCENARIOS[pick]

def sel(label, col):
    opts = LV[col]
    idx = opts.index(pre[col]) if col in pre else 0
    return sb.selectbox(label, opts, index=idx, format_func=nice, key=f"{col}-{pick}")

def num(label, col, step=1.0):
    lo, hi, med = RG[col]
    val = float(pre.get(col, med))
    return sb.slider(label, float(lo), float(hi), val, step=step, key=f"{col}-{pick}")

sb.subheader("📱 Usage")
app = sel("Application", "Application_Type")
device = sel("Device", "Device_Category")
plan = sel("Plan", "Plan_Type")
indoor = sel("Indoor / Outdoor", "Indoor_Outdoor")
speed = num("Moving speed (km/h)", "Mobility_Speed_kmph")

sb.subheader("🗼 Network")
operator = sel("Operator", "Operator")
mode = sel("Network mode", "Network_Mode")
load = num("Cell load (%)", "Cell_Load_pct")
users = num("Active users in cell", "Active_Users_In_Cell")
jitter = num("Jitter (ms)", "Jitter_ms")
signal = num("Signal (dBm)", "Signal_dBm")
latency = num("Latency (ms)", "Latency_ms")
req = num("Requested speed (Mbps)", "Req_Mbps", step=0.5)
alloc = num("Resource allocation (%)", "Resource_Alloc_pct")

sb.subheader("🌍 Location & Conditions")
circle = sel("Circle", "Circle")
tier = sel("City tier", "City_Tier")
area = sel("Area type", "Area_Type")
weather = sel("Weather", "Weather")
event = sel("Event", "Event_Context")

row = dict(Application_Type=app, Signal_dBm=signal, Latency_ms=latency, Req_Mbps=req,
           Resource_Alloc_pct=alloc, Operator=operator, Circle=circle, City_Tier=tier,
           Area_Type=area, Network_Mode=mode, Jitter_ms=jitter, Cell_Load_pct=load,
           Active_Users_In_Cell=int(users), Device_Category=device, Plan_Type=plan,
           Mobility_Speed_kmph=speed, Indoor_Outdoor=indoor, Weather=weather, Event_Context=event)
X = pd.DataFrame([row])[COLS]
for c in LV:
    X[c] = pd.Categorical(X[c], categories=LV[c])

# ---------- Predict ----------
score = float(np.clip(reg.predict(X)[0], 1, 5))
proba = clf.predict_proba(X)[0]
classes = list(clf.classes_)
best = classes[int(np.argmax(proba))]
color = CLASS_COLOR[best]

tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Model insights"])

with tab1:
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown(f"""
        <div class="card" style="text-align:center">
          <div class="small">PREDICTED QUALITY CLASS</div>
          <div style="font-size:4rem; margin:6px 0">{CLASS_EMOJI[best]}</div>
          <span class="badge" style="background:{color}22; color:{color}; border:2px solid {color}">{best.upper()}</span>
          <p style="margin-top:16px; font-size:1.05rem">{CLASS_MSG[best]}</p>
          <div class="small">Confidence: {proba.max()*100:.0f}% &nbsp;•&nbsp; {nice(app)} • {nice(area)} • {nice(event)}</div>
        </div>""", unsafe_allow_html=True)
    with right:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score, number={"suffix": " / 5", "font": {"size": 38}},
            title={"text": "Predicted QoE Score", "font": {"size": 16}},
            gauge={"axis": {"range": [1, 5]}, "bar": {"color": color, "thickness": 0.28},
                   "bgcolor": "#131a2b",
                   "steps": [{"range": [1, 2.2], "color": "#3b1620"}, {"range": [2.2, 3.2], "color": "#3d2c14"},
                             {"range": [3.2, 4.2], "color": "#16331f"}, {"range": [4.2, 5], "color": "#0f3340"}]}))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)",
                          font={"color": "#e8ecf5"})
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### Kis class ka kitna chance?")
    order = ["Poor", "Fair", "Good", "Excellent"]
    p = {c: proba[classes.index(c)] * 100 for c in order}
    bar = go.Figure(go.Bar(x=list(p.values()), y=order, orientation="h",
                           marker_color=[CLASS_COLOR[c] for c in order],
                           text=[f"{v:.1f}%" for v in p.values()], textposition="outside"))
    bar.update_layout(height=250, margin=dict(l=10, r=40, t=10, b=10), xaxis=dict(range=[0, 105], visible=False),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#e8ecf5"})
    st.plotly_chart(bar, width="stretch")

    # Quick tips
    tips = []
    if load > 75: tips.append("Cell load bahut high hai, ye experience girane ka bada reason hai.")
    if jitter > 50: tips.append("Jitter zyada hai, connection shaky rahega.")
    if users > 180: tips.append("Cell mein bohot users hain, crowding ka asar padega.")
    if app in ("AR_VR", "Cloud_Gaming"): tips.append(f"{nice(app)} naturally sabse demanding apps mein se hai.")
    if tips:
        st.markdown("#### 💡 Dhyaan dene wali baatein")
        for t in tips: st.info(t)

with tab2:
    m = meta["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    for col, v, l in [(c1, f"{m['rows']:,}", "Training rows"), (c2, f"{m['r2']:.2f}", "R² (QoE score)"),
                      (c3, f"{m['acc']*100:.1f}%", "Class accuracy"), (c4, f"{m['f1']:.2f}", "Macro F1")]:
        col.markdown(f'<div class="kpi"><b>{v}</b><span>{l}</span></div>', unsafe_allow_html=True)

    imp = meta["importance"]
    top = list(imp.items())[:10][::-1]
    fi = go.Figure(go.Bar(x=[v for _, v in top], y=[nice(k) for k, _ in top], orientation="h",
                          marker=dict(color=[v for _, v in top], colorscale="Purp")))
    fi.update_layout(title="Model ke liye sabse important columns", height=420,
                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#e8ecf5"})
    st.plotly_chart(fi, width="stretch")
    st.caption("Model: HistGradientBoosting. Application type, jitter aur cell crowding experience ko sabse zyada "
               "decide karte hain. Signal aur latency ka dataset mein koi khaas asar nahi dikha, kyunki data simulated lagta hai.")
