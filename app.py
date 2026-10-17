"""SignalScope: a 5G experience predictor for Indian networks."""
import json
import math

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SignalScope | 5G Experience Predictor", page_icon="📶",
                   layout="wide", initial_sidebar_state="collapsed")


# ───────────────────────────── model + lookups ─────────────────────────────
@st.cache_resource
def load():
    return joblib.load("model.pkl"), json.load(open("meta.json"))


bundle, meta = load()
reg, COLS = bundle["reg"], bundle["columns"]
clf = bundle["clf"]
LV, RG = meta["levels"], meta["ranges"]

ORDER = ["Poor", "Fair", "Good", "Excellent"]
CCOL = {"Poor": "#FF4D4D", "Fair": "#FF9F1C", "Good": "#2EC27E", "Excellent": "#2F6BFF"}
BANDS = [(1, 2.2, "Poor"), (2.2, 3.2, "Fair"), (3.2, 4.2, "Good"), (4.2, 5.0, "Excellent")]

APPS = {  # key: (emoji, label, what it needs)
    "Video_Streaming": ("🎬", "Video streaming", "Needs steady speed so the picture stays sharp and never buffers."),
    "Live_Sports_Streaming": ("🏏", "Live sports", "Needs steady speed and low delay, especially during big matches."),
    "Social_Media": ("📱", "Social media", "Light usage. Feeds, reels and photo uploads."),
    "Web_Browsing": ("🌐", "Web browsing", "Light usage. Pages should open quickly."),
    "Video_Call": ("📹", "Video calls", "Needs a stable connection with very little jitter."),
    "Voice_Call": ("📞", "Voice calls", "Needs a stable connection so the audio does not break up."),
    "Cloud_Gaming": ("🎮", "Cloud gaming", "Very demanding. Needs low delay and a rock-steady connection."),
    "AR_VR": ("🥽", "AR / VR", "The most demanding activity. Needs high speed, low delay and no jitter."),
    "File_Download": ("⬇️", "Big downloads", "Needs sustained speed for large files and app updates."),
    "IoT_Telemetry": ("📡", "IoT & sensors", "Tiny data packets from smart meters, trackers and devices."),
}
AREAS = {"Dense_Urban": ("🏙️", "Dense city"), "Urban": ("🏢", "City"), "Suburban": ("🏘️", "Suburb"),
         "Rural": ("🌾", "Village / rural"), "Highway": ("🛣️", "Highway"), "Railway_Corridor": ("🚆", "Railway line")}
EVENTS = {"Normal": ("🗓️", "Regular day"), "Office_Rush": ("💼", "Office rush hour"), "IPL_Match": ("🏏", "IPL match"),
          "Festival": ("🪔", "Festival"), "Concert": ("🎤", "Concert"), "Election_Rally": ("📣", "Election rally")}
WEATHER = {"Clear": "☀️ Clear", "Cloudy": "☁️ Cloudy", "Fog": "🌫️ Fog", "Heavy_Rain": "🌧️ Heavy rain"}
DEVICES = {"Budget_5G": "📱 Budget 5G phone", "Mid_5G": "📱 Mid-range 5G phone", "Flagship_5G": "📱 Flagship 5G phone",
           "FWA_Router": "📶 5G home router", "IoT_Module": "📟 IoT module"}
PLANS = {"Prepaid": "Prepaid", "Postpaid": "Postpaid", "Enterprise": "Business"}
TIERS = {"Metro": "Metro", "Tier_2": "Tier 2 city", "Tier_3": "Tier 3 town"}
MODES = {"5G_SA": "Standalone (SA)", "5G_NSA": "Non-standalone (NSA)"}
INOUT = {"Indoor": "🏠 Indoors", "Outdoor": "🌳 Outdoors"}
CROWD = {"Quiet": ("😌", "Few people around, such as early morning, a village or off-peak hours."),
         "Normal": ("🙂", "A typical day with moderate traffic on the tower."),
         "Busy": ("😬", "Lots of users, such as evenings, markets or the daily commute."),
         "Packed": ("🥵", "A very heavy crowd, such as a stadium, festival or rush hour.")}
SIGNAL = {"Weak": ("🔴", -105), "Fair": ("🟠", -90), "Good": ("🟢", -75), "Excellent": ("🔵", -55)}
MOVE = {"Standing still": ("🧍", 0), "Walking": ("🚶", 5), "City driving": ("🚗", 35), "Train / highway": ("🚆", 90)}
OPS = {"Airtel": "Bharti Airtel", "Jio": "Reliance Jio", "Vi": "Vodafone Idea"}
nice = lambda s: s.replace("_", " ")

LOGOS = {
    "Airtel": '<svg viewBox="0 0 120 120"><rect width="120" height="120" rx="30" fill="#E40000"/>'
              '<text x="60" y="84" text-anchor="middle" font-family="Arial Black,Arial,sans-serif" font-weight="900" '
              'font-size="84" fill="#fff">a</text>'
              '<path d="M24 98 Q60 112 96 98" stroke="#fff" stroke-width="5" fill="none" stroke-linecap="round" opacity=".9"/></svg>',
    "Jio": '<svg viewBox="0 0 120 120"><rect width="120" height="120" rx="30" fill="#0A2885"/>'
           '<text x="60" y="76" text-anchor="middle" font-family="Arial Black,Arial,sans-serif" font-weight="900" '
           'font-size="50" fill="#fff" letter-spacing="-2">jio</text>'
           '<circle cx="96" cy="30" r="7" fill="#E50914"/></svg>',
    "Vi": '<svg viewBox="0 0 120 120"><rect width="120" height="120" rx="30" fill="#FFD100"/>'
          '<text x="58" y="86" text-anchor="middle" font-family="Arial Black,Arial,sans-serif" font-weight="900" '
          'font-size="76" fill="#E4002B">V</text>'
          '<text x="88" y="86" text-anchor="middle" font-family="Arial Black,Arial,sans-serif" font-weight="900" '
          'font-size="52" fill="#1a1a1a">i</text></svg>',
}


def H(s):
    """Collapse HTML to one line so markdown never treats it as a code block."""
    return " ".join(line.strip() for line in s.strip().splitlines())


def show(html):
    st.markdown(H(html), unsafe_allow_html=True)


# ───────────────────────────── styling ─────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=DM+Sans:wght@400;500;700&display=swap');
:root{--bg:#F7F0E3;--ink:#15131C;--yellow:#FFD23F;--coral:#FF6B4A;--mint:#3DDC97;--blue:#2F6BFF;--soft:#5b5670;
--head:'Bricolage Grotesque','DM Sans',system-ui,sans-serif;}
.stApp{background-color:var(--bg);background-image:radial-gradient(rgba(21,19,28,.10) 1.2px,transparent 1.2px);background-size:22px 22px;}
html,body,.stApp,[data-testid="stMarkdownContainer"],.stApp button,.stApp input,.stApp label{font-family:'DM Sans',system-ui,sans-serif;color:var(--ink);}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stHeader"]{display:none!important}
.block-container{max-width:1180px;padding:1.4rem 1.4rem 4rem}
h1,h2,h3{font-family:var(--head)!important;letter-spacing:-.02em;color:var(--ink)}

/* hero */
.hero{display:flex;align-items:center;gap:30px;margin:.4rem 0 2.2rem}
.hero-l{flex:1.15;min-width:0}
.chip{display:inline-block;background:#fff;border:2.5px solid var(--ink);border-radius:999px;padding:7px 16px;font-weight:700;font-size:.85rem;box-shadow:3px 3px 0 var(--ink);margin-bottom:18px}
.hero h1{font-size:3.5rem;line-height:1.02;margin:0 0 16px;font-weight:800}
.hl{background:var(--yellow);padding:0 .18em;border-radius:12px;box-decoration-break:clone;-webkit-box-decoration-break:clone;border:2.5px solid var(--ink);display:inline-block;transform:rotate(-1.5deg)}
.hero p{font-size:1.12rem;line-height:1.6;color:#3a3650;max-width:560px;margin:0 0 20px}
.pillrow{display:flex;flex-wrap:wrap;gap:10px}
.pillrow span{background:var(--mint);border:2.5px solid var(--ink);border-radius:12px;padding:7px 13px;font-weight:700;font-size:.88rem;box-shadow:3px 3px 0 var(--ink)}
.pillrow span:nth-child(2){background:#FFB8D2}.pillrow span:nth-child(3){background:#BFD4FF}
.hero-r{flex:1;min-width:0;display:flex;justify-content:center}
.scene{position:relative;width:420px;height:430px;perspective:1100px;flex:none}
.ring{position:absolute;left:55px;top:90px;width:60px;height:60px;border-radius:50%;border:3px solid var(--ink);opacity:0;animation:ring 3.6s infinite ease-out}
.r2{animation-delay:1.2s}.r3{animation-delay:2.4s}
@keyframes ring{0%{transform:scale(.4);opacity:.9}100%{transform:scale(6.2);opacity:0}}
.tower{position:absolute;left:30px;bottom:34px;width:110px;height:300px}
.phone{position:absolute;right:14px;top:34px;width:228px;height:372px;background:var(--ink);border-radius:38px;padding:11px;
transform:rotateY(-18deg) rotateX(6deg) rotateZ(2deg);transform-style:preserve-3d;box-shadow:14px 18px 0 rgba(21,19,28,.22);animation:floaty 5.5s ease-in-out infinite}
@keyframes floaty{0%,100%{transform:rotateY(-18deg) rotateX(6deg) rotateZ(2deg) translateY(0)}50%{transform:rotateY(-10deg) rotateX(3deg) rotateZ(2deg) translateY(-14px)}}
.screen{height:100%;background:#fff;border-radius:28px;padding:20px 16px;display:flex;flex-direction:column;gap:10px;position:relative;overflow:hidden}
.notch{position:absolute;top:9px;left:50%;transform:translateX(-50%);width:62px;height:14px;background:var(--ink);border-radius:9px}
.ph-k{margin-top:14px;font-size:.66rem;letter-spacing:.12em;font-weight:700;color:var(--soft)}
.ph-v{font-family:var(--head);font-size:2.05rem;font-weight:800;color:var(--blue);line-height:1}
.ph-s{font-size:.8rem;font-weight:700}
.ph-bar{height:11px;border-radius:7px;background:#ecebf3;overflow:hidden}.ph-bar i{display:block;height:100%;border-radius:7px}
.ph-row{display:flex;justify-content:space-between;font-size:.68rem;font-weight:700;margin-bottom:3px}
.stk{position:absolute;font-size:2.1rem;background:#fff;border:2.5px solid var(--ink);border-radius:16px;width:58px;height:58px;display:flex;align-items:center;justify-content:center;box-shadow:4px 4px 0 var(--ink);animation:bob 4.5s ease-in-out infinite}
.s1{left:150px;top:6px;transform:rotate(-8deg);background:var(--yellow)}.s2{left:150px;top:214px;transform:rotate(9deg);animation-delay:.8s;background:#FFB8D2}
.s3{left:104px;bottom:6px;transform:rotate(7deg);animation-delay:1.6s;background:#BFD4FF}
@keyframes bob{0%,100%{translate:0 0}50%{translate:0 -9px}}
.stamp{position:absolute;left:0;top:18px;transform:rotate(-12deg);border:3px solid var(--coral);color:var(--coral);border-radius:14px;padding:6px 11px;font-family:var(--head);font-weight:800;font-size:.8rem;line-height:1.15;text-align:center;background:rgba(255,255,255,.7)}

/* how it works */
.how{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin:0 0 2.4rem}
.how div.c{background:#fff;border:2.5px solid var(--ink);border-radius:20px;padding:20px 22px;box-shadow:6px 6px 0 var(--ink)}
.tilt{transition:transform .35s cubic-bezier(.2,.8,.2,1),box-shadow .35s}
.tilt:hover{transform:perspective(900px) rotateX(5deg) rotateY(-7deg) translateY(-5px);box-shadow:12px 12px 0 var(--ink)!important}
.how .em{font-size:2rem;display:inline-flex;width:56px;height:56px;border-radius:16px;border:2.5px solid var(--ink);align-items:center;justify-content:center;margin-bottom:10px}
.how h3{margin:0 0 4px;font-size:1.2rem}.how p{margin:0;color:#4a4660;font-size:.96rem;line-height:1.5}

/* section titles */
.sec{display:flex;gap:14px;align-items:center;margin:0 0 6px}
.sec .num{flex:none;width:46px;height:46px;border-radius:14px;background:var(--yellow);border:2.5px solid var(--ink);box-shadow:3px 3px 0 var(--ink);display:flex;align-items:center;justify-content:center;font-family:var(--head);font-weight:800;font-size:1.35rem;transform:rotate(-5deg)}
.sec h2{margin:0;font-size:1.6rem;line-height:1.1}.sec p{margin:2px 0 0;color:var(--soft);font-size:.95rem}
.title{font-family:var(--head);font-weight:800;font-size:2rem;margin:2.2rem 0 .3rem}

/* step cards */
[class*="st-key-card"]{background:#fff;border:2.5px solid var(--ink);border-radius:24px;padding:22px 26px 18px;box-shadow:7px 7px 0 var(--ink);margin-bottom:26px;gap:.9rem}
[data-testid="stWidgetLabel"] p{font-weight:700;font-size:.78rem;text-transform:uppercase;letter-spacing:.07em;color:#4a4660}
[data-baseweb="select"]>div,[data-baseweb="input"],[data-baseweb="base-input"]{border-radius:12px!important;border:2.5px solid var(--ink)!important;background:#fff!important;min-height:46px}
[data-baseweb="input"] input{font-weight:700}
[data-testid="stNumberInput"] button{border-radius:10px}

/* pills / segmented */
button[data-variant="pills"],button[data-variant="segmented_control"]{border:2.5px solid var(--ink)!important;border-radius:999px!important;background:#fff!important;box-shadow:2px 2px 0 var(--ink)!important;transition:all .15s;padding:.3rem .9rem!important;min-height:38px}
button[data-variant="pills"] *,button[data-variant="segmented_control"] *{color:var(--ink)!important;font-weight:700}
button[data-variant="pills"]:hover,button[data-variant="segmented_control"]:hover{transform:translate(-1px,-1px);box-shadow:4px 4px 0 var(--ink)!important;background:#FFF3C4!important}
button[data-variant="pills"][aria-checked="true"],button[data-variant="segmented_control"][aria-checked="true"]{background:var(--yellow)!important;box-shadow:3px 3px 0 var(--ink)!important}
button[data-variant="pills"][aria-checked="true"] *,button[data-variant="segmented_control"][aria-checked="true"] *{font-weight:800}
[data-testid="stButtonGroup"]{overflow:visible!important}
[role="radiogroup"]{flex-wrap:wrap!important;overflow:visible!important;-webkit-mask-image:none!important;mask-image:none!important;gap:.6rem!important;padding:2px 6px 8px 2px!important}

/* buttons */
[data-testid="stNumberInputContainer"]{border:2.5px solid var(--ink)!important;border-radius:12px!important;background:#fff!important;overflow:hidden}
[data-testid="stNumberInputContainer"] [data-baseweb]{border:0!important}
button[data-testid="stBaseButton-secondary"],button[data-testid="stBaseButton-primary"],.stButton>button{border:2.5px solid var(--ink);border-radius:14px;font-weight:800;background:#fff;color:var(--ink);box-shadow:4px 4px 0 var(--ink);min-height:46px;transition:all .15s}
.stButton>button:hover{transform:translate(-2px,-2px);box-shadow:6px 6px 0 var(--ink);border-color:var(--ink);color:var(--ink);background:#fff}
.stButton>button:active{transform:translate(3px,3px);box-shadow:1px 1px 0 var(--ink)}
.stButton>button[kind="primary"]{background:var(--mint)}
.st-key-cta button{background:var(--yellow)!important;font-family:var(--head);font-size:1.35rem;min-height:66px;border-radius:18px;box-shadow:6px 6px 0 var(--ink)}
.st-key-cta button:hover{background:#FFE07A!important}

/* operator cards */
.op{background:#fff;border:2.5px solid var(--ink);border-radius:20px;padding:16px 10px 14px;text-align:center;box-shadow:5px 5px 0 var(--ink);position:relative;transition:transform .3s cubic-bezier(.2,.8,.2,1),box-shadow .3s;margin-bottom:10px}
.op svg{width:76px;height:76px;filter:drop-shadow(3px 4px 0 rgba(21,19,28,.25))}
.op b{display:block;font-family:var(--head);font-size:1.25rem;margin-top:8px}.op small{color:var(--soft);font-weight:500}
.op:hover{transform:perspective(700px) rotateY(-9deg) translateY(-4px);box-shadow:9px 9px 0 var(--ink)}
.op.sel{background:#E5FFF2;transform:perspective(700px) rotateY(-7deg) translateY(-6px);box-shadow:9px 9px 0 var(--ink)}
.op.sel:after{content:"✓";position:absolute;right:-10px;top:-12px;width:34px;height:34px;border-radius:50%;background:var(--mint);border:2.5px solid var(--ink);font-weight:800;display:flex;align-items:center;justify-content:center;transform:rotate(8deg)}
.st-key-ops [data-testid="stHorizontalBlock"]{flex-wrap:nowrap!important;gap:1rem}
.st-key-ops [data-testid="stColumn"]{min-width:0!important;flex:1 1 0!important}

/* activity preview */
.pv{background:#FFF3C4;border:2.5px solid var(--ink);border-radius:20px;padding:16px 18px;display:flex;gap:14px;align-items:center;box-shadow:4px 4px 0 var(--ink)}
.pv .big{font-size:2.6rem;width:70px;height:70px;border-radius:18px;background:#fff;border:2.5px solid var(--ink);display:flex;align-items:center;justify-content:center;flex:none;transform:rotate(-6deg)}
.pv b{font-family:var(--head);font-size:1.15rem}.pv span{display:block;color:#4a4660;font-size:.9rem;line-height:1.4}
.note{background:#fff;border:2px dashed var(--ink);border-radius:14px;padding:10px 14px;font-size:.9rem;color:#3a3650}
.warn{background:#FFE9A8;border:2.5px solid var(--ink);border-radius:14px;padding:10px 14px;font-weight:700;font-size:.92rem;margin-bottom:14px}

/* results */
.res{display:grid;grid-template-columns:1.1fr 1fr;gap:22px;margin-bottom:22px}
.rc{background:#fff;border:2.5px solid var(--ink);border-radius:24px;padding:24px;box-shadow:7px 7px 0 var(--ink);min-width:0}
.rc h3{margin:0 0 4px;font-size:1.2rem}.rc .sub{color:var(--soft);font-size:.9rem;margin-bottom:14px}
.ctx{display:flex;align-items:center;gap:12px;margin-bottom:14px}.ctx svg{width:44px;height:44px;flex:none}
.ctx span{font-size:.88rem;color:#3a3650;line-height:1.35}
.badge{display:inline-block;font-family:var(--head);font-weight:800;font-size:2.5rem;padding:4px 22px;border-radius:16px;border:3px solid var(--ink);box-shadow:5px 5px 0 var(--ink);transform:rotate(-2deg);margin:4px 0 14px}
.verdict{font-size:1.08rem;line-height:1.55;margin:0 0 12px}
.dial{display:block;width:100%;max-width:340px;margin:0 auto}
.needle{transform-origin:150px 150px;animation:sweep 1.4s cubic-bezier(.2,.9,.25,1.1) both}
@keyframes sweep{from{transform:rotate(0deg)}to{transform:rotate(var(--a))}}
.legend{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:10px}
.legend span{font-size:.78rem;font-weight:700;border:2px solid var(--ink);border-radius:999px;padding:3px 10px}
.row{display:grid;grid-template-columns:128px 1fr 48px;gap:12px;align-items:center;margin:10px 0;font-size:.94rem}
.row .lab{display:flex;align-items:center;gap:8px;font-weight:700;min-width:0}.row .lab svg{width:22px;height:22px;flex:none}
.trk{height:16px;border-radius:9px;background:#ECEBF3;border:2px solid var(--ink);overflow:hidden}
.trk i{display:block;height:100%;border-right:2px solid var(--ink);animation:grow .9s cubic-bezier(.2,.8,.2,1) both;transform-origin:left}
@keyframes grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
.row .val{font-weight:800;text-align:right}
.you{background:var(--ink);color:#fff;border-radius:7px;font-size:.65rem;padding:1px 6px;font-weight:700}
.row.cur{background:#FFF3C4;border-radius:12px;margin-left:-8px;margin-right:-8px;padding:4px 8px}
.r2{margin:12px 0;padding:2px 0}.r2 .top{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;font-weight:700;font-size:.94rem;gap:8px}.r2 .lab{display:flex;align-items:center;gap:8px;min-width:0}.r2 .lab svg{width:22px;height:22px;flex:none}.r2 .val{font-weight:800}.r2.cur{background:#FFF3C4;border-radius:12px;margin-left:-8px;margin-right:-8px;padding:6px 8px}
.tips{list-style:none;padding:0;margin:0}.tips li{display:flex;gap:12px;align-items:center;padding:11px 0;border-bottom:2px dashed #d8d3e6;font-weight:500}
.tips li:last-child{border:0}.tips .d{flex:none;background:var(--mint);border:2.5px solid var(--ink);border-radius:10px;padding:3px 9px;font-weight:800;font-size:.85rem;box-shadow:2px 2px 0 var(--ink)}
.cmp{display:grid;grid-template-columns:repeat(3,1fr);gap:22px;margin-bottom:22px}
.about{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-bottom:18px}
.about p,.about li{color:#3a3650;line-height:1.6;font-size:.96rem}.about ul{padding-left:1.1rem;margin:.3rem 0 0}
details{background:#fff;border:2.5px solid var(--ink);border-radius:14px;padding:12px 16px;margin:10px 0;box-shadow:3px 3px 0 var(--ink)}
details summary{font-weight:800;cursor:pointer}details p{margin:8px 0 0;color:#3a3650;line-height:1.55}
.foot{text-align:center;color:var(--soft);font-size:.88rem;margin-top:2.4rem}

@media (max-width:900px){
 .hero{flex-direction:column;align-items:stretch}.hero h1{font-size:2.5rem}.hero-r{justify-content:center}
 .res,.cmp,.about,.how{grid-template-columns:1fr}
}
@media (max-width:640px){
 .block-container{padding:.8rem .8rem 3rem}.hero h1{font-size:2.05rem}.hero p{font-size:1rem}
 .scene{transform:scale(.78);transform-origin:top center;height:340px;margin-bottom:-70px}
 [class*="st-key-card"]{padding:16px 14px 12px;box-shadow:5px 5px 0 var(--ink);border-radius:20px}
 .op{padding:10px 4px;border-radius:16px}.op svg{width:48px;height:48px}.op b{font-size:1rem}.op small{display:none}
 .st-key-ops [data-testid="stHorizontalBlock"]{gap:.5rem}
 .st-key-ops button{font-size:.78rem;padding:.2rem .3rem}
 .sec h2{font-size:1.3rem}.sec .num{width:40px;height:40px}.title{font-size:1.6rem}
 .row{grid-template-columns:96px 1fr 40px;gap:8px;font-size:.86rem}
 .badge{font-size:2rem}.rc{padding:18px;box-shadow:5px 5px 0 var(--ink)}
}
</style>
"""
st.markdown(H(CSS), unsafe_allow_html=True)

# ───────────────────────────── state ─────────────────────────────
D = dict(operator="Airtel", device="Mid_5G", plan="Prepaid", mode="5G_SA", app="Video_Streaming", circle="Mumbai",
         tier="Metro", area="Urban", indoor="Indoor", weather="Clear", event="Normal", entry="Simple",
         crowd="Normal", signal="Good", move="Walking")
for k, v in D.items():
    st.session_state.setdefault(k, v)

XF = {  # exact-value fields: column -> (state key, label, help, step, format)
    "Signal_dBm": ("x_sig", "Signal strength (dBm)", "Closer to -40 is stronger. Below -100 is weak.", 1.0, "%.0f"),
    "Latency_ms": ("x_lat", "Latency (ms)", "Round-trip delay. Lower is better.", 1.0, "%.0f"),
    "Jitter_ms": ("x_jit", "Jitter (ms)", "How much the delay fluctuates. Lower is better.", 1.0, "%.1f"),
    "Cell_Load_pct": ("x_load", "Tower load (%)", "How busy the nearby tower is.", 1.0, "%.0f"),
    "Active_Users_In_Cell": ("x_users", "Active users on tower", "People connected to the same tower.", 1, "%d"),
    "Mobility_Speed_kmph": ("x_mob", "Your speed (km/h)", "0 if you are standing still.", 1.0, "%.0f"),
    "Req_Mbps": ("x_req", "Speed you need (Mbps)", "Typical need for your activity is pre-filled.", 0.5, "%.2f"),
    "Resource_Alloc_pct": ("x_alloc", "Resources given to you (%)", "Share of tower capacity allocated to you.", 1.0, "%.0f"),
}


def _cast(col, v):
    lo, hi, _ = RG[col]
    v = min(max(float(v), lo), hi)
    return int(round(v)) if col == "Active_Users_In_Cell" else round(v, 2)


for col, (k, *_r) in XF.items():
    st.session_state.setdefault(k, _cast(col, meta["global_typical"][col]))


def load_typical():
    t = meta["app_typical"][st.session_state.app]
    for col, (k, *_r) in XF.items():
        st.session_state[k] = _cast(col, t[col])


def keep(key):
    """If a pill is un-clicked, restore the last choice so nothing is ever blank."""
    if st.session_state.get(key) is None:
        st.session_state[key] = st.session_state.get("_last_" + key, D[key])
    else:
        st.session_state["_last_" + key] = st.session_state[key]


def pills(label, options, key, fmt=None):
    return st.pills(label, options, selection_mode="single", format_func=fmt or str, key=key,
                    on_change=keep, args=(key,)) or st.session_state.get("_last_" + key, D[key])


def reset_all():
    for k in list(st.session_state.keys()):
        del st.session_state[k]


# ───────────────────────────── model helpers ─────────────────────────────
def frame(rows):
    X = pd.DataFrame(rows)[COLS]
    for c in LV:
        X[c] = pd.Categorical(X[c], categories=LV[c])
    return X


def scores(rows):
    return np.clip(reg.predict(frame(rows)), 1, 5)


def band(s):
    return next(n for lo, hi, n in BANDS if s < hi or n == "Excellent")


def dial(score):
    cx, cy, r = 150, 150, 108

    def pt(deg, rad=r):
        a = math.radians(deg)
        return cx + rad * math.cos(a), cy - rad * math.sin(a)

    arcs = ""
    for lo, hi, name in BANDS:
        a1 = 180 - (lo - 1) / 4 * 180 - 1.2
        a2 = 180 - (hi - 1) / 4 * 180 + 1.2
        (x1, y1), (x2, y2) = pt(a1), pt(a2)
        arcs += (f'<path d="M{x1:.1f} {y1:.1f} A{r} {r} 0 0 1 {x2:.1f} {y2:.1f}" fill="none" stroke="{CCOL[name]}" '
                 f'stroke-width="26"/>')
    ang = (score - 1) / 4 * 180
    return (f'<svg class="dial" viewBox="0 0 300 200" role="img" aria-label="Score {score:.1f} out of 5">{arcs}'
            f'<g class="needle" style="--a:{ang:.1f}deg"><path d="M150 150 L{cx - 96} {cy}" stroke="#15131C" stroke-width="7" '
            f'stroke-linecap="round"/></g><circle cx="150" cy="150" r="13" fill="#15131C"/><circle cx="150" cy="150" r="5" fill="#FFD23F"/>'
            f'<text x="150" y="190" text-anchor="middle" font-family="Bricolage Grotesque,Arial Black,sans-serif" font-weight="800" '
            f'font-size="34" fill="#15131C">{score:.1f}<tspan font-size="17" fill="#5b5670"> / 5</tspan></text></svg>')


def bar_rows(items, current, label_fn, icon_fn=None):
    """Stacked rows: name and score on top, bar below. Bars are scaled to the spread so differences stay visible."""
    vals = [v for _, v in items]
    lo, hi = min(vals) - 0.3, max(vals) + 0.1
    out = ""
    for k, s in items:
        c = CCOL[band(s)]
        icon = icon_fn(k) if icon_fn else ""
        cur = " cur" if k == current else ""
        tag = ' <span class="you">YOU</span>' if k == current else ""
        w = max(8, (s - lo) / (hi - lo) * 100)
        out += (f'<div class="r2{cur}"><div class="top"><span class="lab">{icon}<span>{label_fn(k)}</span>{tag}</span>'
                f'<span class="val">{s:.1f}</span></div><div class="trk"><i style="width:{w:.0f}%;background:{c}"></i></div></div>')
    return out


CROWD_COLS = ("Cell_Load_pct", "Active_Users_In_Cell", "Jitter_ms")


def switch(row, col, new):
    """Change an area or event and shift crowding by how much that option typically changes it."""
    old_t, new_t = meta["by"][col][row[col]], meta["by"][col][new]
    r = {**row, col: new}
    for c in CROWD_COLS:
        lo, hi, _ = RG[c]
        v = min(max(row[c] + new_t[c] - old_t[c], lo), hi)
        r[c] = int(round(v)) if c == "Active_Users_In_Cell" else float(v)
    return r


def levers(row, base):
    found = []

    def trial(text, new_row):
        found.append((scores([new_row])[0] - base, text))

    q = meta["crowd"]["Quiet"]
    if row["Cell_Load_pct"] > q["Cell_Load_pct"] + 5:
        trial("Connect when the tower is quiet, such as early morning or off-peak hours.",
              {**row, **{k: q[k] for k in CROWD_COLS}})
    if row["Event_Context"] != "Normal":
        trial("Avoid big-event hours like match days, festivals and rallies.", switch(row, "Event_Context", "Normal"))
    if row["Indoor_Outdoor"] == "Indoor":
        trial("Step outdoors or next to a window if you can.", {**row, "Indoor_Outdoor": "Outdoor"})
    if row["Network_Mode"] == "5G_NSA":
        trial("Use a Standalone (SA) 5G network if your operator offers it.", {**row, "Network_Mode": "5G_SA"})
    areas = list(AREAS)
    sa = scores([switch(row, "Area_Type", a) for a in areas])
    top = areas[int(np.argmax(sa))]
    if top != row["Area_Type"]:
        trial(f"Move to a {AREAS[top][1].lower()} location if possible.", switch(row, "Area_Type", top))
    return sorted([f for f in found if f[0] >= 0.03], reverse=True)[:4]


# ───────────────────────────── hero ─────────────────────────────
show("""
<div class="hero">
 <div class="hero-l">
  <div class="chip">📶 Free to use · No sign-up · Trained on 50,000 5G sessions</div>
  <h1>Know how your 5G will perform <span class="hl">before</span> you connect.</h1>
  <p>Choose your operator, what you plan to do and where you will be. SignalScope predicts the experience you can expect,
  scores it out of 5, and shows where, when and how you can get a better connection.</p>
  <div class="pillrow"><span>🎯 Quality rating</span><span>⭐ Score out of 5</span><span>🧭 Best place &amp; time</span></div>
 </div>
 <div class="hero-r"><div class="scene">
  <div class="ring"></div><div class="ring r2"></div><div class="ring r3"></div>
  <div class="stamp">PREDICT<br>BEFORE YOU<br>CONNECT</div>
  <svg class="tower" viewBox="0 0 110 300"><g stroke="#15131C" stroke-width="5" fill="none" stroke-linejoin="round" stroke-linecap="round">
   <path d="M55 24 L14 292 M55 24 L96 292 M33 160 L77 160 M24 220 L86 220 M40 100 L70 100 M33 160 L86 220 M77 160 L24 220 M40 100 L77 160 M70 100 L33 160"/>
   </g><circle cx="55" cy="22" r="11" fill="#FF6B4A" stroke="#15131C" stroke-width="5"/></svg>
  <div class="phone"><div class="screen"><div class="notch"></div>
   <div class="ph-k">SAMPLE RESULT</div><div class="ph-v">Excellent</div><div class="ph-s">4.6 / 5 · Video streaming</div>
   <div><div class="ph-row"><span>Speed</span><span>High</span></div><div class="ph-bar"><i style="width:88%;background:#2F6BFF"></i></div></div>
   <div><div class="ph-row"><span>Stability</span><span>High</span></div><div class="ph-bar"><i style="width:82%;background:#2EC27E"></i></div></div>
   <div><div class="ph-row"><span>Crowding</span><span>Low</span></div><div class="ph-bar"><i style="width:24%;background:#FFD23F"></i></div></div>
   <div style="margin-top:auto;background:#FFD23F;border:2.5px solid #15131C;border-radius:12px;text-align:center;font-weight:800;padding:9px;font-size:.82rem">Check my network →</div>
  </div></div>
  <div class="stk s1">📶</div><div class="stk s2">⚡</div><div class="stk s3">🏏</div>
 </div></div>
</div>
<div class="how">
 <div class="c tilt"><div class="em" style="background:#FFD23F">📝</div><h3>1. Describe your situation</h3><p>Pick your operator, activity, location and how crowded it is. It takes about a minute.</p></div>
 <div class="c tilt"><div class="em" style="background:#FFB8D2">🎯</div><h3>2. Get your prediction</h3><p>See a clear quality rating and a score out of 5 for that exact situation.</p></div>
 <div class="c tilt"><div class="em" style="background:#BFD4FF">🧭</div><h3>3. Compare and improve</h3><p>Find out which place, time and operator work best, and what to change.</p></div>
</div>
""")

# ───────────────────────────── step 1 ─────────────────────────────
with st.container(key="card_1"):
    show('<div class="sec"><div class="num">1</div><div><h2>Choose your network</h2><p>Select your mobile operator and tell us about your device.</p></div></div>')
    with st.container(key="ops"):
        cols = st.columns(3)
        for c, op in zip(cols, OPS):
            sel = st.session_state.operator == op
            with c:
                show(f'<div class="op{" sel" if sel else ""}">{LOGOS[op]}<b>{op}</b><small>{OPS[op]}</small></div>')
                st.button("Selected" if sel else "Select", key=f"op_{op}", width="stretch",
                          type="primary" if sel else "secondary",
                          on_click=lambda o=op: st.session_state.update(operator=o))
    a, b, c = st.columns([1.5, 1, 1.1])
    with a:
        device = pills("Device", list(DEVICES), "device", DEVICES.get)
    with b:
        plan = pills("Plan type", list(PLANS), "plan", PLANS.get)
    with c:
        mode = pills("5G network type", list(MODES), "mode", MODES.get)
    show('<div class="note">Not sure about the 5G network type? Leave it as it is. '
         'Standalone (SA) is a pure 5G network, while Non-standalone (NSA) also leans on 4G.</div>')

# ───────────────────────────── step 2 ─────────────────────────────
with st.container(key="card_2"):
    show('<div class="sec"><div class="num">2</div><div><h2>What will you do?</h2><p>Different activities need different things from the network.</p></div></div>')
    a, b = st.columns([1.7, 1])
    with a:
        app = pills("Activity", list(APPS), "app", lambda k: f"{APPS[k][0]} {APPS[k][1]}")
    with b:
        e, l, need = APPS[app]
        show(f'<div class="pv"><div class="big">{e}</div><div><b>{l}</b><span>{need}</span></div></div>')

# ───────────────────────────── step 3 ─────────────────────────────
with st.container(key="card_3"):
    show('<div class="sec"><div class="num">3</div><div><h2>Where and when?</h2><p>Location and timing change how busy the network is.</p></div></div>')
    a, b, c = st.columns([1, 1, 1])
    with a:
        circle = st.selectbox("Telecom circle", LV["Circle"], key="circle", format_func=nice)
    with b:
        tier = pills("City size", list(TIERS), "tier", TIERS.get)
    with c:
        indoor = pills("You will be", list(INOUT), "indoor", INOUT.get)
    area = pills("Type of area", list(AREAS), "area", lambda k: f"{AREAS[k][0]} {AREAS[k][1]}")
    a, b = st.columns(2)
    with a:
        weather = pills("Weather", list(WEATHER), "weather", WEATHER.get)
    with b:
        event = pills("What is happening?", list(EVENTS), "event", lambda k: f"{EVENTS[k][0]} {EVENTS[k][1]}")

# ───────────────────────────── step 4 ─────────────────────────────
with st.container(key="card_4"):
    show('<div class="sec"><div class="num">4</div><div><h2>Network conditions</h2><p>Describe them in plain words, or type exact measurements if you have them.</p></div></div>')
    entry = st.segmented_control("How would you like to enter this?", ["Simple", "Exact values"], key="entry",
                                 on_change=keep, args=("entry",)) or st.session_state.get("_last_entry", "Simple")
    if entry == "Simple":
        a, b, c = st.columns(3)
        with a:
            crowd = pills("How crowded is the tower?", list(CROWD), "crowd", lambda k: f"{CROWD[k][0]} {k}")
        with b:
            sig = pills("Signal strength", list(SIGNAL), "signal", lambda k: f"{SIGNAL[k][0]} {k}")
        with c:
            mv = pills("Are you moving?", list(MOVE), "move", lambda k: f"{MOVE[k][0]} {k}")
        cr = meta["crowd"][crowd]
        show(f'<div class="note"><b>{crowd}:</b> {CROWD[crowd][1]} '
             f'In our data this means about {cr["Cell_Load_pct"]:.0f}% tower load and {cr["Active_Users_In_Cell"]} active users.</div>')
        t = meta["app_typical"][app]
        vals = dict(Signal_dBm=SIGNAL[sig][1], Latency_ms=t["Latency_ms"], Req_Mbps=t["Req_Mbps"],
                    Resource_Alloc_pct=t["Resource_Alloc_pct"], Jitter_ms=cr["Jitter_ms"],
                    Cell_Load_pct=cr["Cell_Load_pct"], Active_Users_In_Cell=cr["Active_Users_In_Cell"],
                    Mobility_Speed_kmph=MOVE[mv][1])
    else:
        show('<div class="note">Enter measurements from a speed-test app or network tool. Every field is limited to the range seen in our data.</div>')
        order = list(XF)
        vals = {}
        for i in range(0, 8, 3):
            cols = st.columns(3)
            for j, col in enumerate(order[i:i + 3]):
                k, label, hlp, step, fmt = XF[col]
                lo, hi, _m = RG[col]
                isint = col == "Active_Users_In_Cell"
                with cols[j]:
                    vals[col] = st.number_input(label, min_value=int(lo) if isint else float(lo),
                                                max_value=int(hi) if isint else float(hi), step=step,
                                                format=fmt, help=hlp, key=k)
        with cols[2]:
            show('<div style="height:1.75rem"></div>')
            st.button("↺ Use typical values", key="typ", on_click=load_typical, width="stretch",
                      help="Fill every field with typical values for your chosen activity.")

row = dict(Application_Type=app, Operator=st.session_state.operator, Circle=circle, City_Tier=tier, Area_Type=area,
           Network_Mode=mode, Device_Category=device, Plan_Type=plan, Indoor_Outdoor=indoor, Weather=weather,
           Event_Context=event, **vals)
row["Active_Users_In_Cell"] = int(row["Active_Users_In_Cell"])

with st.container(key="cta"):
    go = st.button("Check my network  →", key="go", width="stretch")
if go:
    st.session_state.result = row

# ───────────────────────────── results ─────────────────────────────
res = st.session_state.get("result")
if res:
    if res != row:
        show('<div class="warn">⚠️ You changed some answers. Press “Check my network” again to refresh the result below.</div>')
    s = float(scores([res])[0])
    cls = band(s)
    proba = dict(zip(clf.classes_, clf.predict_proba(frame([res]))[0]))
    op, ap = res["Operator"], res["Application_Type"]
    verdict = {"Poor": "Expect a rough ride. {a} is likely to lag, buffer or drop.",
               "Fair": "Usable, but not smooth. {a} may stutter when the network gets busy.",
               "Good": "A solid connection. {a} should run smoothly for most of the session.",
               "Excellent": "Top-tier experience. {a} should feel fast, steady and effortless."}[cls].format(a=APPS[ap][1])
    ctx = f'{op} · {APPS[ap][1]} · {AREAS[res["Area_Type"]][1]} · {EVENTS[res["Event_Context"]][1]}'

    show('<div class="title">Your result</div>')
    likes = "".join(
        f'<div class="row"><div class="lab"><span>{n}</span></div><div class="trk"><i style="width:{proba[n] * 100:.0f}%;background:{CCOL[n]}"></i></div>'
        f'<div class="val">{proba[n] * 100:.0f}%</div></div>' for n in ORDER)
    legend = "".join(f'<span style="background:{CCOL[n]}33">{n} · {lo:g}–{hi:g}</span>' for lo, hi, n in BANDS)
    show(f"""
    <div class="res">
     <div class="rc tilt"><div class="ctx">{LOGOS[op]}<span>{ctx}</span></div>
      <div class="badge" style="background:{CCOL[cls]}">{cls}</div>
      <p class="verdict">{verdict}</p>
      <h3 style="font-size:1rem;margin-top:16px">How sure is the model?</h3>{likes}</div>
     <div class="rc tilt"><h3>Predicted experience score</h3><div class="sub">Quality of Experience (QoE), from 1 (worst) to 5 (best).</div>
      {dial(s)}<div class="legend">{legend}</div></div>
    </div>""")

    tips = levers(res, s)
    if tips:
        li = "".join(f'<li><span class="d">+{d:.2f}</span><span>{t}</span></li>' for d, t in tips)
        show(f'<div class="rc" style="margin-bottom:22px"><h3>🛠️ How to improve this</h3><div class="sub">Each change below is tested on the model, keeping everything else the same.</div><ul class="tips">{li}</ul></div>')
    else:
        show('<div class="rc" style="margin-bottom:22px"><h3>🛠️ How to improve this</h3><div class="sub">Nothing obvious to fix. You are already in a strong position for this activity.</div></div>')

    ar = list(AREAS); evs = list(EVENTS); ops = list(OPS)
    sa = scores([switch(res, "Area_Type", a) for a in ar]); se = scores([switch(res, "Event_Context", e) for e in evs])
    so = scores([{**res, "Operator": o} for o in ops])
    c1 = bar_rows(sorted(zip(ar, sa), key=lambda t: -t[1]), res["Area_Type"], lambda k: AREAS[k][1], lambda k: AREAS[k][0])
    c2 = bar_rows(sorted(zip(evs, se), key=lambda t: -t[1]), res["Event_Context"], lambda k: EVENTS[k][1], lambda k: EVENTS[k][0])
    c3 = bar_rows(sorted(zip(ops, so), key=lambda t: -t[1]), res["Operator"], lambda k: k, lambda k: LOGOS[k])
    spread = max(so) - min(so)
    onote = ("In this data, the three operators score almost the same. Activity and crowding matter far more than the operator."
             if spread < 0.15 else "These scores differ a little by operator. Activity and crowding still matter more.")
    show('<div class="title">Compare your options</div>')
    show(f"""
    <div class="cmp">
     <div class="rc tilt"><h3>📍 Where is it best?</h3><div class="sub">Same situation in a different type of area, with its usual crowding.</div>{c1}</div>
     <div class="rc tilt"><h3>🕒 When is it best?</h3><div class="sub">Same situation on a different occasion, with its usual crowding.</div>{c2}</div>
     <div class="rc tilt"><h3>📡 Which operator?</h3><div class="sub">{onote}</div>{c3}</div>
    </div>""")
    st.button("↺ Start over", key="reset", on_click=reset_all)

# ───────────────────────────── about ─────────────────────────────
m = meta["metrics"]
show('<div class="title">About this project</div>')
show(f"""
<div class="about">
 <div class="rc"><h3>What does SignalScope predict?</h3>
  <p>It predicts the <b>Quality of Experience (QoE)</b> a person gets during a 5G session. You receive a score from 1 to 5 and a quality rating: Poor, Fair, Good or Excellent.</p>
  <p>The prediction comes from 19 inputs such as activity, tower crowding, jitter, area type, weather and local events.</p></div>
 <div class="rc"><h3>How it was built</h3>
  <ul><li>Dataset: {m['rows']:,} 5G sessions across 20 Indian telecom circles and 3 operators.</li>
  <li>Method: exploratory data analysis, then a gradient boosting model (scikit-learn).</li>
  <li>Score accuracy: R² of {m['r2']:.2f} on unseen data.</li>
  <li>Rating accuracy: {m['acc'] * 100:.0f}% across four classes (macro F1 {m['f1']:.2f}).</li>
  <li>Interface: Python and Streamlit, fully responsive.</li></ul></div>
</div>
<div class="rc" style="margin-bottom:18px"><h3>What the data says</h3>
 <ul style="color:#3a3650;line-height:1.7;padding-left:1.1rem;margin:.3rem 0 0">
  <li><b>Activity matters most.</b> IoT sensors fare best, while AR/VR and cloud gaming fare worst.</li>
  <li><b>Crowding hurts.</b> High tower load, many active users and high jitter all lower the score.</li>
  <li><b>Operator, circle and device make little difference</b> in this dataset.</li>
  <li><b>Limitation:</b> signal strength and latency show almost no link to experience, which suggests the data is simulated. Treat predictions as an estimate, not a guarantee.</li></ul></div>
<details><summary>What is jitter?</summary><p>Jitter is how much the delay on your connection changes from moment to moment. High jitter causes choppy calls and laggy games.</p></details>
<details><summary>What is tower (cell) load?</summary><p>It is the share of a mobile tower's capacity currently in use. The busier the tower, the less room there is for each user.</p></details>
<details><summary>What is Standalone (SA) vs Non-standalone (NSA) 5G?</summary><p>SA runs entirely on a 5G core network. NSA uses a 5G radio connection anchored to a 4G core. SA generally scores slightly higher in this data.</p></details>
<details><summary>What is QoE?</summary><p>Quality of Experience is a 1 to 5 score of how good the service feels to the user. It is not the same as raw speed.</p></details>
<div class="foot"></div>
""")
