"""App-wide look and feel: global CSS, animated hero, and UI helpers.

`inject()` applies the theme CSS, `hero()` renders the animated header, and
`explain()` / `takeaway()` / `style()` are the callout and plotly-theming
helpers every tab uses.
"""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

# Assets for the "Ethereal Shadow" hero (SVG turbulence + displacement filter
# with an animated hue-rotate, ported from the etheral-shadow React/shadcn
# component to pure SVG/CSS - no framer-motion needed).
_MASK = "https://framerusercontent.com/images/ceBGguIpUU8luwByxuQz79t7To.png"
_NOISE = "https://framerusercontent.com/images/g0QcWrxr87K0ufOxIUFBakwYA8.png"


def inject() -> None:
    """Global CSS: background, metric cards, explainer/takeaway callouts."""
    st.markdown("""
    <style>
    .stApp { background: radial-gradient(ellipse at top left, rgba(99,102,241,.08), transparent 50%),
                         radial-gradient(ellipse at bottom right, rgba(244,63,94,.06), transparent 50%),
                         #060609; }
    div[data-testid="stMetric"] { background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08);
      border-radius:14px; padding:1rem 1.2rem; }
    .explain { border-left:3px solid #818cf8; background:rgba(129,140,248,.07);
      padding:.65rem .9rem; border-radius:0 10px 10px 0; color:rgba(255,255,255,.75);
      font-size:.9rem; margin-bottom:.8rem; }
    .takeaway { border-left:3px solid #fb7185; background:rgba(251,113,133,.07);
      padding:.65rem .9rem; border-radius:0 10px 10px 0; color:rgba(255,255,255,.85);
      font-size:.92rem; margin:.4rem 0 .8rem; }
    hr { border-color: rgba(255,255,255,.08); }
    </style>
    """, unsafe_allow_html=True)


def hero() -> None:
    """Animated hero header, rendered in a self-contained frame so Streamlit's
    HTML sanitizer doesn't strip the SVG filter primitives."""
    components.html(f"""
    <div class="wrap">
      <svg width="0" height="0" style="position:absolute">
        <defs>
          <filter id="eshadow">
            <feTurbulence result="undulation" numOctaves="2" baseFrequency="0.0005,0.002"
                          seed="0" type="turbulence"/>
            <feColorMatrix in="undulation" type="hueRotate" values="180">
              <animate attributeName="values" from="0" to="360" dur="8s"
                       repeatCount="indefinite"/>
            </feColorMatrix>
            <feColorMatrix in="dist" result="circulation" type="matrix"
                           values="4 0 0 0 1  4 0 0 0 1  4 0 0 0 1  1 0 0 0 0"/>
            <feDisplacementMap in="SourceGraphic" in2="circulation" scale="45" result="dist"/>
            <feDisplacementMap in="dist" in2="undulation" scale="45" result="output"/>
          </filter>
        </defs>
      </svg>
      <div class="fx"><div class="shape"></div></div>
      <div class="noise"></div>
      <div class="content">
        <div class="badge"><span class="dot"></span>NLP · RAG · LLM AGENTS</div>
        <h1>YouTube Intelligence <span>Engine</span></h1>
        <p>Turning thousands of raw comments into insights anyone can read.</p>
      </div>
    </div>
    <style>
      html, body {{ margin:0; background:transparent; overflow:hidden; }}
      .wrap {{ position:relative; height:290px; border-radius:18px; overflow:hidden;
        background:#08080d; border:1px solid rgba(255,255,255,.08);
        font-family:'Source Sans Pro', system-ui, sans-serif; }}
      .fx {{ position:absolute; inset:-45px; filter:url(#eshadow) blur(4px); }}
      .shape {{ width:100%; height:100%; background-color:rgba(150,160,225,.85);
        -webkit-mask-image:url('{_MASK}'); mask-image:url('{_MASK}');
        -webkit-mask-size:cover; mask-size:cover;
        -webkit-mask-repeat:no-repeat; mask-repeat:no-repeat;
        -webkit-mask-position:center; mask-position:center; }}
      .noise {{ position:absolute; inset:0; background-image:url('{_NOISE}');
        background-size:240px; background-repeat:repeat; opacity:.5; }}
      .content {{ position:absolute; inset:0; display:flex; flex-direction:column;
        align-items:center; justify-content:center; text-align:center; z-index:10; }}
      .badge {{ display:inline-flex; align-items:center; gap:.5rem; padding:.3rem .9rem;
        border-radius:999px; background:rgba(0,0,0,.35); border:1px solid rgba(255,255,255,.12);
        color:rgba(255,255,255,.7); font-size:.8rem; letter-spacing:.05em; margin-bottom:1rem;
        backdrop-filter:blur(6px); }}
      .dot {{ width:8px; height:8px; border-radius:50%; background:#fb7185; }}
      h1 {{ font-size:3rem; font-weight:800; letter-spacing:-.02em; margin:0; line-height:1.1;
        background:linear-gradient(180deg,#fff,rgba(255,255,255,.75));
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
      h1 span {{ background:linear-gradient(90deg,#a5b4fc,#e2e8f0,#fda4af);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
      p {{ color:rgba(255,255,255,.55); font-weight:300; margin-top:.7rem; }}
    </style>
    """, height=300)


def explain(text: str):
    st.markdown(f'<div class="explain">{text}</div>', unsafe_allow_html=True)


def takeaway(text: str):
    st.markdown(f'<div class="takeaway"><b>Takeaway:</b> {text}</div>',
                unsafe_allow_html=True)


PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)")


def style(fig):
    fig.update_layout(**PLOT, margin=dict(t=30, b=10))
    return fig
