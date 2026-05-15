"""GraphDTI Explorer — Apple-style Streamlit UI.

A clean, friendly interface to the GraphDTI model. Loads checkpoints directly
(no FastAPI server required) and forces CPU inference.

Run:
    streamlit run scripts/app_ui.py
"""
from __future__ import annotations

import math
import os
from pathlib import Path

# Hide GPU from this process so it doesn't fight any concurrent training
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd
import streamlit as st
import torch

from graphdti.data.featurize import encode_protein, smiles_to_graph
from graphdti.interpret import atom_attributions, residue_occlusion
from graphdti.serving.app import load_predictor
from torch_geometric.data import Batch


# ============================================================
# Preset libraries
# ============================================================

DRUGS = {
    "Aspirin": ("CC(=O)OC1=CC=CC=C1C(=O)O", "Common", "Painkiller / blood thinner"),
    "Ibuprofen": ("CC(C)Cc1ccc(C(C)C(=O)O)cc1", "Common", "NSAID painkiller"),
    "Acetaminophen": ("CC(=O)Nc1ccc(O)cc1", "Common", "Tylenol — pain & fever"),
    "Caffeine": ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Common", "Stimulant"),
    "Metformin": ("CN(C)C(=N)NC(N)=N", "Common", "Type 2 diabetes"),
    "Atorvastatin": ("CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O", "Common", "Lipitor — cholesterol"),
    "Imatinib (Gleevec)": ("Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1", "Kinase inhibitor", "BCR-ABL inhibitor for CML"),
    "Gefitinib (Iressa)": ("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", "Kinase inhibitor", "EGFR inhibitor — lung cancer"),
    "Erlotinib (Tarceva)": ("COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC", "Kinase inhibitor", "EGFR inhibitor — lung cancer"),
    "Vemurafenib (Zelboraf)": ("CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ncc(Cl)cc23)c1F", "Kinase inhibitor", "BRAF V600E — melanoma"),
    "Dabrafenib (Tafinlar)": ("CC(C)(C)c1nc(-c2cccc(NS(=O)(=O)c3c(F)cccc3F)c2F)c(-c2ccnc(N)n2)s1", "Kinase inhibitor", "BRAF V600E — melanoma"),
    "Sorafenib (Nexavar)": ("CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1", "Kinase inhibitor", "Multi-kinase — liver/kidney cancer"),
    "Sunitinib (Sutent)": ("CCN(CC)CCNC(=O)c1c(C)[nH]c(/C=C2\\C(=O)Nc3ccc(F)cc32)c1C", "Kinase inhibitor", "Multi-kinase — kidney/GIST cancer"),
    "Crizotinib (Xalkori)": ("C[C@@H](Oc1cc(-c2cnn(C3CCNCC3)c2)cnc1N)c1c(Cl)ccc(F)c1Cl", "Kinase inhibitor", "ALK/ROS1 — lung cancer"),
    "Nilotinib (Tasigna)": ("Cc1cn(-c2cc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)n4)c3)cc(C(F)(F)F)c2)cn1", "Kinase inhibitor", "BCR-ABL — CML 2nd line"),
    "Sertraline (Zoloft)": ("CN[C@@H]1CC[C@H](c2ccc(Cl)c(Cl)c2)c2ccccc21", "Psychiatric", "SSRI antidepressant"),
    "Diphenhydramine": ("CN(C)CCOC(c1ccccc1)c1ccccc1", "Antihistamine", "Benadryl"),
    "Losartan": ("CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1", "Cardiovascular", "Hypertension"),
}

PROTEINS = {
    "ABL1 kinase (CML — target of imatinib)": (
        "MGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSR"
        "NAAEYLLSSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTVADGLITTLH"
        "YPAPKRNKPTVYGVSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIK"
        "HPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGEN"
        "HLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEK"
        "DYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSISDEVEKELGKQGV",
        "Cancer", "Chronic myeloid leukemia driver kinase",
    ),
    "EGFR kinase (lung cancer — target of gefitinib)": (
        "FKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQ"
        "LMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHA"
        "EGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCW"
        "MIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFY",
        "Cancer", "Lung cancer driver kinase",
    ),
    "BRAF V600E kinase (melanoma — target of vemurafenib)": (
        "DLTVKIGDFGLATEKSRWSGSHQFEQLSGSILWMAPEVIRMQDKNPYSFQSDVYAFGIVLYELMTGQLPYSNINNRDQII"
        "FMVGRGYLSPDLSKVRSNCPKAMKRLMAECLKKKRDERPLFPQILASIELLARSLPKIHRSASEPSLNRAGFQTEDFSLY"
        "ACASPKTPIQAGGYGAFPVH",
        "Cancer", "Melanoma driver kinase (V600E mutant)",
    ),
}


# ============================================================
# Apple-style CSS
# ============================================================

CUSTOM_CSS = """
<style>
    /* Use system font stack (Apple's default on macOS, sensible fallbacks elsewhere) */
    html, body, [class*="css"], button, input, textarea, select {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text",
                     "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* Streamlit chrome cleanup */
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
    .stDeployButton { display: none; }

    /* Generous page padding */
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 6rem !important;
        max-width: 1100px !important;
    }

    /* Hero */
    .gdti-hero {
        text-align: center;
        padding: 4rem 0 3rem 0;
        margin-bottom: 2rem;
    }
    .gdti-hero .eyebrow {
        font-size: 0.95rem;
        font-weight: 500;
        color: #0071e3;
        letter-spacing: 0.02em;
        margin-bottom: 0.7rem;
    }
    .gdti-hero h1 {
        font-size: 4.5rem;
        font-weight: 700;
        line-height: 1.05;
        letter-spacing: -0.025em;
        color: #1d1d1f;
        margin: 0 0 1rem 0;
    }
    .gdti-hero h1 .accent {
        background: linear-gradient(135deg, #0071e3 0%, #5e5ce6 50%, #ff375f 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .gdti-hero .subtitle {
        font-size: 1.5rem;
        line-height: 1.4;
        color: #6e6e73;
        max-width: 720px;
        margin: 0 auto 1.5rem auto;
        font-weight: 400;
    }
    .gdti-hero .stats {
        display: flex;
        justify-content: center;
        gap: 2.5rem;
        margin-top: 2.5rem;
        flex-wrap: wrap;
    }
    .gdti-hero .stat {
        text-align: center;
    }
    .gdti-hero .stat .n {
        font-size: 2rem;
        font-weight: 600;
        color: #1d1d1f;
        line-height: 1;
        margin-bottom: 0.3rem;
    }
    .gdti-hero .stat .l {
        font-size: 0.85rem;
        color: #6e6e73;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Section heading */
    .gdti-section-h {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #1d1d1f;
        margin: 3.5rem 0 0.5rem 0;
    }
    .gdti-section-sub {
        font-size: 1.1rem;
        color: #6e6e73;
        margin-bottom: 2rem;
    }

    /* Cards (Apple's signature soft elevated panels) */
    .gdti-card {
        background: #ffffff;
        border-radius: 22px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.03);
        border: 1px solid rgba(0, 0, 0, 0.04);
        margin-bottom: 1.5rem;
    }
    .gdti-soft {
        background: #f5f5f7;
        border-radius: 18px;
        padding: 1.5rem;
        border: 1px solid rgba(0, 0, 0, 0.03);
    }

    /* Result block */
    .gdti-result {
        border-radius: 22px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    .gdti-result.win  { background: linear-gradient(180deg, #e8f8ee 0%, #ffffff 100%); }
    .gdti-result.warn { background: linear-gradient(180deg, #fff4e1 0%, #ffffff 100%); }
    .gdti-result.fail { background: linear-gradient(180deg, #ffeaea 0%, #ffffff 100%); }
    .gdti-result .eyebrow {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6e6e73;
        margin: 0;
    }
    .gdti-result .num {
        font-size: 5rem;
        font-weight: 700;
        line-height: 1;
        margin: 0.6rem 0;
        letter-spacing: -0.03em;
    }
    .gdti-result.win  .num { color: #34c759; }
    .gdti-result.warn .num { color: #ff9500; }
    .gdti-result.fail .num { color: #ff3b30; }
    .gdti-result .label {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1d1d1f;
    }
    .gdti-result .threshold-note {
        font-size: 0.9rem;
        color: #86868b;
        margin-top: 0.5rem;
    }

    /* Pills / badges */
    .gdti-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        font-size: 0.85rem;
        font-weight: 500;
        background: #f5f5f7;
        color: #1d1d1f;
        border-radius: 999px;
        margin-right: 6px;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    .gdti-pill.green { background: #e8f8ee; color: #28a745; border-color: rgba(40,167,69,0.15); }
    .gdti-pill.amber { background: #fff4e1; color: #e08e0b; border-color: rgba(224,142,11,0.15); }
    .gdti-pill.red   { background: #ffeaea; color: #ff3b30; border-color: rgba(255,59,48,0.15); }
    .gdti-pill.blue  { background: #e7f1ff; color: #0071e3; border-color: rgba(0,113,227,0.15); }

    /* Streamlit buttons → Apple-style pills */
    .stButton > button {
        background: #0071e3 !important;
        color: white !important;
        border: none !important;
        border-radius: 980px !important;
        padding: 0.7rem 1.8rem !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        box-shadow: 0 1px 3px rgba(0, 113, 227, 0.2);
        transition: all 0.2s ease-out !important;
    }
    .stButton > button:hover {
        background: #0077ed !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 113, 227, 0.25);
    }
    .stButton > button[kind="primary"] {
        background: #1d1d1f !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #2d2d2f !important;
    }
    .stDownloadButton > button {
        background: #f5f5f7 !important;
        color: #1d1d1f !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
    }
    .stDownloadButton > button:hover {
        background: #ebebed !important;
    }

    /* Inputs */
    .stTextInput input, .stTextArea textarea, .stSelectbox > div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        border: 1px solid rgba(0,0,0,0.12) !important;
        background: #f5f5f7 !important;
        transition: border-color 0.2s, background 0.2s !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        background: #ffffff !important;
        border-color: #0071e3 !important;
        box-shadow: 0 0 0 4px rgba(0, 113, 227, 0.1) !important;
    }
    .stTextInput label, .stTextArea label, .stSelectbox label, .stSlider label, .stMultiSelect label {
        font-weight: 500 !important;
        color: #1d1d1f !important;
        font-size: 0.9rem !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #f5f5f7;
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 0.5rem 1.2rem !important;
        font-weight: 500 !important;
        color: #6e6e73 !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #1d1d1f !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    /* Sequence heatmap */
    .seq-heatmap {
        font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace !important;
        font-size: 0.78rem;
        line-height: 1.85;
        word-break: break-all;
        background: #fbfbfd;
        padding: 1.25rem;
        border-radius: 14px;
        border: 1px solid rgba(0,0,0,0.05);
    }

    /* Molecule frame */
    .mol-frame {
        background: white;
        border-radius: 18px;
        padding: 1.25rem;
        display: flex;
        justify-content: center;
        align-items: center;
        border: 1px solid rgba(0,0,0,0.06);
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }

    /* Sidebar polish */
    [data-testid="stSidebar"] {
        background: #fbfbfd !important;
        border-right: 1px solid rgba(0,0,0,0.05);
    }

    /* Captions / helpers */
    .helper { color: #86868b; font-size: 0.9rem; }
</style>
"""


# ============================================================
# Helpers
# ============================================================

@st.cache_resource(show_spinner=False)
def get_predictor(ckpt_path: str):
    return load_predictor(ckpt_path, threshold=0.5, device="cpu")


def list_checkpoints() -> list[str]:
    ckpts = sorted(Path("checkpoints").glob("dti*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in ckpts]


def smiles_to_svg(smiles: str, width: int = 380, height: int = 260) -> str | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import rdMolDraw2D
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>\n", "")
    except Exception:
        return None


def probability_ring(prob: float, threshold: float) -> str:
    """A clean Apple-style circular progress ring."""
    pct = max(0.0, min(1.0, prob))
    if prob >= 0.75:
        color = "#34c759"
    elif prob >= threshold:
        color = "#ff9500"
    else:
        color = "#ff3b30"
    radius = 70
    circumference = 2 * math.pi * radius
    dash = circumference * pct
    gap = circumference - dash
    return f"""
    <div style="display:flex;justify-content:center;align-items:center;height:200px">
      <svg width="180" height="180" viewBox="0 0 180 180" style="transform:rotate(-90deg)">
        <circle cx="90" cy="90" r="{radius}" fill="none" stroke="#f5f5f7" stroke-width="14"/>
        <circle cx="90" cy="90" r="{radius}" fill="none" stroke="{color}" stroke-width="14"
                stroke-linecap="round" stroke-dasharray="{dash:.1f} {gap:.1f}"
                style="transition: stroke-dasharray 0.8s cubic-bezier(0.22, 1, 0.36, 1)"/>
      </svg>
      <div style="position:absolute;text-align:center">
        <div style="font-size:2.4rem;font-weight:700;color:{color};line-height:1;letter-spacing:-0.02em">{prob:.2f}</div>
        <div style="font-size:0.7rem;color:#86868b;text-transform:uppercase;letter-spacing:0.1em;margin-top:0.4rem">Probability</div>
      </div>
    </div>
    """


def render_result_block(prob: float, threshold: float) -> str:
    if prob >= 0.75:
        klass, label = "win", "Strong binder"
    elif prob >= threshold:
        klass, label = "warn", "Possible binder"
    else:
        klass, label = "fail", "Likely non-binder"
    return f"""
    <div class="gdti-result {klass}">
      <p class="eyebrow">Binding probability</p>
      <div class="num">{prob:.3f}</div>
      <div class="label">{label}</div>
      <div class="threshold-note">Threshold {threshold:.2f}</div>
    </div>
    """


def render_sequence_heatmap(tokens: list[str], scores: list[float]) -> str:
    max_abs = max(max(abs(s) for s in scores), 1e-6)
    spans = []
    for tok, sc in zip(tokens, scores):
        intensity = abs(sc) / max_abs
        if sc >= 0:
            rgb = f"rgba(255, 59, 48, {intensity * 0.85:.2f})"
        else:
            rgb = f"rgba(0, 113, 227, {intensity * 0.85:.2f})"
        spans.append(f'<span style="background:{rgb};padding:2px 1px;border-radius:3px">{tok}</span>')
    return f'<div class="seq-heatmap">{"".join(spans)}</div>'


def explain_pair(predictor, smiles, protein, do_atom, do_residue, occ_window):
    graph = smiles_to_graph(smiles)
    if graph is None:
        raise ValueError("Could not parse SMILES.")
    protein_t = encode_protein(protein)
    batch = Batch.from_data_list([graph]).to(predictor.device)
    protein_b = protein_t.unsqueeze(0).to(predictor.device)
    with torch.no_grad():
        prob = float(predictor.model.predict_proba(batch, protein_b).item())
    atom_data, res_data = None, None
    if do_atom:
        atom_data = atom_attributions(predictor.model, graph, protein_t, steps=16, device=predictor.device).to_dict()
    if do_residue:
        res_data = residue_occlusion(
            predictor.model, graph, protein_t, protein,
            window=occ_window, stride=max(1, occ_window // 2), device=predictor.device,
        ).to_dict()
    return prob, atom_data, res_data


# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title="GraphDTI — Drug-Target Interaction Predictor",
    page_icon="🧬",
    layout="wide",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Session-state defaults
DEFAULT_DRUG = "Imatinib (Gleevec)"
DEFAULT_PROTEIN = "ABL1 kinase (CML — target of imatinib)"
DEFAULT_SCREEN_PROTEIN = "EGFR kinase (lung cancer — target of gefitinib)"

if "pred_drug_choice" not in st.session_state:
    st.session_state.pred_drug_choice = DEFAULT_DRUG
    st.session_state.pred_drug_text = DRUGS[DEFAULT_DRUG][0]
if "pred_prot_choice" not in st.session_state:
    st.session_state.pred_prot_choice = DEFAULT_PROTEIN
    st.session_state.pred_prot_text = PROTEINS[DEFAULT_PROTEIN][0]
if "scr_prot_choice" not in st.session_state:
    st.session_state.scr_prot_choice = DEFAULT_SCREEN_PROTEIN
    st.session_state.scr_prot_text = PROTEINS[DEFAULT_SCREEN_PROTEIN][0]


def _sync_drug_predict():
    c = st.session_state.pred_drug_choice
    st.session_state.pred_drug_text = DRUGS[c][0] if c in DRUGS else ""


def _sync_protein_predict():
    c = st.session_state.pred_prot_choice
    st.session_state.pred_prot_text = PROTEINS[c][0] if c in PROTEINS else ""


def _sync_protein_screen():
    c = st.session_state.scr_prot_choice
    st.session_state.scr_prot_text = PROTEINS[c][0] if c in PROTEINS else ""


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("### Settings")
    ckpts = list_checkpoints()
    if not ckpts:
        st.error("No checkpoints found. Train a model first.")
        st.stop()
    ckpt_path = st.selectbox("Model", ckpts, index=0, label_visibility="visible")
    predictor = get_predictor(ckpt_path)
    threshold = st.slider("Threshold", 0.0, 1.0, 0.42, 0.01)

    st.markdown("---")
    st.caption(f"**Loaded:** `{Path(ckpt_path).name}`")
    st.caption(f"**Device:** `{predictor.device}`")

    st.markdown("---")
    st.markdown("**Probability bands**")
    st.markdown(
        """
        <div style="line-height:1.9;font-size:0.9rem">
          <span class="gdti-pill green">🟢 ≥ 0.75</span> Strong<br>
          <span class="gdti-pill amber">🟡 ≥ threshold</span> Possible<br>
          <span class="gdti-pill red">🔴 below</span> Non-binder
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption(
        "GraphDTI uses a Graph Isomorphism Network for the drug and a 1-D CNN "
        "for the protein, trained on 558k measured pairs from BindingDB."
    )


# ============================================================
# Hero
# ============================================================

st.markdown(
    """
    <div class="gdti-hero">
      <div class="eyebrow">GraphDTI · Open source · AUROC 0.91</div>
      <h1>Predict <span class="accent">drug-protein binding</span><br>in a millisecond.</h1>
      <p class="subtitle">Score any small molecule against any target. Built on a graph neural network trained over thirty years of measured chemistry.</p>
      <div class="stats">
        <div class="stat"><div class="n">558K</div><div class="l">Training pairs</div></div>
        <div class="stat"><div class="n">244K</div><div class="l">Unique compounds</div></div>
        <div class="stat"><div class="n">3,007</div><div class="l">Protein targets</div></div>
        <div class="stat"><div class="n">0.91</div><div class="l">Validation AUROC</div></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Tabs
# ============================================================

tab_predict, tab_compare, tab_screen, tab_about = st.tabs(
    ["Predict", "Compare", "Screen library", "How it works"]
)


# ===== Predict =====
with tab_predict:
    st.markdown('<h2 class="gdti-section-h">Try a single pair</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="gdti-section-sub">Pick a drug and a target protein. Get an instant binding probability with optional atom-level explanation.</p>',
        unsafe_allow_html=True,
    )

    col_d, col_p = st.columns(2, gap="large")

    with col_d:
        st.markdown("**Drug**")
        st.selectbox(
            "Quick pick",
            ["(custom SMILES)"] + list(DRUGS.keys()),
            key="pred_drug_choice",
            on_change=_sync_drug_predict,
            label_visibility="collapsed",
        )
        st.text_input("SMILES", key="pred_drug_text", label_visibility="collapsed",
                      placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O")
        drug_smiles = st.session_state.pred_drug_text
        drug_choice = st.session_state.pred_drug_choice
        if drug_choice in DRUGS:
            _, cat, desc = DRUGS[drug_choice]
            st.markdown(f'<span class="gdti-pill blue">{cat}</span><span class="helper" style="margin-left:6px">{desc}</span>',
                        unsafe_allow_html=True)
        if drug_smiles:
            svg = smiles_to_svg(drug_smiles, width=380, height=260)
            if svg:
                st.markdown(f'<div class="mol-frame" style="margin-top:1rem">{svg}</div>',
                            unsafe_allow_html=True)
            else:
                st.warning("Could not render this SMILES.")

    with col_p:
        st.markdown("**Protein target**")
        st.selectbox(
            "Quick pick",
            ["(custom sequence)"] + list(PROTEINS.keys()),
            key="pred_prot_choice",
            on_change=_sync_protein_predict,
            label_visibility="collapsed",
        )
        st.text_area("Sequence", key="pred_prot_text", height=240, label_visibility="collapsed",
                     placeholder="Paste from UniProt FASTA")
        protein_seq = st.session_state.pred_prot_text
        protein_choice = st.session_state.pred_prot_choice
        if protein_choice in PROTEINS:
            seq, cat, desc = PROTEINS[protein_choice]
            st.markdown(
                f'<span class="gdti-pill blue">{cat}</span><span class="helper" style="margin-left:6px">{desc} · {len(seq)} residues</span>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    bcol1, bcol2, _ = st.columns([1, 1.4, 4])
    with bcol1:
        do_predict = st.button("Predict", type="primary", use_container_width=True, key="btn_predict")
    with bcol2:
        do_explain = st.button("Predict & explain", use_container_width=True, key="btn_explain")

    if do_predict or do_explain:
        if not drug_smiles or not protein_seq.strip():
            st.error("Both drug SMILES and protein sequence are required.")
        else:
            try:
                with st.spinner("Computing..."):
                    if do_explain:
                        prob, atom_data, res_data = explain_pair(
                            predictor, drug_smiles, protein_seq.strip(),
                            do_atom=True, do_residue=True, occ_window=10,
                        )
                    else:
                        prob = predictor.predict(drug_smiles, protein_seq.strip())
                        atom_data, res_data = None, None
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            colg, colr = st.columns([1, 1.4], gap="large")
            with colg:
                st.markdown(probability_ring(prob, threshold), unsafe_allow_html=True)
            with colr:
                st.markdown(render_result_block(prob, threshold), unsafe_allow_html=True)

            if atom_data:
                st.markdown('<h3 style="margin-top:2.5rem;font-weight:600">Atom contributions</h3>',
                            unsafe_allow_html=True)
                st.markdown(
                    '<p class="helper">Per-atom contribution to the prediction. Positive bars pushed the score toward "binds".</p>',
                    unsafe_allow_html=True,
                )
                df_a = pd.DataFrame({"atom": atom_data["tokens"], "score": atom_data["scores"]})
                df_a_top = df_a.reindex(df_a["score"].abs().sort_values(ascending=False).index).head(12)
                st.bar_chart(df_a_top.set_index("atom"), height=280, color="#0071e3")

            if res_data:
                st.markdown('<h3 style="margin-top:2.5rem;font-weight:600">Protein region map</h3>',
                            unsafe_allow_html=True)
                st.markdown(
                    '<p class="helper">Each residue is colored by its influence on the prediction. <span style="color:#ff3b30">Red</span> = relied on, <span style="color:#0071e3">blue</span> = argued against.</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(render_sequence_heatmap(res_data["tokens"], res_data["scores"]),
                            unsafe_allow_html=True)


# ===== Compare =====
with tab_compare:
    st.markdown('<h2 class="gdti-section-h">Compare drugs head-to-head</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="gdti-section-sub">Score 2–6 drugs against the same target. Useful for picking the most promising binder.</p>',
        unsafe_allow_html=True,
    )

    protein_choice = st.selectbox(
        "Target protein", list(PROTEINS.keys()), key="cmp_prot_choice", index=2,
    )
    protein_seq = PROTEINS[protein_choice][0]

    drug_choices = st.multiselect(
        "Drugs to compare",
        list(DRUGS.keys()),
        default=["Vemurafenib (Zelboraf)", "Dabrafenib (Tafinlar)", "Sorafenib (Nexavar)",
                 "Aspirin", "Caffeine"],
        max_selections=6,
        key="cmp_drug_choices",
    )

    if st.button("Compare", type="primary", key="cmp_btn"):
        if len(drug_choices) < 2:
            st.error("Pick at least 2 drugs.")
        else:
            results = []
            progress = st.progress(0)
            for i, name in enumerate(drug_choices, 1):
                smi = DRUGS[name][0]
                try:
                    p = predictor.predict(smi, protein_seq)
                except Exception:
                    p = None
                results.append({"name": name, "smiles": smi, "probability": p})
                progress.progress(i / len(drug_choices))
            progress.empty()

            df = pd.DataFrame(results).sort_values("probability", ascending=False, na_position="last")
            chart_df = df[["name", "probability"]].dropna().set_index("name")
            st.markdown(
                '<h3 style="margin-top:2.5rem;font-weight:600">Predicted binding probability</h3>',
                unsafe_allow_html=True,
            )
            st.bar_chart(chart_df, height=420, color="#0071e3")

            st.markdown(
                '<h3 style="margin-top:2.5rem;font-weight:600">Each compound</h3>',
                unsafe_allow_html=True,
            )
            for _, row in df.iterrows():
                with st.container():
                    cols = st.columns([1.3, 2.2, 1], gap="medium")
                    with cols[0]:
                        svg = smiles_to_svg(row["smiles"], width=200, height=150)
                        if svg:
                            st.markdown(
                                f'<div class="mol-frame" style="padding:0.4rem">{svg}</div>',
                                unsafe_allow_html=True,
                            )
                    with cols[1]:
                        st.markdown(f"**{row['name']}**")
                        cat_desc = DRUGS[row["name"]]
                        st.caption(f"{cat_desc[1]} — {cat_desc[2]}")
                        if row["probability"] is None:
                            continue
                        p = row["probability"]
                        if p >= 0.75:
                            badge = '<span class="gdti-pill green">Strong binder</span>'
                        elif p >= threshold:
                            badge = '<span class="gdti-pill amber">Possible</span>'
                        else:
                            badge = '<span class="gdti-pill red">Non-binder</span>'
                        st.markdown(badge, unsafe_allow_html=True)
                    with cols[2]:
                        if row["probability"] is not None:
                            color = "#34c759" if row["probability"] >= 0.75 else (
                                "#ff9500" if row["probability"] >= threshold else "#ff3b30")
                            st.markdown(
                                f'<div style="text-align:right;font-size:2.2rem;font-weight:700;'
                                f'color:{color};line-height:1">{row["probability"]:.3f}</div>',
                                unsafe_allow_html=True,
                            )


# ===== Bulk screen =====
with tab_screen:
    st.markdown('<h2 class="gdti-section-h">Virtual screen a library</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="gdti-section-sub">Score many compounds against one target. Output: a ranked CSV.</p>',
        unsafe_allow_html=True,
    )

    st.selectbox("Target protein", list(PROTEINS.keys()), key="scr_prot_choice",
                 on_change=_sync_protein_screen)
    st.text_area("Sequence (editable)", key="scr_prot_text", height=140)
    protein_seq = st.session_state.scr_prot_text

    default_lib = "\n".join([
        f"{n},{DRUGS[n][0]}" for n in
        ["Imatinib (Gleevec)", "Gefitinib (Iressa)", "Erlotinib (Tarceva)",
         "Sorafenib (Nexavar)", "Aspirin", "Caffeine", "Ibuprofen", "Atorvastatin"]
    ])
    library_text = st.text_area(
        "Compound library", value=default_lib, height=220, key="scr_lib",
        help="One per line. Format: `name,smiles` or just `smiles`.",
    )

    if st.button("Run screen", type="primary", key="scr_btn"):
        if not protein_seq.strip() or not library_text.strip():
            st.error("Need both a protein sequence and a library.")
        else:
            rows = []
            for i, line in enumerate(library_text.strip().splitlines()):
                line = line.strip()
                if not line:
                    continue
                if "," in line:
                    name, smiles = [x.strip() for x in line.split(",", 1)]
                else:
                    smiles, name = line, f"compound_{i+1}"
                rows.append({"name": name, "smiles": smiles})

            results = []
            progress = st.progress(0)
            status = st.empty()
            for i, r in enumerate(rows, 1):
                status.text(f"Scoring {i}/{len(rows)}: {r['name']}")
                try:
                    p = predictor.predict(r["smiles"], protein_seq.strip())
                    results.append({"name": r["name"], "smiles": r["smiles"], "probability": p})
                except Exception as e:
                    results.append({"name": r["name"], "smiles": r["smiles"], "probability": None,
                                    "error": str(e)[:80]})
                progress.progress(i / len(rows))
            status.empty()
            progress.empty()

            df = pd.DataFrame(results).sort_values("probability", ascending=False, na_position="last")
            df["status"] = df["probability"].apply(
                lambda x: "🟢 Strong" if x is not None and x >= 0.75
                else ("🟡 Possible" if x is not None and x >= threshold
                      else ("🔴 Non-binder" if x is not None else "error"))
            )
            df_view = df.copy()
            df_view["probability"] = df_view["probability"].apply(
                lambda x: f"{x:.4f}" if x is not None else "—")
            n_binders = sum(1 for x in df["probability"] if x is not None and x >= threshold)

            st.markdown(
                f'<h3 style="margin-top:2.5rem;font-weight:600">{n_binders} of {len(rows)} predicted binders</h3>',
                unsafe_allow_html=True,
            )
            st.dataframe(df_view[["name", "status", "probability", "smiles"]],
                         hide_index=True, use_container_width=True)
            st.download_button(
                "Download CSV",
                df.to_csv(index=False),
                file_name="screen_results.csv",
                mime="text/csv",
            )


# ===== About =====
with tab_about:
    st.markdown('<h2 class="gdti-section-h">How GraphDTI works</h2>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="gdti-card">
          <h3 style="margin-top:0;font-weight:600">The problem</h3>
          <p style="color:#1d1d1f;font-size:1.05rem;line-height:1.6;margin-bottom:0">
            Given a small-molecule drug and a target protein, predict whether they
            bind. This is the <b>virtual screening</b> step of drug discovery:
            cheaply rank millions of candidate molecules before sending the
            top hundred to a wet-lab assay. Wet-lab assays cost $50–500 per molecule;
            this model costs essentially nothing per prediction.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="gdti-card">
          <h3 style="margin-top:0;font-weight:600">The architecture</h3>
          <p style="color:#1d1d1f;font-size:1.05rem;line-height:1.6">
            <b>Two encoders</b> turn each input into a 256-dimensional vector,
            then a small head combines them into a single number.
          </p>
          <ul style="color:#1d1d1f;font-size:1.05rem;line-height:1.7">
            <li><b>Drug encoder.</b> A 5-layer Graph Isomorphism Network (GINE) over the molecular graph parsed from SMILES. Atoms are nodes, bonds are edges, both carrying chemical features.</li>
            <li><b>Protein encoder.</b> A 1-D dilated CNN over amino-acid embeddings. Captures local sequence motifs at multiple length scales.</li>
            <li><b>Interaction head.</b> A bilinear layer mixes both vectors, then a 2-layer MLP outputs a binding logit.</li>
          </ul>
          <p style="color:#1d1d1f;font-size:1.05rem;line-height:1.6;margin-bottom:0">
            Trained in two stages: contrastive pretraining of the drug encoder
            using Morgan-fingerprint hard negatives, then supervised fine-tuning
            with binary cross-entropy.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="gdti-card">
          <h3 style="margin-top:0;font-weight:600">The training data</h3>
          <p style="color:#1d1d1f;font-size:1.05rem;line-height:1.6;margin-bottom:0">
            <b>BindingDB</b> is a public database of 3.18 million measured binding
            affinities curated from published scientific papers. We label each pair
            as a binder (Ki ≤ 1 μM) or non-binder. The current model was trained
            on <b>558,571 pairs</b> covering ~244k unique molecules and ~3,000 proteins.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<h3 style="margin-top:2rem;font-weight:600;color:#1d1d1f">Performance</h3>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        '<div class="gdti-soft" style="text-align:center"><div style="font-size:3rem;font-weight:700;color:#0071e3;line-height:1">0.909</div><div style="color:#6e6e73;margin-top:0.4rem">AUROC</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        '<div class="gdti-soft" style="text-align:center"><div style="font-size:3rem;font-weight:700;color:#0071e3;line-height:1">0.960</div><div style="color:#6e6e73;margin-top:0.4rem">PR-AUC</div></div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        '<div class="gdti-soft" style="text-align:center"><div style="font-size:3rem;font-weight:700;color:#0071e3;line-height:1">0.904</div><div style="color:#6e6e73;margin-top:0.4rem">F1 score</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="gdti-card" style="margin-top:2rem">
          <h3 style="margin-top:0;font-weight:600">What it's good at</h3>
          <ul style="color:#1d1d1f;font-size:1.05rem;line-height:1.8;margin:0">
            <li>Ranking candidate drugs against a target</li>
            <li>Drug repurposing — scoring approved drugs against novel targets</li>
            <li>Comparing structural analogs against the same protein</li>
            <li>Kinase targets (well represented in BindingDB)</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="gdti-card">
          <h3 style="margin-top:0;font-weight:600">What it's not for</h3>
          <ul style="color:#1d1d1f;font-size:1.05rem;line-height:1.8;margin:0">
            <li>Definitive yes/no claims — always validate top hits in wet lab</li>
            <li>Predicting binding affinity in nM — only probability</li>
            <li>Out-of-domain proteins not represented in BindingDB</li>
            <li>3D-aware questions (uses 2D topology only; can't tell stereoisomers apart)</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="gdti-card">
          <h3 style="margin-top:0;font-weight:600">Where to find inputs</h3>
          <p style="color:#1d1d1f;font-size:1.05rem;line-height:1.6;margin-bottom:0">
            <b>SMILES</b> for any drug → search at
            <a href="https://pubchem.ncbi.nlm.nih.gov" target="_blank" style="color:#0071e3;text-decoration:none">PubChem</a>, copy the "Canonical SMILES".<br>
            <b>Protein sequence</b> → search at
            <a href="https://uniprot.org" target="_blank" style="color:#0071e3;text-decoration:none">UniProt</a>, copy the FASTA (omit the <code>&gt;</code> header line).
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Built with PyTorch · torch-geometric · RDKit · FastAPI · Streamlit · BindingDB"
    )

    st.markdown(
        '<div style="text-align:center;color:#86868b;margin-top:3rem;font-size:0.9rem">© 2026 GraphDTI · Open source · MIT License</div>',
        unsafe_allow_html=True,
    )
