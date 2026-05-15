"""GraphDTI Explorer — a polished Streamlit UI for the GraphDTI model.

Loads the model checkpoint directly so it works without the FastAPI server.
Forces CPU inference to keep the GPU free.

Run:
    streamlit run scripts/app_ui.py
"""
from __future__ import annotations

import base64
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
    # name: (smiles, category, description)
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
# Styling
# ============================================================

CUSTOM_CSS = """
<style>
    /* Tighter top margin */
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Cards */
    .gdti-card {
        background: rgba(250, 250, 250, 0.04);
        border: 1px solid rgba(120, 120, 120, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .gdti-hero {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(168, 85, 247, 0.08) 100%);
        border: 1px solid rgba(120, 120, 120, 0.15);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
    }
    .gdti-hero h1 {
        font-size: 2.5rem;
        margin: 0 0 0.5rem 0;
        background: linear-gradient(90deg, #3b82f6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }
    .gdti-hero p {
        font-size: 1.1rem;
        color: rgba(160, 160, 170, 1);
        margin: 0;
        line-height: 1.5;
    }
    .gdti-pill {
        display: inline-block;
        padding: 4px 12px;
        font-size: 0.75rem;
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border-radius: 999px;
        margin-right: 6px;
        font-weight: 500;
    }
    .gdti-pill.green { background: rgba(34, 197, 94, 0.15); color: #4ade80; }
    .gdti-pill.amber { background: rgba(245, 158, 11, 0.15); color: #fbbf24; }
    .gdti-pill.red   { background: rgba(239, 68, 68, 0.15); color: #f87171; }

    /* Result block */
    .gdti-result {
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
    }
    .gdti-result.win   { background: rgba(34, 197, 94, 0.10);  border-left: 4px solid #22c55e; }
    .gdti-result.warn  { background: rgba(245, 158, 11, 0.10); border-left: 4px solid #f59e0b; }
    .gdti-result.fail  { background: rgba(239, 68, 68, 0.10);  border-left: 4px solid #ef4444; }
    .gdti-result h2    { margin: 0; font-size: 0.85rem; color: rgba(160, 160, 170, 1); text-transform: uppercase; letter-spacing: 0.05em; }
    .gdti-result .num  { font-size: 3.5rem; font-weight: 800; line-height: 1; margin: 0.3rem 0; }
    .gdti-result.win  .num { color: #22c55e; }
    .gdti-result.warn .num { color: #f59e0b; }
    .gdti-result.fail .num { color: #ef4444; }
    .gdti-result .lbl  { font-size: 1.1rem; font-weight: 600; }

    /* Sequence heatmap */
    .seq-heatmap {
        font-family: 'JetBrains Mono', Consolas, monospace;
        font-size: 0.78rem;
        line-height: 1.7;
        word-break: break-all;
        background: rgba(255, 255, 255, 0.02);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid rgba(120, 120, 120, 0.15);
    }

    /* Mol container */
    .mol-frame {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        display: flex;
        justify-content: center;
        border: 1px solid rgba(120, 120, 120, 0.15);
    }

    /* Helper text */
    .helper { color: rgba(150, 150, 160, 1); font-size: 0.85rem; margin-top: 4px; }
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


def smiles_to_svg(smiles: str, width: int = 320, height: int = 220) -> str | None:
    """Render a SMILES string as an SVG molecule image (white background)."""
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
        svg = drawer.GetDrawingText()
        # Streamlit's HTML renderer is picky about <?xml> headers; strip them
        return svg.replace("<?xml version='1.0' encoding='iso-8859-1'?>\n", "")
    except Exception:
        return None


def probability_gauge(prob: float, threshold: float) -> str:
    """Return an HTML/SVG gauge showing the probability."""
    pct = max(0.0, min(1.0, prob))
    angle = -90 + 180 * pct  # -90 (left) to +90 (right)
    if prob >= 0.75:
        color = "#22c55e"
    elif prob >= threshold:
        color = "#f59e0b"
    else:
        color = "#ef4444"
    # Build arc from -90 (start) to angle (current value)
    import math
    end_x = 100 + 80 * math.cos(math.radians(angle))
    end_y = 100 + 80 * math.sin(math.radians(angle))
    large_arc = 1 if pct > 0.5 else 0
    return f"""
    <div style="display:flex;justify-content:center;margin:1rem 0">
    <svg width="220" height="140" viewBox="0 0 200 130">
      <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="rgba(120,120,120,0.2)" stroke-width="14" stroke-linecap="round"/>
      <path d="M 20 100 A 80 80 0 {large_arc} 1 {end_x:.2f} {end_y:.2f}"
            fill="none" stroke="{color}" stroke-width="14" stroke-linecap="round"
            style="transition:stroke-dashoffset 0.6s ease-out"/>
      <text x="100" y="92" text-anchor="middle" fill="{color}" font-size="32" font-weight="700"
            font-family="system-ui, sans-serif">{prob:.2f}</text>
      <text x="100" y="115" text-anchor="middle" fill="rgba(150,150,160,1)" font-size="10"
            font-family="system-ui, sans-serif">PROBABILITY</text>
    </svg>
    </div>
    """


def render_result_card(prob: float, threshold: float) -> str:
    if prob >= 0.75:
        klass, badge, label = "win",  "🟢", "Strong binder"
    elif prob >= threshold:
        klass, badge, label = "warn", "🟡", "Possible binder (borderline)"
    else:
        klass, badge, label = "fail", "🔴", "Likely non-binder"
    return f"""
    <div class="gdti-result {klass}">
      <h2>Binding probability</h2>
      <div class="num">{prob:.4f}</div>
      <div class="lbl">{badge} {label} <span style="color:rgba(150,150,160,1);font-weight:400">· threshold {threshold:.2f}</span></div>
    </div>
    """


def render_sequence_heatmap(tokens: list[str], scores: list[float]) -> str:
    max_abs = max(max(abs(s) for s in scores), 1e-6)
    spans = []
    for tok, sc in zip(tokens, scores):
        intensity = abs(sc) / max_abs
        if sc >= 0:
            rgb = f"rgba(239, 68, 68, {intensity:.2f})"
        else:
            rgb = f"rgba(59, 130, 246, {intensity:.2f})"
        spans.append(f'<span style="background:{rgb};padding:1px 1px;border-radius:2px">{tok}</span>')
    return f'<div class="seq-heatmap">{"".join(spans)}</div>'


def explain_pair(predictor, smiles: str, protein: str, do_atom: bool, do_residue: bool, occ_window: int):
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
        atom_data = atom_attributions(
            predictor.model, graph, protein_t, steps=16, device=predictor.device
        ).to_dict()
    if do_residue:
        res_data = residue_occlusion(
            predictor.model, graph, protein_t, protein,
            window=occ_window, stride=max(1, occ_window // 2), device=predictor.device,
        ).to_dict()
    return prob, atom_data, res_data


# ============================================================
# Page
# ============================================================

st.set_page_config(page_title="GraphDTI Explorer", page_icon="🧬", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----- Session-state defaults (so widgets are seeded once, then driven by callbacks) -----
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
    choice = st.session_state.pred_drug_choice
    if choice in DRUGS:
        st.session_state.pred_drug_text = DRUGS[choice][0]
    elif choice == "(custom SMILES)":
        st.session_state.pred_drug_text = ""


def _sync_protein_predict():
    choice = st.session_state.pred_prot_choice
    if choice in PROTEINS:
        st.session_state.pred_prot_text = PROTEINS[choice][0]
    elif choice == "(custom sequence)":
        st.session_state.pred_prot_text = ""


def _sync_protein_screen():
    choice = st.session_state.scr_prot_choice
    if choice in PROTEINS:
        st.session_state.scr_prot_text = PROTEINS[choice][0]


# ----- Sidebar -----
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    ckpts = list_checkpoints()
    if not ckpts:
        st.error("No checkpoints found in `checkpoints/`. Train a model first.")
        st.stop()
    ckpt_path = st.selectbox(
        "Model checkpoint",
        ckpts,
        index=0,
        help="Newer/larger checkpoints give better predictions.",
    )
    predictor = get_predictor(ckpt_path)
    threshold = st.slider(
        "Decision threshold",
        0.0, 1.0, 0.5, 0.05,
        help="Probability ≥ threshold → predicted binder. Real BindingDB-trained models work best around 0.42.",
    )

    st.divider()
    st.markdown("**Loaded model**")
    st.caption(f"`{Path(ckpt_path).name}`")
    st.caption(f"Device: `{predictor.device}`")

    st.divider()
    st.markdown("**Probability bands**")
    st.markdown("🟢 ≥ 0.75 — strong binder")
    st.markdown("🟡 0.50 – 0.75 — borderline")
    st.markdown("🔴 < 0.50 — likely non-binder")

    st.divider()
    st.caption(
        "Built on GraphDTI: a Graph Isomorphism Network (drug) + 1D CNN (protein), "
        "trained on 558k measured drug-protein pairs from BindingDB."
    )


# ----- Hero -----
st.markdown(
    """
    <div class="gdti-hero">
      <h1>🧬 GraphDTI Explorer</h1>
      <p>Predict whether a drug binds a protein — instantly, in your browser.</p>
      <div style="margin-top:0.75rem">
        <span class="gdti-pill">GNN + CNN</span>
        <span class="gdti-pill green">AUROC 0.91</span>
        <span class="gdti-pill amber">558k BindingDB pairs</span>
        <span class="gdti-pill">Explainable</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----- Tabs -----
tab_predict, tab_compare, tab_screen, tab_about = st.tabs(
    ["🎯 Predict", "⚖️ Compare drugs", "📊 Screen library", "📖 How it works"]
)


# =========================================================================
# Tab 1: Single prediction
# =========================================================================
with tab_predict:
    st.markdown("##### Pick a drug and a target protein. Get a binding probability and (optionally) an explanation.")

    col_d, col_p = st.columns(2)

    with col_d:
        st.markdown("##### 💊 Drug")
        drug_options = ["(custom SMILES)"] + list(DRUGS.keys())
        st.selectbox(
            "Choose a known drug",
            drug_options,
            key="pred_drug_choice",
            on_change=_sync_drug_predict,
        )
        st.text_input(
            "SMILES (editable)",
            key="pred_drug_text",
            placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O",
        )
        drug_smiles = st.session_state.pred_drug_text
        drug_choice = st.session_state.pred_drug_choice
        if drug_choice in DRUGS:
            _, cat, desc = DRUGS[drug_choice]
            st.markdown(f"_{cat}_ — {desc}")
        if drug_smiles:
            svg = smiles_to_svg(drug_smiles, width=360, height=240)
            if svg:
                st.markdown(f'<div class="mol-frame">{svg}</div>', unsafe_allow_html=True)
            else:
                st.warning("Could not render this SMILES — check the syntax.")

    with col_p:
        st.markdown("##### 🧬 Protein target")
        protein_options = ["(custom sequence)"] + list(PROTEINS.keys())
        st.selectbox(
            "Choose a known target",
            protein_options,
            key="pred_prot_choice",
            on_change=_sync_protein_predict,
        )
        st.text_area(
            "Sequence (editable)",
            key="pred_prot_text",
            height=200,
            placeholder="Paste sequence from UniProt (skip the '>' FASTA header)",
        )
        protein_seq = st.session_state.pred_prot_text
        protein_choice = st.session_state.pred_prot_choice
        if protein_choice in PROTEINS:
            seq, cat, desc = PROTEINS[protein_choice]
            st.markdown(f"_{cat}_ — {desc} · {len(seq)} residues")

    st.divider()
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        do_predict = st.button("🎯 Predict", type="primary", use_container_width=True)
    with c2:
        do_explain = st.button("🔍 Predict & Explain", use_container_width=True)

    if do_predict or do_explain:
        if not drug_smiles or not protein_seq.strip():
            st.error("Both drug SMILES and protein sequence are required.")
        else:
            try:
                with st.spinner("Featurizing molecule and scoring..."):
                    if do_explain:
                        prob, atom_data, res_data = explain_pair(
                            predictor, drug_smiles, protein_seq.strip(),
                            do_atom=True, do_residue=True, occ_window=10
                        )
                    else:
                        prob = predictor.predict(drug_smiles, protein_seq.strip())
                        atom_data, res_data = None, None
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

            # Probability gauge + result card
            colg, colr = st.columns([1, 2])
            with colg:
                st.markdown(probability_gauge(prob, threshold), unsafe_allow_html=True)
            with colr:
                st.markdown(render_result_card(prob, threshold), unsafe_allow_html=True)

            if atom_data:
                st.markdown("#### Atom contributions")
                st.caption(
                    "Per-atom contribution to the prediction. Positive = pushed toward 'binds'. "
                    "The model's reasons for the prediction live here."
                )
                df_a = pd.DataFrame({"atom": atom_data["tokens"], "score": atom_data["scores"]})
                df_a_top = df_a.reindex(df_a["score"].abs().sort_values(ascending=False).index).head(12)
                st.bar_chart(df_a_top.set_index("atom"), height=260, color="#3b82f6")

            if res_data:
                st.markdown("#### Protein region contributions")
                st.caption(
                    "Red = masking this region dropped the score (model relied on it). "
                    "Blue = masking it increased the score (argued *against* binding)."
                )
                st.markdown(
                    render_sequence_heatmap(res_data["tokens"], res_data["scores"]),
                    unsafe_allow_html=True,
                )
                df_r = pd.DataFrame({
                    "position": list(range(len(res_data["scores"]))),
                    "residue": res_data["tokens"],
                    "score": res_data["scores"],
                })
                df_r_top = df_r.reindex(df_r["score"].abs().sort_values(ascending=False).index).head(15)
                with st.expander("Top 15 most-influential residue windows"):
                    st.dataframe(df_r_top, hide_index=True, use_container_width=True)


# =========================================================================
# Tab 2: Compare drugs
# =========================================================================
with tab_compare:
    st.markdown("##### Score 2–6 drugs against the same protein. Useful for picking the most promising binder.")

    protein_choice = st.selectbox(
        "Target protein",
        list(PROTEINS.keys()),
        key="cmp_prot_choice",
        index=2,
    )
    protein_seq = PROTEINS[protein_choice][0]

    default_drugs = ["Vemurafenib (Zelboraf)", "Dabrafenib (Tafinlar)", "Sorafenib (Nexavar)", "Aspirin", "Caffeine"]
    drug_choices = st.multiselect(
        "Drugs to compare (pick 2-6)",
        list(DRUGS.keys()),
        default=default_drugs,
        max_selections=6,
        key="cmp_drug_choices",
    )

    if st.button("⚖️ Compare", type="primary", key="cmp_btn", use_container_width=False):
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

            # Chart
            chart_df = df[["name", "probability"]].dropna().set_index("name")
            st.markdown("#### Predicted binding probability")
            st.bar_chart(chart_df, height=380, color="#3b82f6")

            # Table with structures
            st.markdown("#### Detailed results")
            for _, row in df.iterrows():
                with st.container():
                    cols = st.columns([1.3, 2, 1])
                    with cols[0]:
                        svg = smiles_to_svg(row["smiles"], width=180, height=130)
                        if svg:
                            st.markdown(f'<div class="mol-frame" style="padding:0.3rem">{svg}</div>',
                                        unsafe_allow_html=True)
                    with cols[1]:
                        st.markdown(f"**{row['name']}**")
                        cat_desc = DRUGS[row["name"]]
                        st.caption(f"{cat_desc[1]} — {cat_desc[2]}")
                        if row["probability"] is None:
                            st.write("Could not score.")
                            continue
                        p = row["probability"]
                        if p >= 0.75:
                            badge = '<span class="gdti-pill green">🟢 Strong binder</span>'
                        elif p >= threshold:
                            badge = '<span class="gdti-pill amber">🟡 Borderline</span>'
                        else:
                            badge = '<span class="gdti-pill red">🔴 Likely non-binder</span>'
                        st.markdown(badge, unsafe_allow_html=True)
                    with cols[2]:
                        if row["probability"] is not None:
                            st.metric("Probability", f"{row['probability']:.4f}")


# =========================================================================
# Tab 3: Bulk screen
# =========================================================================
with tab_screen:
    st.markdown("##### Score a large compound library against one target. Output: ranked CSV.")

    st.selectbox(
        "Target protein",
        list(PROTEINS.keys()),
        key="scr_prot_choice",
        on_change=_sync_protein_screen,
    )
    st.text_area("Sequence (editable)", key="scr_prot_text", height=120)
    protein_seq = st.session_state.scr_prot_text

    st.markdown("##### Compound library")
    st.caption("One per line. Format: `name,smiles` or just `smiles`.")

    default_lib = "\n".join([
        f"{n},{DRUGS[n][0]}" for n in
        ["Imatinib (Gleevec)", "Gefitinib (Iressa)", "Erlotinib (Tarceva)",
         "Sorafenib (Nexavar)", "Aspirin", "Caffeine", "Ibuprofen", "Atorvastatin"]
    ])
    library_text = st.text_area("Library", value=default_lib, height=200, key="scr_lib")

    if st.button("🚀 Run screen", type="primary", key="scr_btn"):
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
                    results.append({"name": r["name"], "smiles": r["smiles"],
                                    "probability": None, "error": str(e)[:80]})
                progress.progress(i / len(rows))
            status.empty()
            progress.empty()

            df = pd.DataFrame(results).sort_values("probability", ascending=False, na_position="last")
            df["status"] = df["probability"].apply(
                lambda x: "🟢 BINDER" if x is not None and x >= 0.75
                else ("🟡 borderline" if x is not None and x >= threshold
                      else ("🔴 non-binder" if x is not None else "error"))
            )
            df_view = df.copy()
            df_view["probability"] = df_view["probability"].apply(lambda x: f"{x:.4f}" if x is not None else "—")

            n_binders = sum(1 for x in df["probability"] if x is not None and x >= threshold)
            st.markdown(f"#### Results: **{n_binders} / {len(rows)}** predicted binders")
            st.dataframe(df_view[["name", "status", "probability", "smiles"]],
                         hide_index=True, use_container_width=True)
            st.download_button(
                "⬇️ Download as CSV",
                df.to_csv(index=False),
                file_name="screen_results.csv",
                mime="text/csv",
            )


# =========================================================================
# Tab 4: How it works
# =========================================================================
with tab_about:
    st.markdown("## How GraphDTI works")
    st.markdown(
        """
        ### The problem

        Given a small-molecule drug (as a SMILES string) and a protein target
        (as an amino-acid sequence), predict the probability that the drug
        binds the protein.

        This is the **virtual screening** step of drug discovery: cheaply rank
        millions of candidate molecules before sending the top 100-1000 to a
        wet-lab assay. Wet-lab assays cost $50–500 per molecule and take weeks;
        this model costs essentially nothing and takes milliseconds per pair.
        """
    )

    st.markdown("### The architecture")
    st.markdown(
        """
        **Two encoders → bilinear interaction → MLP head → probability.**

        - **Drug encoder**: a 5-layer **GIN** (Graph Isomorphism Network) with
          edge features. The SMILES is parsed into an atom-bond graph
          (atoms = nodes, bonds = edges, each carrying chemical features),
          and the GIN produces a 256-dimensional embedding of the molecule.
        - **Protein encoder**: a 1-D dilated **CNN** over amino-acid embeddings.
          Captures local sequence motifs (binding-site patches) at multiple
          length scales. Outputs another 256-dim vector.
        - **Interaction head**: a bilinear layer mixes both vectors, then a
          2-layer MLP turns the combined representation into a single
          binding logit.

        The model is trained in two stages:
        1. **Contrastive pretraining** of the GIN encoder using
           Morgan-fingerprint hard negatives.
        2. **Supervised fine-tuning** with binary cross-entropy on labels from
           BindingDB.
        """
    )

    st.markdown("### Training data")
    st.markdown(
        """
        **BindingDB** is a public database of 3.18 million measured binding
        affinities curated from published scientific papers. We filter to
        pairs with valid SMILES + protein + measured Ki, then label each pair
        as a **binder** (Ki ≤ 1 μM) or **non-binder**.

        The current model was trained on **558,571 pairs** covering
        ~244k unique molecules and ~3,000 unique protein targets.
        """
    )

    st.markdown("### Performance")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("AUROC", "0.909", help="Area under ROC curve on 60k held-out pairs.")
    col_b.metric("PR-AUC", "0.960", help="Area under precision-recall curve.")
    col_c.metric("Best F1", "0.904", help="At optimal threshold 0.42.")

    st.markdown("### What it's good at")
    st.markdown(
        """
        - ✅ Ranking candidate drugs against a target (top of the ranking is
          enriched for real binders)
        - ✅ Drug repurposing — scoring approved drugs against novel targets
        - ✅ Comparing structural analogs against the same protein
        - ✅ Kinase targets (well-represented in BindingDB)
        """
    )

    st.markdown("### What it's not for")
    st.markdown(
        """
        - ❌ Definitive yes/no claims — always validate top hits in wet lab
        - ❌ Predicting *how much* a drug binds (binding affinity in nM)
        - ❌ Out-of-domain proteins not represented in BindingDB (e.g., novel
          viral proteins, snake-venom toxins)
        - ❌ 3D-aware questions — the model uses 2D molecular topology only,
          so it can't tell stereoisomers apart or model induced-fit binding
        """
    )

    st.markdown("### Where to find inputs")
    st.markdown(
        """
        - **SMILES** for any drug → search the name on
          [PubChem](https://pubchem.ncbi.nlm.nih.gov), copy the
          "Canonical SMILES" field.
        - **Protein sequence** → search the name on
          [UniProt](https://uniprot.org), copy the FASTA sequence
          (omit the `>` header line).
        """
    )

    st.markdown("### Built with")
    st.caption(
        "PyTorch · torch-geometric · RDKit · FastAPI · Streamlit · BindingDB"
    )
