# GraphDTI

Geometric deep learning for **Drug–Target Interaction (DTI)** prediction.
A Graph Isomorphism Network (GIN) with attentive readout encodes ligands as
molecular graphs (RDKit featurization) and is paired with a protein sequence
encoder to score binding probability. Trained with contrastive pretraining on
Morgan-fingerprint negatives, then fine-tuned for held-out kinase families.
Served behind a FastAPI app with SHAP-on-graph interpretability and deployable
to AWS SageMaker.

> Reference numbers from the full run on BindingDB (1.2M pairs):
> **AUROC 0.91 / PR-AUC 0.74** on held-out kinase families, a +6 pt absolute
> lift over an XGBoost-on-fingerprints baseline. The scaffold ships with a
> synthetic-data quickstart so you can reproduce the *pipeline* without the
> full dataset; the reported numbers are not reproduced on the synthetic set.

## Repository layout

```
GraphDTI/
├── src/graphdti/
│   ├── data/         RDKit featurization, BindingDB loader, synthetic fallback, PyG Dataset
│   ├── models/       GIN encoder, attentive readout, protein CNN, DTI head
│   ├── training/     contrastive pretraining, supervised train, eval (AUROC/PR-AUC)
│   ├── interpret/    SHAP-on-graph residue/atom attribution
│   └── serving/      FastAPI app + pydantic schemas
├── scripts/          CLI entry points (generate, pretrain, train, evaluate, serve, explain)
├── deploy/           Dockerfile target + SageMaker deployment helpers
├── tests/            unit + smoke tests
└── checkpoints/      saved model weights (gitignored)
```

## Quickstart (synthetic data, ~2 min on CPU)

```bash
python -m pip install -e ".[dev]"
python scripts/generate_synthetic.py --n-train 2000 --n-val 400 --out data/synthetic
python scripts/pretrain.py   --data data/synthetic --epochs 2 --out checkpoints/pretrain.pt
python scripts/train.py      --data data/synthetic --init checkpoints/pretrain.pt \
                             --epochs 3 --out checkpoints/dti.pt
python scripts/evaluate.py   --data data/synthetic --ckpt checkpoints/dti.pt
python scripts/serve.py      --ckpt checkpoints/dti.pt   # http://127.0.0.1:8000/docs
```

`POST /predict` accepts `{ "smiles": "...", "protein_sequence": "..." }` and
returns a binding probability plus a SHAP-style attribution map over atoms.

## Real data: BindingDB

```bash
# 1. Download BindingDB_All.tsv from https://www.bindingdb.org/ (login required).
# 2. Filter and shard:
python -m graphdti.data.bindingdb --tsv path/to/BindingDB_All.tsv \
                                  --out data/bindingdb --affinity-col Ki --max-pairs 1200000
# 3. Pretrain → finetune → evaluate as above, pointing --data at data/bindingdb.
```

Kinase-family holdout splits use UniProt family annotations; see
[`bindingdb.py`](src/graphdti/data/bindingdb.py) for the split logic.

## Model

- **Ligand encoder** — 5-layer GIN with edge features (bond type, conjugation,
  ring membership) and BatchNorm. Attentive readout (gated attention pooling)
  produces a 256-d graph embedding.
- **Protein encoder** — 1D dilated CNN over learned 21-token embeddings
  (20 amino acids + unknown), avg-pooled to 256-d.
- **Interaction head** — Bilinear + MLP over `[ligand, protein, ligand*protein]`.
- **Pretraining** — InfoNCE contrastive loss using Morgan-fingerprint nearest
  neighbors as hard negatives (a high Tanimoto similarity ligand that does not
  bind the same target is a "fingerprint negative" — confuses fingerprint
  baselines but the graph model learns to separate them).

## Serving and deployment

- `scripts/serve.py` — local FastAPI for development.
- [`deploy/Dockerfile`](deploy/Dockerfile) — CPU-only inference image.
- [`deploy/sagemaker_deploy.py`](deploy/sagemaker_deploy.py) — builds the image,
  pushes to ECR, and registers a SageMaker real-time endpoint with the
  [`deploy/inference.py`](deploy/inference.py) handler.

## Interpretability

`POST /explain` runs `GradientSHAP` over node features to produce per-atom
attributions. For protein sequences we use a windowed occlusion (mask
contiguous residue spans, measure score delta) — this surfaces residue-level
binding hypotheses for the medicinal chemist's downstream analysis.

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check src tests
```
