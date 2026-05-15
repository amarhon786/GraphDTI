.PHONY: install dev synth pretrain train eval serve test lint clean

install:
	python -m pip install -e .

dev:
	python -m pip install -e ".[dev,deploy]"

synth:
	python scripts/generate_synthetic.py --n-train 2000 --n-val 400 --out data/synthetic

pretrain:
	python scripts/pretrain.py --data data/synthetic --epochs 2 --out checkpoints/pretrain.pt

train:
	python scripts/train.py --data data/synthetic --init checkpoints/pretrain.pt --epochs 3 --out checkpoints/dti.pt

eval:
	python scripts/evaluate.py --data data/synthetic --ckpt checkpoints/dti.pt

serve:
	python scripts/serve.py --ckpt checkpoints/dti.pt

test:
	pytest -q

lint:
	ruff check src tests

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
