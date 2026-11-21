PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
DLTF ?= $(PY) -m dltf.cli
NPM ?= npm

.PHONY: help setup setup-cuda test lint train cv hpo analyze export web-dev web-install web-build all clean

help:
	@echo "dl-train-framework"
	@echo ""
	@echo "  make setup       create .venv and install pinned CPU dependencies"
	@echo "  make setup-cuda  same, but with CUDA 12.4 torch wheels"
	@echo "  make test        run pytest (~1 min)"
	@echo "  make lint        ruff check src tests"
	@echo "  make train       train the tabular MLP"
	@echo "  make cv          5-fold cross-validation on the tabular task"
	@echo "  make hpo         Optuna search for the image CNN"
	@echo "  make analyze     error analysis for the latest image run"
	@echo "  make export      ONNX export + web artifacts"
	@echo "  make all         the full pipeline (train, cv, hpo, train, analyze, export)"
	@echo "  make web-dev     npm run dev in web/"
	@echo "  make web-build   production build of the web app"

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
	$(PIP) install -e . --no-deps
	@echo "Done. Use '$(PY)' or 'make test'."

setup-cuda:
	python3 -m venv .venv
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
	$(PIP) install -e . --no-deps

test:
	$(PY) -m pytest

lint:
	.venv/bin/ruff check src tests

train:
	$(DLTF) train --config configs/tabular_mlp.yaml

cv:
	$(DLTF) cv --config configs/tabular_mlp.yaml

hpo:
	$(DLTF) hpo --config configs/hpo_image_cnn.yaml

analyze:
	$(DLTF) analyze --config configs/image_cnn.yaml --task image

export:
	$(DLTF) export --config configs/image_cnn.yaml

web-install:
	cd web && $(NPM) install

web-dev:
	cd web && $(NPM) run dev

web-build:
	cd web && $(NPM) run build

all:
	$(DLTF) all

clean:
	rm -rf runs artifacts .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
